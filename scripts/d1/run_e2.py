#!/usr/bin/env python3
"""Bounded VisDrone acceptance using the existing D1 trainer and exact resume policy."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from scripts.d1.cache_features import cache_contract
from scripts.d1.evaluate_visdrone import export_predictions
from scripts.d1.p1p2_runtime import E1FrozenTrainer, atomic_torch, clean_child_env
from scripts.d1.prepare_visdrone import NAMES, encoded, file_sha, immutable
from scripts.d1.run_wp8_p1_control import LOCKED_TRAIN, write_json
from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
from ultralytics.models.yolo.detect.foundation_val import D1FoundationDetectionValidator
from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader, validate_npy_evidence
from ultralytics.nn.foundation_detection_model import (
    D1_AUX_REPORT_NAMES,
    DEFAULT_D1_MODEL_CFG,
    D1FoundationDetectionModel,
)
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.utils import YAML, ops
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/experiments/d1/p1p2/e2-visdrone.yaml"
EXPECTED = {
    "schema_version": "d1-e2-engineering-v1",
    "dataset": "VisDrone2019-DET",
    "nc": 10,
    "seed": 0,
    "imgsz": 640,
    "schedule_epochs": 300,
    "world_size": 6,
    "per_gpu_batch": 16,
    "global_batch": 96,
    "workers": 4,
    "prefetch_factor": 1,
    "value_fusion_mode": "weighted_sum",
    "balance_loss_coeff": 0.01,
    "router_z_loss_coeff": 0.001,
    "latent_aux_gain": 0.1,
    "mixture_aux_budget": 3.0,
    "amp_init_scale": 16,
    "amp_growth_interval": 1000000,
    "max_det": 500,
    "conf": 0.001,
    "profiles": {
        "smoke": {"train_samples": 96, "val_samples": 8, "window_epochs": 1},
        "smoke-resume": {"train_samples": 96, "val_samples": 8, "window_epochs": 2},
        "benchmark": {"train_samples": 6471, "val_samples": 548, "window_epochs": 3},
    },
}


def load_contract(path=CONFIG):
    contract = YAML.load(path)
    if encoded(contract) != encoded(EXPECTED):
        raise ValueError("Unregistered E2 contract; update and review the bounded experiment first")
    return contract


def source_identity():
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise ValueError("E2 execution requires a clean source commit")
    return {"commit": commit, "contract_sha256": file_sha(CONFIG)}


def model_config():
    config = YAML.load(DEFAULT_D1_MODEL_CFG)
    config["detect"]["nc"] = 10
    config["latent_mixture"].update(value_fusion_mode="weighted_sum", value_fusion_weights=[1.0, 1.0, 1.0])
    return config


def construct_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = D1FoundationDetectionModel(model_config(), verbose=False)
        initialize_mixture_loss_ema_buffer(model)
    model.detect.max_det = 500
    return model


class E2Trainer(E1FrozenTrainer):
    """Reuse finite-update and exact-resume logic with explicitly registered E2 limits."""

    required_schedule_epochs = 300
    required_global_batch = 96
    required_workers = 4

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.detect.max_det = 500
        return model

    def e1_on_train_start(self, trainer):
        super().e1_on_train_start(trainer)
        model = unwrap_model(self.model)
        if model.detect.nc != 10 or model.detect.max_det != 500:
            raise ValueError("E2 requires the registered 10-class, top500 detection head")


def prepare(workspace, data, cache):
    contract, identity = load_contract(), source_identity()
    workspace, data, cache = workspace.resolve(), data.resolve(), cache.resolve()
    if workspace.is_relative_to(ROOT):
        raise ValueError("E2 outputs must be outside the repository")
    data_manifest = json.loads((data / "manifest.json").read_text())
    tracked = json.loads((ROOT / "experiments/d1/manifests/e2-visdrone-data.json").read_text())
    if data_manifest != tracked:
        raise ValueError("VisDrone data differs from the registered E2 preparation")
    ready = {}
    for split, count in (("train", 6471), ("val", 548)):
        name = f"visdrone-{split}"
        ready[split] = validate_npy_evidence(cache / name, cache / "summary.json", name, count)
        if NpyFeatureCacheReader(cache / name).contract != cache_contract(ROOT):
            raise ValueError("Cache no longer matches the frozen Teacher/input contract")
    fs = [
        subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(p)], text=True).strip()
        for p in (data, cache)
    ]
    if any(value in ("nfs", "nfs4", "cifs") for value in fs) or data.stat().st_dev != cache.stat().st_dev:
        raise ValueError("E2 RGB and features must share local storage")
    for split in ("train", "val"):
        if data_manifest["splits"][split]["paths_sha256"] != file_sha(data / f"{split}.txt"):
            raise ValueError("Data split list checksum changed")
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    files = {}
    for profile in ("smoke", "benchmark"):
        cfg = {"path": str(data), "nc": 10, "names": list(NAMES)}
        for split in ("train", "val"):
            rows = (data / f"{split}.txt").read_text().splitlines()
            selected = rows[: contract["profiles"][profile][f"{split}_samples"]]
            paths = [str(data / p) for p in selected]
            dest = inputs / f"{profile}-{split}.txt"
            immutable(dest, ("\n".join(paths) + "\n").encode())
            cfg[split] = str(dest)
            files[dest.name] = file_sha(dest)
        dest = inputs / f"{profile}.yaml"
        immutable(dest, yaml.safe_dump(cfg, sort_keys=False).encode())
        files[dest.name] = file_sha(dest)
    immutable(inputs / "model.yaml", yaml.safe_dump(model_config(), sort_keys=False).encode())
    model = construct_model()
    tensors = {k: v.detach().clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    initial = inputs / "initial.pt"
    if initial.exists():
        saved = torch.load(initial, map_location="cpu", weights_only=True)
        if set(saved) != set(tensors) or any(not torch.equal(saved[k], v) for k, v in tensors.items()):
            raise ValueError("Initial state changed")
    else:
        atomic_torch(initial, tensors)
    files.update({"model.yaml": file_sha(inputs / "model.yaml"), "initial.pt": file_sha(initial)})
    result = {
        "identity": identity,
        "contract": contract,
        "workspace": str(workspace),
        "data": str(data),
        "cache": str(cache),
        "readiness": ready,
        "input_sha256": files,
        "parameters": sum(p.numel() for p in model.parameters()),
        "no_teacher_in_optimizer": not any("teacher" in n.lower() for n, _ in model.named_parameters()),
    }
    immutable(workspace / "matrix.json", encoded(result))
    return result


def matrix(workspace):
    result = json.loads((workspace / "matrix.json").read_text())
    if result["identity"] != source_identity() or result["contract"] != load_contract():
        raise ValueError("E2 matrix belongs to another source/contract")
    for name, digest in result["input_sha256"].items():
        if file_sha(workspace / "inputs" / name) != digest:
            raise ValueError("Prepared runtime input changed")
    return result


def run_spec(mat, profile):
    if profile not in EXPECTED["profiles"]:
        raise ValueError("Only registered engineering profiles may run")
    original = "smoke" if profile == "smoke-resume" else profile
    run_id = f"E2-{original}"
    workspace = Path(mat["workspace"])
    return {
        "identity": {"source": mat["identity"], "run_id": run_id},
        "workspace": str(workspace),
        "run_id": run_id,
        "profile": "benchmark" if original == "benchmark" else "E2-smoke",
        "variant": "B",
        "window": EXPECTED["profiles"][profile]["window_epochs"],
        "seed": 0,
        "parameters": mat["parameters"],
        "output": str(workspace / "reports" / run_id),
        "initial_state": str(workspace / "inputs/initial.pt"),
        "data_profile": original,
    }


def overrides(mat, spec):
    return {
        **deepcopy(LOCKED_TRAIN),
        "epochs": 300,
        "batch": 96,
        "nbs": 96,
        "workers": 4,
        "device": "0,1,2,3,4,5",
        "patience": 300,
        "save_period": 1,
        "conf": 0.001,
        "max_det": 500,
        "latent_aux_gain": 0.1,
        "mixture_aux_budget": 3.0,
        "exist_ok": True,
        "verbose": False,
        "data": str(Path(mat["workspace"]) / "inputs" / f"{spec['data_profile']}.yaml"),
        "model": str(Path(mat["workspace"]) / "inputs/model.yaml"),
        "project": str(Path(mat["workspace"]) / "runs"),
        "name": spec["run_id"],
    }


def train(workspace, profile):
    mat = matrix(workspace)
    if int(os.environ.get("WORLD_SIZE", "0")) != 6 or int(os.environ.get("LOCAL_RANK", "-1")) not in range(6):
        raise ValueError("E2 train must use external six-rank torchrun")
    spec = run_spec(mat, profile)
    common = overrides(mat, spec)
    last = workspace / "runs" / spec["run_id"] / "weights/last.pt"
    if profile == "smoke-resume":
        if not last.is_file() or not (Path(spec["output"]) / "resume.pt").is_file():
            raise FileNotFoundError("E2 smoke checkpoint and exact resume state are required")
        common["resume"] = str(last)
    elif last.exists():
        raise FileExistsError("Do not overwrite an existing engineering run")
    cache = Path(mat["cache"])
    trainer = E2Trainer(
        overrides=common,
        run_spec=spec,
        feature_caches={s: cache / f"visdrone-{s}" for s in ("train", "val")},
        trusted_feature_cache=True,
        feature_prefetch_factor=1,
        amp_init_scale=16,
        amp_growth_interval=1000000,
    )
    trainer.train()


class VisDroneValidator(D1FoundationDetectionValidator):
    """Keep monitoring AP separate and export original-image, zero-based predictions."""

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.nc != 10:
            raise ValueError("VisDrone evaluator requires ten classes")
        self.is_coco = self.is_lvis = False
        self.class_map = list(range(10))
        self.args.save_json = True
        self.degenerate_predictions = 0

    def pred_to_json(self, predn, pbatch):
        boxes = ops.xyxy2xywh(predn["bboxes"])
        boxes[:, :2] -= boxes[:, 2:] / 2
        if any(not torch.isfinite(predn[key]).all() for key in ("bboxes", "conf", "cls")):
            raise FloatingPointError("Nonfinite prediction")
        for box, score, category in zip(boxes.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            if category != int(category) or not 0 <= category < 10 or not 0 <= score <= 1:
                raise ValueError("Invalid VisDrone class or confidence")
            if box[2] <= 0 or box[3] <= 0:
                self.degenerate_predictions += 1
                continue
            self.jdict.append(
                {"image_id": Path(pbatch["im_file"]).stem, "bbox": box, "score": score, "category_id": int(category)}
            )

    def eval_json(self, stats):
        return stats


def evaluate(workspace, profile):
    mat = matrix(workspace)
    spec = run_spec(mat, profile)
    out = workspace / "evaluations" / spec["run_id"]
    checkpoint = workspace / "runs" / spec["run_id"] / "weights/last.pt"
    expected_epoch = 2 if profile == "smoke" else 3
    return evaluate_registered(
        mat,
        {**spec, "evaluation_profile": profile},
        construct_model(),
        checkpoint,
        out,
        expected_epoch,
        overrides(mat, spec),
    )


def evaluate_registered(mat, spec, model, checkpoint, out, expected_epoch, common):
    """Strictly reload a registered VisDrone model and export for the pinned MATLAB scorer."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] + 1 != expected_epoch:
        raise ValueError("Checkpoint did not finish its bounded window")
    stored_model = saved.get("ema") if saved.get("ema") is not None else saved["model"]
    model.load_state_dict(stored_model.float().state_dict(), strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values() if isinstance(value, torch.Tensor)):
        raise FloatingPointError("Nonfinite checkpoint state")
    if any("teacher" in name.lower() for name in model.state_dict()):
        raise ValueError("Teacher found in downstream checkpoint")
    common = deepcopy(common)
    common.update(device="0", batch=32, project=str(out), name="validator", save_json=True)
    cache = Path(mat["cache"])
    trainer = D1FoundationDetectionTrainer(
        overrides=common,
        feature_caches={s: cache / f"visdrone-{s}" for s in ("train", "val")},
        trusted_feature_cache=True,
        feature_prefetch_factor=1,
    )
    trainer.model = model.to(trainer.device).eval()
    trainer.set_model_attributes()
    trainer.amp, trainer.world_size, trainer.epoch, trainer.epochs = True, 1, 0, 1
    trainer.stopper, trainer.ema = SimpleNamespace(possible_stop=True), SimpleNamespace(ema=None)
    trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", *D1_AUX_REPORT_NAMES, "mixture_aux_loss")
    trainer.loss_items = torch.zeros(len(trainer.loss_names), device=trainer.device)
    validator = VisDroneValidator(
        save_dir=out,
        args=copy(trainer.args),
        feature_cache=cache / "visdrone-val",
        trusted_cache=True,
        prefetch_factor=1,
    )
    validator.data, validator.stride = trainer.data, 32
    validator.dataloader = validator.get_dataloader(trainer.data["val"], 32)
    trainer.validator = validator
    started = time.monotonic()
    stats = validator(trainer=trainer)
    paths = Path(trainer.data["val"]).read_text().splitlines()
    ids = [Path(p).stem for p in paths]
    if validator.seen != len(ids) or sorted(Path(p).stem for p in validator.dataloader.dataset.im_files) != sorted(ids):
        raise ValueError("Evaluation coverage mismatch")
    if not all(math.isfinite(float(v)) for v in stats.values()):
        raise FloatingPointError("Nonfinite monitoring metrics")
    exported = export_predictions(validator.jdict, ids, out / "predictions-txt")
    report = {
        "status": "PASSED",
        "identity": mat["identity"],
        "profile": spec.get("evaluation_profile", spec["profile"]),
        "checkpoint_epoch": expected_epoch,
        "checkpoint_sha256": file_sha(checkpoint),
        "strict_reload": True,
        "teacher_parameters": 0,
        "head_max_det": model.detect.max_det,
        "seen": validator.seen,
        "internal_monitoring_metrics": {k: float(v) for k, v in stats.items()},
        "seconds": time.monotonic() - started,
        "degenerate_predictions_dropped": validator.degenerate_predictions,
        "prediction_export_sha256": file_sha(out / "predictions-txt/export.json"),
        "prediction_files": exported["image_count"],
        "official_metrics": None,
        "official_evaluation": "Pending pinned MATLAB; monitoring metrics are not official VisDrone AP",
    }
    write_json(out / "report.json", report)
    return report


def benchmark_summary(workspace):
    mat = matrix(workspace)
    durations, rows, previous_retries = [], [], 0
    for epoch in (1, 2, 3):
        root = workspace / "reports/E2-benchmark"
        validation = json.loads((root / "validation" / f"epoch-{epoch:03d}.json").read_text())
        ranks = [json.loads((root / "epochs" / f"rank-{rank}-epoch-{epoch:03d}.json").read_text()) for rank in range(6)]
        if validation["seen"] != 548 or any(r["batches"] != 68 or r["optimizer_steps"] != epoch * 68 for r in ranks):
            raise ValueError("Full-split benchmark updates or validation coverage mismatch")
        durations.append(validation["epoch_wall_seconds"])
        retries = sum(r["amp_retries"] for r in ranks)
        rows.append(
            {
                "epoch": epoch,
                "wall_seconds": durations[-1],
                "max_rank_data_wait_seconds": max(r["data_wait_seconds"] for r in ranks),
                "peak_gpu_bytes": max(r["peak_allocated_bytes"] for r in ranks),
                "amp_retries": retries - previous_retries,
            }
        )
        previous_retries = retries
    median = statistics.median(durations[1:])
    return {
        "identity": mat["identity"],
        "epochs": rows,
        "timed_epochs": [2, 3],
        "median_epoch_seconds": median,
        "train_images_per_epoch": 6471,
        "validation_images_per_epoch": 548,
        "updates_per_epoch": 68,
        "estimated_60_epoch_hours": median * 60 / 3600,
        "estimated_300_epoch_hours": median * 300 / 3600,
        "estimated_300_epoch_gpu_hours": median * 300 * 6 / 3600,
        "estimate_scope": "Epoch train, internal val and checkpoint; excludes startup, MATLAB official eval and Teacher extraction",
        "limitation": "Two measured early warmup epochs, not a convergence or end-to-end cost guarantee",
    }


def resource_snapshot():
    """Record shared-host pressure without inspecting or changing other workloads."""
    files = (
        "/proc/loadavg",
        "/proc/pressure/cpu",
        "/sys/fs/cgroup/cpu/cpu.stat",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    )
    result = {p: Path(p).read_text().strip() for p in files if Path(p).exists()}
    result["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=15,
    ).strip()
    result["unix_time"] = time.time()
    return result


def regression_command():
    manifest = json.loads((ROOT / "experiments/d1/manifests/e2-regression.json").read_text())
    cases = [*manifest["tests"], "test_d1_e2_engineering.py", "test_d1_p5_ablation.py"]
    return [sys.executable, "-u", "-m", "pytest", "-v", "--color=no", *[f"tests/{name}" for name in cases]]


def run_all(args):
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (args.workspace / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.workspace / "matrix.json").exists():
            raise FileExistsError("Use explicit verified stage commands to recover a partial E2 pipeline")
        (args.workspace / "logs").mkdir(exist_ok=True)
        stages = [
            ("regression", None),
            ("smoke", True),
            ("smoke-resume", True),
            ("smoke", False),
            ("benchmark", True),
            ("benchmark", False),
        ]
        active = None

        def interrupted(*_):
            if active is not None and active.poll() is None:
                os.killpg(active.pid, signal.SIGTERM)
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            write_json(args.workspace / "status.json", {"status": "RUNNING", "phase": "prepare", "pid": os.getpid()})
            prepare(args.workspace, args.data, args.cache)
            for profile, training in stages:
                phase = "regression" if training is None else ("train-" if training else "evaluate-") + profile
                if profile == "smoke-resume":
                    last = args.workspace / "runs/E2-smoke/weights/last.pt"
                    state = args.workspace / "reports/E2-smoke/resume.pt"
                    saved = torch.load(state, map_location="cpu", weights_only=False)
                    if saved["epoch"] != 0 or saved["optimizer_steps"] != 1 or len(saved["ranks"]) != 6:
                        raise ValueError("Smoke did not produce the required first-epoch recovery state")
                    write_json(
                        args.workspace / "resume-input.json",
                        {
                            "epoch": 1,
                            "optimizer_steps": 1,
                            "last_sha256": file_sha(last),
                            "resume_sha256": file_sha(state),
                            "ranks": 6,
                            "state_keys": sorted(saved),
                        },
                    )
                write_json(args.workspace / "status.json", {"status": "RUNNING", "phase": phase, "pid": os.getpid()})
                command = [sys.executable, "-u"]
                if training is None:
                    command = regression_command()
                elif training:
                    command += [
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=6",
                        "--module",
                        "scripts.d1.run_e2",
                    ]
                else:
                    command += ["-m", "scripts.d1.run_e2"]
                if training is not None:
                    command += [
                        "train" if training else "evaluate",
                        "--workspace",
                        str(args.workspace),
                        "--profile",
                        profile,
                    ]
                resources = resource_snapshot()
                if training:
                    devices = [line.split(",") for line in resources["gpu"].splitlines()]
                    if len(devices) != 6 or any(float(row[1]) > 1024 for row in devices):
                        raise RuntimeError("Another GPU workload is active; E2 will not compete for its memory")
                write_json(args.workspace / "logs" / f"{phase}-resources-start.json", resources)
                env = clean_child_env()
                if training is None:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                with (args.workspace / "logs" / f"{phase}.log").open("w") as log:
                    started = time.monotonic()
                    active = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    write_json(args.workspace / "child.json", {"phase": phase, "pid": active.pid, "command": command})
                    code = active.wait()
                write_json(
                    args.workspace / "logs" / f"{phase}-cost.json",
                    {"seconds": time.monotonic() - started, "exit_code": code},
                )
                write_json(args.workspace / "logs" / f"{phase}-resources-end.json", resource_snapshot())
                if code:
                    raise RuntimeError(f"Stage failed: {phase}; inspect its log")
            summary = benchmark_summary(args.workspace)
            write_json(args.workspace / "benchmark-summary.json", summary)
            write_json(
                args.workspace / "status.json",
                {
                    "status": "AWAITING_MATLAB",
                    "phase": "ENGINEERING_PASSED",
                    "note": "Official MATLAB scoring and final evidence audit remain; E3/E4 not launched",
                },
            )
        except BaseException as exc:
            if active is not None and active.poll() is None:
                os.killpg(active.pid, signal.SIGTERM)
                try:
                    active.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL)
                    active.wait(timeout=10)
            write_json(args.workspace / "status.json", {"status": "FAILED", "error": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate", "summarize", "all"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--profile", choices=tuple(EXPECTED["profiles"]), default="smoke")
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    torch.set_num_threads(1)
    if args.mode in ("all", "prepare") and (args.data is None or args.cache is None):
        parser.error("Preparation needs explicit data and cache roots")
    if args.mode == "all":
        run_all(args)
    elif args.mode == "prepare":
        prepare(args.workspace, args.data, args.cache)
    elif args.mode == "train":
        train(args.workspace, args.profile)
    elif args.mode == "evaluate":
        if args.profile == "smoke-resume":
            parser.error("Evaluate the completed smoke run, not its resume alias")
        evaluate(args.workspace, args.profile)
    else:
        write_json(args.workspace / "benchmark-summary.json", benchmark_summary(args.workspace))


if __name__ == "__main__":
    main()

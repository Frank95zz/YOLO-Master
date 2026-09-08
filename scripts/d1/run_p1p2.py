"""Prepare and run the explicitly approved E0/E1 COCO matrix; never auto-expand to E2."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.d1 import run_wp8_p1_control as scratch
from scripts.d1.cache_features import split_paths
from scripts.d1.diagnose_wp8 import DiagnosticValidator
from scripts.d1.launch_wp8_p1 import OfficialValidator, official_metrics
from scripts.d1.p1p2_data import nvme_path
from scripts.d1.p1p2_runtime import (
    E1FrozenTrainer,
    E1ScratchTrainer,
    atomic_torch,
    clean_child_env,
    load_initial_tensors,
)
from scripts.d1.run_wp8_train import code_fingerprint
from ultralytics.nn.foundation.cache import sha256_file
from ultralytics.nn.foundation_detection_model import D1_AUX_REPORT_NAMES, D1FoundationDetectionModel
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/experiments/d1/p1p2/e1-coco2017.yaml"
FROZEN_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml"
VARIANTS = {
    "A": {"value_fusion_mode": "router_only", "latent_aux_gain": 0.1},
    "B": {"value_fusion_mode": "weighted_sum", "latent_aux_gain": 0.1},
    "C": {"value_fusion_mode": "weighted_sum", "latent_aux_gain": 0.0},
    "S": {"value_fusion_mode": "none", "latent_aux_gain": 0.0},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def identity():
    """Bind runs to clean source code and a separately hashed immutable contract."""
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("Commit code before preparing or launching an experiment")
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": code_fingerprint(ROOT),
        "contract_sha256": sha256_file(CONFIG),
    }


def load_contract(path=CONFIG):
    value = YAML.load(path)
    expected = {
        "schema_version": "d1-p1p2-e1-v1",
        "dataset": "coco2017",
        "nc": 80,
        "seed": 0,
        "imgsz": 640,
        "schedule_epochs": 100,
        "screen_fraction": 0.5,
        "screen_epochs": 50,
        "world_size": 6,
        "per_gpu_batch": 64,
        "global_batch": 384,
        "workers": 4,
        "prefetch_factor": 1,
        "save_period": 5,
        "official_eval_period": 5,
        "amp_init_scale": 16,
        "amp_growth_interval": 1000000,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "mixture_aux_budget": 3.0,
        "balance_loss_coeff": 0.01,
        "router_z_loss_coeff": 0.001,
        "max_parameter_delta": 0.01,
        "selection_tie_ap": 0.005,
        "pause_retention": 0.8,
        "variants": VARIANTS,
    }
    if value != expected or any(type(value.get(k)) is not int for k in ("nc", "seed", "screen_epochs", "world_size")):
        raise ValueError("Unknown or modified E1 contract; register a new protocol rather than bypassing this guard")
    return value


def verify_source_identity(recorded, current, revision=None):
    """Require unchanged code or an explicitly recorded, hash-bound diagnostic repair."""
    if recorded["contract_sha256"] != current["contract_sha256"]:
        raise ValueError("Experiment contract changed since preparation")
    reference = recorded
    if recorded["source_sha256"] != current["source_sha256"]:
        if (
            not revision
            or revision.get("schema_version") != "d1-e1-diagnostic-revision-v1"
            or revision.get("approved") is not True
            or revision.get("original_identity") != recorded
        ):
            raise ValueError("Source identity changed without an approved diagnostic revision")
        reference = revision["execution_identity"]
        if any(reference[key] != current[key] for key in ("source_sha256", "contract_sha256")):
            raise ValueError("Current source differs from the approved diagnostic revision")
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", recorded["commit"], reference["commit"], "--", "ultralytics", "scripts/d1"],
            cwd=ROOT,
            text=True,
        ).splitlines()
        if not changed or not set(changed) <= {"scripts/d1/p1p2_runtime.py", "scripts/d1/run_p1p2.py"}:
            raise ValueError("Diagnostic revision also changed model, data, configuration or other training sources")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", recorded["commit"], reference["commit"]], cwd=ROOT, check=False
        ).returncode:
            raise ValueError("Diagnostic revision is not a descendant of the original source")
    if reference["commit"] != current["commit"]:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", reference["commit"], current["commit"]], cwd=ROOT, check=False
        )
        if result.returncode != 0:
            raise ValueError("Current checkout is not a descendant of the recorded code commit")


def model_config(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown E1 variant")
    if variant == "S":
        return YAML.load(ROOT / scratch.MODEL)
    config = YAML.load(FROZEN_MODEL)
    selected = VARIANTS[variant]
    config["latent_mixture"]["value_fusion_mode"] = selected["value_fusion_mode"]
    config["latent_mixture"]["value_fusion_weights"] = [1.0, 1.0, 1.0]
    config["loss"]["latent_aux_gain"] = selected["latent_aux_gain"]
    return config


def construct_model(variant):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = (
            DetectionModel(model_config(variant), nc=80, ch=3, verbose=False)
            if variant == "S"
            else (D1FoundationDetectionModel(model_config(variant)))
        )
        if variant != "S":
            initialize_mixture_loss_ema_buffer(model)
    return model


def prepare(workspace: Path):
    contract = load_contract()
    source = identity()
    receipt_path = workspace / "data/receipt.json"
    receipt = read_json(receipt_path)
    if receipt["status"] != "passed" or sha256_file(workspace / "data/files.json") != receipt["files_sha256"]:
        raise ValueError("Verified RGB copy receipt is missing or changed")
    nvme_root = Path(receipt["nvme_root"])
    data_root = nvme_path(Path(receipt["data_root"]), nvme_root)
    cache_root = nvme_path(Path(receipt["cache_root"]), nvme_root)
    for split in ("train2017", "val2017"):
        if sha256_file(cache_root / f"{split}-index.json") != receipt["splits"][split]["cache_index_sha256"]:
            raise ValueError("Cache index changed after full verification")
        if sha256_file(cache_root / f"{split}-samples.jsonl") != receipt["splits"][split]["samples_manifest_sha256"]:
            raise ValueError("Cache samples manifest changed after full verification")
    inputs = workspace / "inputs"
    inputs.mkdir(exist_ok=False)
    coco = YAML.load(ROOT / "ultralytics/cfg/datasets/coco.yaml")
    data_files = {}
    for profile, limit in (("E0", 32), ("benchmark", None), ("E1", None)):
        lists = {}
        for split, role in (("train2017", "train"), ("val2017", "val")):
            relative, _ = split_paths(ROOT, split, limit)
            paths = [str(nvme_path(data_root / path, nvme_root)) for path in relative]
            listing = inputs / f"{profile}-{split}.txt"
            listing.write_text("\n".join(paths) + "\n")
            lists[role] = str(listing)
        data_yaml = inputs / f"{profile}-coco2017.yaml"
        YAML.save(data_yaml, {"path": str(data_root), **lists, "names": coco["names"]})
        data_files[profile] = str(data_yaml)
    models, counts = {}, {}
    common_state = None
    for variant in VARIANTS:
        model = construct_model(variant)
        if variant == "A":
            common_state = {k: v.detach().clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
            atomic_torch(inputs / "initial-frozen.pt", common_state)
        elif variant != "S":
            load_initial_tensors(model, common_state)
        else:
            atomic_torch(inputs / "initial-scratch.pt", model.state_dict())
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected = 3510624 if variant == "S" else 3542567
        if count != expected:
            raise ValueError(f"Unexpected {variant} parameter count {count}")
        counts[variant] = count
        path = inputs / f"model-{variant}.yaml"
        YAML.save(path, model_config(variant))
        models[variant] = str(path)
    if abs(counts["S"] / counts["A"] - 1) > contract["max_parameter_delta"]:
        raise ValueError("Scratch parameter budget does not match")
    file_hashes = {p.name: sha256_file(p) for p in inputs.iterdir() if p.is_file()}
    matrix = {
        "schema_version": contract["schema_version"],
        "identity": source,
        "contract": contract,
        "workspace": str(workspace),
        "data_receipt_sha256": sha256_file(receipt_path),
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "nvme_root": str(nvme_root),
        "models": models,
        "parameters": counts,
        "data_files": data_files,
        "input_hashes": file_hashes,
    }
    scratch.write_json(workspace / "matrix.json", matrix)
    return matrix


def load_matrix(workspace, *, check_code=True):
    matrix = read_json(workspace / "matrix.json")
    if matrix["contract"] != load_contract():
        raise ValueError("Matrix contract changed")
    if check_code:
        current = identity()
        revision_path = workspace / "execution-revision.json"
        revision = read_json(revision_path) if revision_path.exists() else None
        verify_source_identity(matrix["identity"], current, revision)
        matrix["execution_identity"] = current
    if matrix["data_receipt_sha256"] != sha256_file(workspace / "data/receipt.json"):
        raise ValueError("Data verification identity changed")
    receipt = read_json(workspace / "data/receipt.json")
    for key in ("data_root", "cache_root", "nvme_root"):
        if matrix[key] != receipt[key]:
            raise ValueError("Matrix paths differ from the verified data receipt")
        nvme_path(Path(matrix[key]), Path(receipt["nvme_root"]))
    if Path(matrix["workspace"]).resolve() != workspace.resolve():
        raise ValueError("Workspace identity differs")
    for name, digest in matrix["input_hashes"].items():
        if sha256_file(workspace / "inputs" / name) != digest:
            raise ValueError(f"Prepared input changed: {name}")
    return matrix


def spec_for(matrix, profile, variant, window=None):
    if profile not in {"E0", "E0-resume", "benchmark", "E1"} or variant not in VARIANTS:
        raise ValueError("Unregistered profile/variant")
    expected = 50 if profile == "E1" else 2 if profile == "E0-resume" else 1
    if window is not None and window != expected:
        raise ValueError("Window cannot be overridden")
    original_profile = "E0" if profile == "E0-resume" else profile
    run_id = f"{original_profile}-{variant}"
    workspace = Path(matrix["workspace"])
    return {
        "identity": {
            "source": matrix["identity"],
            "run_id": run_id,
            "variant": variant,
            "data_receipt_sha256": matrix["data_receipt_sha256"],
        },
        "workspace": str(workspace),
        "run_id": run_id,
        "execution_identity": matrix.get("execution_identity", matrix["identity"]),
        "profile": original_profile,
        "variant": variant,
        "window": expected,
        "seed": 0,
        "parameters": matrix["parameters"][variant],
        "output": str(workspace / "reports" / run_id),
        "initial_state": str(workspace / "inputs" / ("initial-scratch.pt" if variant == "S" else "initial-frozen.pt")),
    }


def overrides_for(matrix, spec):
    return {
        **deepcopy(scratch.LOCKED_TRAIN),
        "data": matrix["data_files"][spec["profile"]],
        "model": matrix["models"][spec["variant"]],
        "workers": 4,
        "save_period": 5,
        "device": "0,1,2,3,4,5",
        "project": str(Path(spec["workspace"]) / "runs"),
        "name": spec["run_id"],
        "exist_ok": True,
        "verbose": False,
        "latent_aux_gain": VARIANTS[spec["variant"]]["latent_aux_gain"],
        "mixture_aux_budget": 3.0,
    }


def worker(args):
    matrix = load_matrix(args.workspace)
    if int(os.environ.get("WORLD_SIZE", "0")) != 6 or int(os.environ.get("LOCAL_RANK", "-1")) not in range(6):
        raise ValueError("Only externally launched six-rank torchrun is supported")
    spec = spec_for(matrix, args.profile, args.variant)
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    overrides = overrides_for(matrix, spec)
    if args.profile == "E0-resume" or args.resume:
        overrides["resume"] = str(args.workspace / "runs" / spec["run_id"] / "weights/last.pt")
        if not (output / "resume.pt").is_file():
            raise FileNotFoundError("Exact online resume state is required")
    elif (
        (output / "training-result.json").exists()
        or (output / "resume.pt").exists()
        or (args.workspace / "runs" / spec["run_id"] / "weights/last.pt").exists()
    ):
        raise FileExistsError("Run already contains training state; use explicit validated resume")
    if spec["variant"] == "S":
        trainer = E1ScratchTrainer(overrides=overrides, run_spec=spec)
    else:
        cache = Path(matrix["cache_root"])
        trainer = E1FrozenTrainer(
            overrides=overrides,
            run_spec=spec,
            feature_caches={"train": cache / "train2017", "val": cache / "val2017"},
            trusted_feature_cache=True,
            max_open_feature_shards=4,
            feature_prefetch_factor=1,
            amp_init_scale=16,
            amp_growth_interval=1000000,
        )
    trainer.train()


class FrozenOfficialValidator(DiagnosticValidator):
    def eval_json(self, stats):
        return stats


def evaluate(args):
    matrix = load_matrix(args.workspace)
    profile, variant = args.run_id.split("-")
    spec = spec_for(matrix, profile, variant)
    return evaluate_registered(args, matrix, spec, construct_model(variant), overrides_for(matrix, spec))


def evaluate_registered(args, matrix, spec, model, common):
    """Strictly evaluate a registered model with the shared COCO protocol."""
    profile, variant = spec["profile"], spec["variant"]
    expected_root = args.workspace / "runs" / args.run_id / "weights"
    checkpoint_path = args.checkpoint.resolve()
    if checkpoint_path.parent != expected_root.resolve():
        raise ValueError("Evaluation checkpoint is outside the registered run")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("ema")
    if saved is None:
        saved = checkpoint.get("model")
    model = model.eval()
    model.load_state_dict(saved.float().state_dict(), strict=True)
    if not all(
        torch.isfinite(tensor).all() for tensor in model.state_dict().values() if isinstance(tensor, torch.Tensor)
    ):
        raise FloatingPointError("Checkpoint contains non-finite values")
    args.output.mkdir(parents=True, exist_ok=True)
    common = deepcopy(common)
    common.update(
        device="0",
        batch=128,
        workers=4,
        project=str(args.output),
        name="eval",
        save_json=True,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=False,
    )
    # Both paths use the trainer validation protocol and FP32 weights under AMP.
    if variant == "S":
        trainer = scratch.ScratchTrainer(overrides=common)
    else:
        from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer

        cache = Path(matrix["cache_root"])
        trainer = D1FoundationDetectionTrainer(
            overrides=common,
            feature_caches={"train": cache / "train2017", "val": cache / "val2017"},
            trusted_feature_cache=True,
            feature_prefetch_factor=1,
        )
    trainer.model = model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.model.eval()
    trainer.amp, trainer.world_size, trainer.epoch, trainer.epochs = True, 1, 0, 1
    trainer.stopper = SimpleNamespace(possible_stop=True)
    trainer.ema = SimpleNamespace(ema=None)
    trainer.loss_names = (
        ("box_loss", "cls_loss", "dfl_loss")
        if variant == "S"
        else ("box_loss", "cls_loss", "dfl_loss", *D1_AUX_REPORT_NAMES, "mixture_aux_loss")
    )
    trainer.loss_items = torch.zeros(len(trainer.loss_names), device=trainer.device)
    extra = (
        {}
        if variant == "S"
        else {"feature_cache": Path(matrix["cache_root"]) / "val2017", "trusted_cache": True, "prefetch_factor": 1}
    )
    validator_cls = OfficialValidator if variant == "S" else FrozenOfficialValidator
    validator = validator_cls(save_dir=args.output, args=copy(trainer.args), **extra)
    validator.data = trainer.data
    validator.stride = 32
    validator.dataloader = validator.get_dataloader(trainer.data["val"], 128)
    trainer.validator = validator
    started = time.monotonic()
    internal = validator(trainer=trainer)
    listing = Path(matrix["data_files"][profile])
    ids = sorted(int(Path(path).stem) for path in Path(YAML.load(listing)["val"]).read_text().splitlines())
    seen = sorted(int(Path(path).stem) for path in validator.dataloader.dataset.im_files)
    if seen != ids or validator.seen != len(ids):
        raise RuntimeError("Evaluation image coverage mismatch")
    prediction_path = args.output / "predictions.json"
    scratch.write_json(prediction_path, validator.jdict)
    official = official_metrics(Path(matrix["data_root"]) / "annotations/instances_val2017.json", prediction_path, ids)
    result = {
        "status": "passed",
        "identity": matrix["identity"],
        "run_id": args.run_id,
        "execution_identity": matrix.get("execution_identity", matrix["identity"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "strict_reload": True,
        "seen": len(ids),
        "official": official,
        "internal": {k: float(v) for k, v in internal.items()},
        "seconds": time.monotonic() - started,
        "prediction_sha256": sha256_file(prediction_path),
        "teacher_parameters": 0,
    }
    scratch.write_json(args.output / "report.json", result)
    if hasattr(validator.dataloader, "close"):
        validator.dataloader.close()
    return result


def summarize(workspace):
    matrix = load_matrix(workspace)
    results = {}
    for variant in VARIANTS:
        output = workspace / "reports" / f"E1-{variant}"
        training = read_json(output / "training-result.json")
        final = read_json(output / "official/epoch-050/report.json")
        if training["status"] != "completed" or training["epochs"] != 50 or final["seen"] != 5000:
            raise ValueError("Cannot summarize incomplete or different-budget runs")
        if final["identity"] != matrix["identity"] or final["status"] != "passed":
            raise ValueError("Mixed source identity")
        last = workspace / "runs" / f"E1-{variant}" / "weights/last.pt"
        if training["last_sha256"] != sha256_file(last) or final["checkpoint_sha256"] != training["last_sha256"]:
            raise ValueError("Final result is not bound to the completed checkpoint")
        final_reports = {}
        for name in ("last", "standard-best"):
            final_report = read_json(output / "final" / name / "report.json")
            checkpoint = last.with_name(f"{name}.pt")
            validate_evaluation(final_report, matrix, f"E1-{variant}", checkpoint)
            final_reports[name] = final_report
        results[variant] = {
            "AP": final["official"]["AP_all"],
            "training": training,
            "evaluation": final,
            "cost": read_json(output / "cost.json"),
            "final_evaluations": final_reports,
        }
    best_ap = max(results[key]["AP"] for key in "ABC")
    ties = [key for key in "ABC" if best_ap - results[key]["AP"] <= 0.005]
    selected = min(ties, key=lambda key: (results[key]["cost"]["seconds"], key))
    baseline = results["S"]["AP"]
    if baseline <= 0:
        raise ValueError("Scratch AP must be positive to compute retention")
    retention = results[selected]["AP"] / baseline
    report = {
        "status": "completed",
        "identity": matrix["identity"],
        "results": results,
        "execution_identity": matrix.get("execution_identity", matrix["identity"]),
        "selected": selected,
        "fusion": VARIANTS[selected]["value_fusion_mode"],
        "retention": retention,
        "pause_full_coco": retention < 0.8,
        "scope": "E1 single-seed screening only; not a P1/P2 success claim",
    }
    scratch.write_json(workspace / "E1-summary.json", report)
    return report


def validate_evaluation(report, matrix, run_id, checkpoint):
    """Only reuse a completed evaluation of the exact registered checkpoint."""
    if (
        report.get("status") != "passed"
        or report.get("seen") != 5000
        or report.get("identity") != matrix["identity"]
        or report.get("run_id") != run_id
        or report.get("strict_reload") is not True
        or report.get("checkpoint_sha256") != sha256_file(checkpoint)
    ):
        raise ValueError("Existing final evaluation has a different checkpoint or protocol")


class Pipeline:
    """Run one registered sequential matrix and publish durable status and child identities."""

    def __init__(self, args):
        self.args, self.workspace = args, args.workspace
        self.child = None
        self.interrupted = False
        self.started = time.time()
        self.workspace.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

    def stop(self, *_):
        self.interrupted = True
        if self.child and self.child.poll() is None:
            # torchrun remains alive; its workers finish a full epoch and checkpoint.
            import psutil

            for process in psutil.Process(self.child.pid).children(recursive=True):
                if "scripts.d1.run_p1p2" in " ".join(process.cmdline()) and "worker" in process.cmdline():
                    process.send_signal(signal.SIGUSR1)

    def status(self, phase, **extra):
        scratch.write_json(
            self.workspace / "status.json",
            {
                "status": "running",
                "phase": phase,
                "pid": os.getpid(),
                "started_unix": self.started,
                "updated_unix": time.time(),
                **extra,
            },
        )

    def child_run(self, label, command):
        log_path = self.workspace / "logs" / f"{label}.log"
        log_path.parent.mkdir(exist_ok=True)
        start = time.monotonic()
        started_unix = time.time()
        job_path = self.workspace / "jobs" / f"{label}-{time.time_ns()}.json"
        result = None
        try:
            with log_path.open("a") as log:
                self.child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=clean_child_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.status(label, child_pid=self.child.pid, command=command, log=str(log_path))
                result = self.child.wait()
        finally:
            scratch.write_json(
                job_path,
                {
                    "label": label,
                    "started_unix": started_unix,
                    "ended_unix": time.time(),
                    "seconds": time.monotonic() - start,
                    "returncode": result,
                    "execution_identity": self.matrix.get("execution_identity", self.matrix["identity"]),
                },
            )
        if result or self.interrupted:
            raise RuntimeError(f"{label} exited {result}; interrupted={self.interrupted}")
        return time.monotonic() - start

    def train_command(self, profile, variant):
        return [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=6",
            "--module",
            "scripts.d1.run_p1p2",
            "worker",
            "--workspace",
            str(self.workspace),
            "--profile",
            profile,
            "--variant",
            variant,
        ]

    def execute(self):
        matrix = load_matrix(self.workspace)
        self.matrix = matrix
        if not self.args.approved:
            raise ValueError("Explicit --approved is required for this matrix")
        if self.args.command == "engineering":
            rows = {}
            for variant in VARIANTS:
                self.child_run(f"E0-{variant}", self.train_command("E0", variant))
                output = self.workspace / "reports" / f"E0-{variant}"
                report = read_json(output / "training-result.json")
                if report["epochs"] != 1 or report["optimizer_steps"] != 1:
                    raise ValueError("32-image engineering run did not make exactly one optimizer update")
                checkpoint = self.workspace / "runs" / f"E0-{variant}" / "weights/last.pt"
                self.child_run(
                    f"E0-eval-{variant}",
                    [
                        sys.executable,
                        "-m",
                        "scripts.d1.run_p1p2",
                        "evaluate",
                        "--workspace",
                        str(self.workspace),
                        "--run-id",
                        f"E0-{variant}",
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(output / "evaluation"),
                    ],
                )
                rows[variant] = read_json(output / "evaluation/report.json")
            self.child_run("E0-A-resume", self.train_command("E0-resume", "A"))
            resumed = read_json(self.workspace / "reports/E0-A/training-result.json")
            if resumed["epochs"] != 2 or resumed["optimizer_steps"] != 2:
                raise ValueError("Engineering exact resume failed")
            scratch.write_json(
                self.workspace / "E0-engineering.json",
                {
                    "status": "passed",
                    "identity": matrix["identity"],
                    "variants": rows,
                    "resume": resumed,
                },
            )
            return
        engineering = read_json(self.workspace / "E0-engineering.json")
        if engineering["status"] != "passed" or engineering["identity"] != matrix["identity"]:
            raise ValueError("Complete E0 before benchmarking or E1")
        if self.args.command == "benchmark":
            seconds = {}
            for variant in VARIANTS:
                seconds[variant] = self.child_run(f"benchmark-{variant}", self.train_command("benchmark", variant))
            estimate = sum(seconds.values()) * 50
            report = {
                "status": "passed",
                "identity": matrix["identity"],
                "epoch_job_seconds": seconds,
                "estimated_E1_seconds": estimate,
                "check_interval_seconds": estimate / 4,
                "limitations": "Includes startup per benchmark; periodic standard evaluations are estimated separately",
            }
            # Ten independent standard evaluations per run, conservatively budgeted at 90 seconds each.
            report["estimated_E1_seconds"] += 4 * 12 * 90
            report["check_interval_seconds"] = report["estimated_E1_seconds"] / 4
            scratch.write_json(self.workspace / "benchmark.json", report)
            return
        benchmark = read_json(self.workspace / "benchmark.json")
        if benchmark["identity"] != matrix["identity"] or benchmark["status"] != "passed":
            raise ValueError("Benchmark identity mismatch")
        launch_path = self.workspace / ("E1-resume-launch.json" if self.args.resume else "E1-launch.json")
        if launch_path.exists():
            scratch.write_json(
                self.workspace / "launches" / f"{launch_path.stem}-{time.time_ns()}.json", read_json(launch_path)
            )
        estimate = benchmark["estimated_E1_seconds"]
        if self.args.resume:
            plan = read_json(self.workspace / "resume-plan.json")
            if plan["identity"] != matrix["identity"] or plan["execution_identity"] != matrix["execution_identity"]:
                raise ValueError("Resume plan is not bound to the approved execution revision")
            estimate = plan["estimated_remaining_seconds"]
        scratch.write_json(
            launch_path,
            {
                "identity": matrix["identity"],
                "started_unix": time.time(),
                "estimated_seconds": estimate,
                "check_interval_seconds": estimate / 4,
                "execution_identity": matrix.get("execution_identity", matrix["identity"]),
                "resume": self.args.resume,
                "variants": list(VARIANTS),
            },
        )
        for variant in VARIANTS:
            self.finish_variant(matrix, variant)
        summarize(self.workspace)

    def finish_variant(self, matrix, variant):
        run_id = f"E1-{variant}"
        output = self.workspace / "reports" / run_id
        weights = self.workspace / "runs" / run_id / "weights"
        result_path = output / "training-result.json"
        result = read_json(result_path) if result_path.exists() else None
        completed = result is not None and result.get("status") == "completed" and result.get("epochs") == 50
        existing = (weights / "last.pt").exists() or (output / "resume.pt").exists() or result is not None
        if existing and not self.args.resume:
            raise FileExistsError("E1 already has state; explicitly use all --resume")
        if not completed:
            command = self.train_command("E1", variant)
            if existing:
                if not (weights / "last.pt").is_file() or not (output / "resume.pt").is_file():
                    raise FileNotFoundError("Both last.pt and resume.pt are required")
                state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
                if state["identity"] != spec_for(matrix, "E1", variant)["identity"] or not 0 <= state["epoch"] < 49:
                    raise ValueError("Resume state identity/epoch requires explicit recovery before training")
                del state
                command.append("--resume")
            self.child_run(run_id, command)
            result = read_json(result_path)
        if (
            result.get("status") != "completed"
            or result.get("epochs") != 50
            or result.get("optimizer_steps") != 50 * 309
        ):
            raise ValueError("E1 run did not complete the fixed 50-epoch budget")
        if result["last_sha256"] != sha256_file(weights / "last.pt") or result["resume_sha256"] != sha256_file(
            output / "resume.pt"
        ):
            raise ValueError("Completed run checkpoints changed")
        segments = [read_json(path) for path in (self.workspace / "jobs").glob(f"{run_id}-*.json")]
        segments = [row for row in segments if row["label"] == run_id]
        if any(not math.isfinite(row["seconds"]) or row["seconds"] < 0 for row in segments):
            raise ValueError("Invalid job cost segment")
        if segments:
            seconds = sum(row["seconds"] for row in segments)
            scratch.write_json(
                output / "cost.json",
                {
                    "seconds": seconds,
                    "reserved_gpus": 6,
                    "GPUh": seconds * 6 / 3600,
                    "segments": len(segments),
                    "scope": "All recorded training attempts; excludes downtime and final evaluations",
                },
            )
        elif not (output / "cost.json").exists():
            raise ValueError("No auditable training cost is available")
        for name in ("last", "standard-best"):
            final_output = output / "final" / name
            report_path = final_output / "report.json"
            if report_path.exists():
                validate_evaluation(read_json(report_path), matrix, run_id, weights / f"{name}.pt")
                continue
            self.child_run(
                f"E1-{variant}-final-{name}",
                [
                    sys.executable,
                    "-m",
                    "scripts.d1.run_p1p2",
                    "evaluate",
                    "--workspace",
                    str(self.workspace),
                    "--run-id",
                    run_id,
                    "--checkpoint",
                    str(weights / f"{name}.pt"),
                    "--output",
                    str(final_output),
                ],
            )
            validate_evaluation(read_json(report_path), matrix, run_id, weights / f"{name}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "engineering", "benchmark", "all", "worker", "evaluate", "summarize")
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--profile", choices=("E0", "E0-resume", "benchmark", "E1"), default="E1")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="A")
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    if args.workspace == ROOT or ROOT in args.workspace.parents:
        raise ValueError("Work products must stay outside Git")
    torch.set_num_threads(1)
    if args.command == "prepare":
        prepare(args.workspace)
    elif args.command == "worker":
        worker(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "summarize":
        summarize(args.workspace)
    else:
        pipeline = Pipeline(args)
        try:
            pipeline.execute()
            scratch.write_json(
                args.workspace / "status.json",
                {
                    "status": "completed",
                    "phase": args.command,
                    "updated_unix": time.time(),
                },
            )
        except Exception as exc:
            scratch.write_json(
                args.workspace / "status.json",
                {
                    "status": "interrupted" if pipeline.interrupted else "failed",
                    "phase": args.command,
                    "error": repr(exc),
                    "updated_unix": time.time(),
                },
            )
            raise


if __name__ == "__main__":
    main()

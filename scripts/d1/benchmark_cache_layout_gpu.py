#!/usr/bin/env python3
"""Isolated two-GPU D1 cache-layout experiment; never a full-training ETA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts.d1.benchmark_cache_layout import (
    GIB,
    NAMES,
    SHAPE,  # noqa: F401 - Re-exported for cache-layout consumers.
    guard_memory,
    json_digest,
    memory_snapshot,
    read_npy,
    sample_path,
    validate_array,
    write_json,
)

REPO = Path(__file__).resolve().parents[2]


def case_grid():
    return [(workers, backend) for workers in (0, 2, 4) for backend in ("shard", "npy", "npy", "shard")]


class NpyReader:
    """Prototype reader preserving the original dataset's metadata and finite-check policy."""

    def __init__(self, base, root):
        self.base, self.root = base, Path(root)
        self.records, self.contract, self.index = base.records, base.contract, base.index

    def get(self, sample_id, *, device="cpu"):
        import torch

        if str(device) != "cpu":
            raise ValueError("NPY prototype only reads CPU tensors")
        return {name: torch.from_numpy(value) for name, value in read_npy(sample_path(self.root, sample_id)).items()}

    def close(self):
        self.base.close()


def parameter_digest(model):
    result = hashlib.sha256()
    for name, value in model.named_parameters():
        result.update(name.encode())
        result.update(value.detach().contiguous().cpu().numpy().tobytes())
    return result.hexdigest()


def aggregate_case(reports):
    if sorted(report["rank"] for report in reports) != [0, 1]:
        raise ValueError("Exactly two rank reports required")
    if any(len({report[key] for report in reports}) != 1 for key in ("backend", "workers")):
        raise ValueError("Rank configurations differ")
    for report in reports:
        if report["training_steps"] != 64 or report["optimizer_updates"] != 64 or report["measured_steps"] != 48:
            raise ValueError("Missing training or optimizer steps")
        if not report["weights_changed"] or not report["finite_losses"]:
            raise ValueError("Invalid training state")
    train_seconds = max(report["training_seconds"] for report in reports)
    loader_seconds = max(report["loader_seconds"] for report in reports)
    return {
        "backend": reports[0]["backend"],
        "workers": reports[0]["workers"],
        "global_batch": 128,
        "per_gpu_batch": 64,
        "world_size": 2,
        "training_images_per_second": 48 * 128 / train_seconds,
        "loader_images_per_second": 12 * 128 / loader_seconds,
        "training_seconds": train_seconds,
        "loader_seconds": loader_seconds,
        "ranks": reports,
        "full_training_eta_available": False,
    }


def compare_cases(cases):
    result = {}
    for workers in (0, 2, 4):
        selected = [case for case in cases if case["workers"] == workers]
        if [case["backend"] for case in selected] != ["shard", "npy", "npy", "shard"]:
            raise ValueError("Incomplete ABBA case group")
        exact = True
        max_loss_difference = 0.0
        for rank in (0, 1):
            baseline = selected[0]["ranks"][rank]
            for case in selected[1:]:
                current = case["ranks"][rank]
                for key in ("initial_parameters_sha256", "sample_order_sha256", "loader_order_sha256"):
                    if baseline[key] != current[key]:
                        raise ValueError(f"Mismatched experiment inputs: {key}")
                left, right = np.asarray(baseline["loss_trace"]), np.asarray(current["loss_trace"])
                difference = float(np.max(np.abs(left - right)))
                max_loss_difference = max(max_loss_difference, difference)
                if not np.allclose(left, right, rtol=1e-5, atol=1e-6):
                    raise ValueError("Backend loss trajectories differ beyond tolerance")
                exact &= baseline["final_parameters_sha256"] == current["final_parameters_sha256"]
        medians = {
            backend: {
                metric: statistics.median(case[metric] for case in selected if case["backend"] == backend)
                for metric in ("training_images_per_second", "loader_images_per_second")
            }
            for backend in ("shard", "npy")
        }
        result[str(workers)] = {
            "medians": medians,
            "training_speedup": medians["npy"]["training_images_per_second"]
            / medians["shard"]["training_images_per_second"],
            "loader_speedup": medians["npy"]["loader_images_per_second"] / medians["shard"]["loader_images_per_second"],
            "final_parameters_bitwise_equal": exact,
            "max_loss_absolute_difference": max_loss_difference,
        }
    return result


def prepare(request):
    from ultralytics.nn.foundation.cache import FeatureCacheReader
    from ultralytics.utils import YAML

    root, probe = Path(request["root"]), Path(request["probe"])
    original = json.loads((probe / "request.json").read_text())
    manifest = json.loads((probe / "npy-manifest.json").read_text())
    reader = FeatureCacheReader(request["cache"], max_open_shards=4)
    try:
        if reader.index["content_sha256"] != original["source_content_sha256"]:
            raise ValueError("Source cache differs from the NPY experiment")
        if reader.index["contract_sha256"] != manifest["source_contract_sha256"]:
            raise ValueError("NPY contract mismatch")
        ids = original["sample_ids"]
        if len(ids) != 2048 or len(set(ids)) != 2048:
            raise ValueError("Expected exactly 2048 unique previously verified samples")
        missing_labels = []
        for index, sid in enumerate(ids):
            guard_memory(memory_snapshot(), 24 * GIB)
            record = reader.records[sid]
            validate_array(np.load(sample_path(probe / "npy", sid), allow_pickle=False), record)
            for category, suffix in (("images", ".jpg"), ("labels", ".txt")):
                source = Path(request["data_root"]) / category / (sid + suffix)
                target = root / "inputs/coco" / category / (sid + suffix)
                if not source.is_file():
                    if category == "labels":
                        missing_labels.append(sid)
                        continue
                    raise FileNotFoundError(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.symlink_to(source)
            if (index + 1) % 256 == 0:
                write_json(root / "check-progress.json", {"phase": "VERIFY_NPY", "completed": index + 1, "total": 2048})
        write_json(
            root / "inputs/label-scope.json",
            {
                "missing_label_ids": missing_labels,
                "policy": "Preserve the source dataset's missing-label behavior for both backends; do not synthesize labels",
            },
        )
        listing = root / "inputs/train.txt"
        listing.write_text("\n".join(str(root / "inputs/coco/images" / (sid + ".jpg")) for sid in ids) + "\n")
        names = YAML.load(REPO / "ultralytics/cfg/datasets/coco.yaml")["names"]
        YAML.save(
            root / "inputs/coco.yaml",
            {
                "path": str(root / "inputs/coco"),
                "train": str(listing),
                "val": str(listing),
                "names": names,
            },
        )
    finally:
        reader.close()


def prototype_trainer_class():
    from ultralytics.models.yolo.detect.foundation_train import D1FoundationDetectionTrainer

    class PrototypeTrainer(D1FoundationDetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            base = dataset.feature_reader
            # Both backends use the same bounded subset index; not a full-cache memory measurement.
            base.records = {sid: base.records[sid] for sid in dataset.sample_ids}
            if self.layout_backend == "npy":
                dataset.feature_reader = NpyReader(base, self.npy_root)
            return dataset

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            workers = self.args.workers
            try:
                if mode == "val":
                    self.args.workers = 0
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = workers

        def save_model(self):
            return False

        def _refresh_healthy_checkpoint(self):
            return False

    return PrototypeTrainer


def parity(request):
    import copy

    import torch

    from ultralytics.cfg import get_cfg
    from ultralytics.data.d1_cache import D1FeatureCacheDataset, move_d1_batch_to_device
    from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
    from ultralytics.utils import YAML
    from ultralytics.utils.torch_utils import init_seeds

    prepare(request)
    root = Path(request["root"])
    init_seeds(0, deterministic=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    config = YAML.load(REPO / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml")
    names = YAML.load(REPO / "ultralytics/cfg/datasets/coco.yaml")["names"]
    dataset = D1FeatureCacheDataset(
        img_path=str(root / "inputs/train.txt"),
        cache_dir=request["cache"],
        data={"names": names, "nc": 80, "channels": 3},
        hyp=get_cfg(overrides=config),
        imgsz=640,
        batch_size=4,
        trusted_cache=True,
        max_open_shards=4,
        prefetch_factor=1,
    )
    try:
        a = dataset.collate_fn([dataset[index] for index in (0, 511, 1023, 2047)])
        dataset.feature_reader = NpyReader(dataset.feature_reader, Path(request["probe"]) / "npy")
        b = dataset.collate_fn([dataset[index] for index in (0, 511, 1023, 2047)])
        for name in NAMES:
            if not torch.equal(a["features"][name], b["features"][name]):
                raise ValueError("Feature mismatch")
        for key in ("bboxes", "cls", "batch_idx"):
            if not torch.equal(a[key], b[key]):
                raise ValueError("Label mismatch")
        reference = D1FoundationDetectionModel()
        outcomes = []
        for batch in (a, b):
            model = copy.deepcopy(reference).cuda().train()
            batch = move_d1_batch_to_device(batch, "cuda:0", feature_dtype=torch.float16)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, items = model(batch)
            loss.sum().backward()
            gradients = {
                name: parameter.grad.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            if not torch.isfinite(loss).all() or not all(torch.isfinite(g).all() for g in gradients.values()):
                raise ValueError("Nonfinite parity loss or gradient")
            outcomes.append((items.detach().cpu(), gradients))
            del model
        if not torch.equal(outcomes[0][0], outcomes[1][0]):
            raise ValueError("AMP loss parity failed")
        if outcomes[0][1].keys() != outcomes[1][1].keys():
            raise ValueError("Gradient key mismatch")
        if not all(torch.equal(outcomes[0][1][name], gradient) for name, gradient in outcomes[1][1].items()):
            raise ValueError("AMP gradient parity failed")
        write_json(
            root / "parity.json",
            {
                "status": "PASSED",
                "verified_npy_samples": 2048,
                "verified_tensors": 6144,
                "gpu_batch": 4,
                "features_equal": True,
                "labels_equal": True,
                "amp_loss_bitwise_equal": True,
                "amp_gradients_bitwise_equal": True,
                "gradient_tensor_count": len(outcomes[0][1]),
                "optimizer_updates": 0,
            },
        )
    finally:
        dataset.feature_reader.close()


def worker(request, case_id):
    import torch
    import torch.distributed as dist

    from ultralytics.utils import YAML
    from ultralytics.utils.torch_utils import unwrap_model

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two visible GPUs and ranks required")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    case = request["cases"][case_id]
    folder = Path(request["root"]) / f"case-{case_id:02d}-{case['backend']}-w{case['workers']}"
    experiment = YAML.load(REPO / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml")
    model_config = YAML.load(REPO / experiment["model"])
    overrides = {
        **experiment,
        **model_config["loss"],
        "model": str(REPO / experiment["model"]),
        "data": str(Path(request["root"]) / "inputs/coco.yaml"),
        "device": "0,1",
        "epochs": 4,
        "batch": 128,
        "nbs": 128,
        "workers": case["workers"],
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 1.0,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.0,
        "cos_lr": False,
        "patience": 100,
        "seed": 0,
        "amp": True,
        "save": False,
        "val": False,
        "plots": False,
        "pretrained": False,
        "compile": False,
        "cache": False,
        "close_mosaic": 0,
        "verbose": False,
        "project": str(folder),
        "name": "train",
        "exist_ok": True,
    }
    cls = prototype_trainer_class()
    cls.layout_backend, cls.npy_root = case["backend"], Path(request["probe"]) / "npy"
    trainer = cls(
        overrides=overrides,
        feature_caches={"train": request["cache"], "val": request["cache"]},
        trusted_feature_cache=True,
        max_open_feature_shards=4,
        feature_prefetch_factor=1,
        amp_init_scale=1.0,
        amp_growth_interval=1_000_000,
    )
    track = {"steps": 0, "updates": 0, "wait": 0.0, "step": 0.0, "losses": [], "ids": [], "loader_ids": []}

    def progress(phase, **extra):
        write_json(
            folder / f"progress-rank-{rank}.json",
            {
                "phase": phase,
                "rank": rank,
                "pid": os.getpid(),
                "backend": case["backend"],
                "workers": case["workers"],
                "steps": track["steps"],
                "optimizer_updates": track["updates"],
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **extra,
            },
        )

    def ready(t):
        if t.train_loader.num_workers != case["workers"] or len(t.train_loader) != 16:
            raise ValueError("Loader configuration changed")
        if not t.amp or t.train_loader.batch_size != 64 or t.accumulate != 1:
            raise ValueError("AMP/batch/accumulation changed")
        t.optimizer.register_step_post_hook(lambda *unused: track.update(updates=track["updates"] + 1))
        progress("DATALOADER_H2D")
        dist.barrier()
        duration = 0.0
        iterator = iter(t.train_loader)
        for step in range(16):
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = next(iterator)
            track["loader_ids"].extend(batch["sample_id"])
            batch = t.preprocess_batch(batch)
            torch.cuda.synchronize()
            if step >= 4:
                duration += time.perf_counter() - started
            del batch
        track["loader_seconds"] = duration
        t.train_loader.set_epoch(0)
        t.train_loader.reset()
        dist.barrier()

    def start(t):
        track["initial"] = parameter_digest(unwrap_model(t.model))
        torch.cuda.reset_peak_memory_stats()
        progress("TRAINING_START")
        torch.cuda.synchronize()
        track["previous"] = time.perf_counter()

    def epoch_start(t):
        torch.cuda.synchronize()
        track["previous"] = time.perf_counter()

    def batch_start(t):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if track["steps"] >= 16:
            track["wait"] += now - track["previous"]
        track["started"] = now
        track["ids"].extend(t.batch["sample_id"])

    def batch_end(t):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if track["steps"] >= 16:
            track["step"] += now - track["started"]
        track["steps"] += 1
        values = t.loss_items.detach().float().cpu().tolist()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite real training loss")
        track["losses"].append(values)
        if track["steps"] == 1 or track["steps"] % 8 == 0:
            guard_memory(memory_snapshot(), 24 * GIB)
            progress("TRAINING_NORMAL", losses=values, amp_scale=float(t.scaler.get_scale()))
        track["previous"] = time.perf_counter()

    folder.mkdir(parents=True, exist_ok=True)
    for event, callback in (
        ("on_pretrain_routine_end", ready),
        ("on_train_start", start),
        ("on_train_epoch_start", epoch_start),
        ("on_train_batch_start", batch_start),
        ("on_train_batch_end", batch_end),
    ):
        trainer.add_callback(event, callback)
    trainer.train()
    model = unwrap_model(trainer.model)
    final = parameter_digest(model)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Nonfinite final model parameters")
    report = {
        "rank": rank,
        "backend": case["backend"],
        "workers": case["workers"],
        "training_steps": track["steps"],
        "optimizer_updates": track["updates"],
        "measured_steps": max(0, track["steps"] - 16),
        "training_seconds": track["wait"] + track["step"],
        "data_wait_seconds": track["wait"],
        "step_seconds": track["step"],
        "loader_seconds": track["loader_seconds"],
        "loss_trace": track["losses"],
        "finite_losses": True,
        "initial_parameters_sha256": track["initial"],
        "final_parameters_sha256": final,
        "weights_changed": final != track["initial"],
        "sample_order_sha256": json_digest(track["ids"]),
        "loader_order_sha256": json_digest(track["loader_ids"]),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "final_amp_scale": float(trainer.scaler.get_scale()),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    write_json(folder / f"rank-{rank}.json", report)
    progress("COMPLETED")
    if dist.is_initialized():
        dist.destroy_process_group()


def is_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait(timeout=10)
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def supervise(request):
    root = Path(request["root"])
    current = None
    keeper_stopped = False
    status = {"status": "RUNNING", "phase": "STARTING", "pid": os.getpid()}
    started = time.monotonic()

    def publish(**updates):
        status.update(updates)
        status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "status.json", status)

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    publish()
    try:
        keeper_pid = request["keeper_pid"]
        actual = Path(f"/proc/{keeper_pid}/cmdline").read_bytes().split(b"\0")
        actual = [part.decode() for part in actual if part]
        if actual != request["keeper_command"]:
            raise RuntimeError("GPU keeper identity changed; refusing to stop it")
        os.kill(keeper_pid, signal.SIGTERM)
        keeper_stopped = True
        for _ in range(150):
            if not is_alive(keeper_pid):
                break
            time.sleep(0.1)
        if is_alive(keeper_pid):
            raise RuntimeError("GPU keeper did not stop cleanly")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(request["gpu_uuids"]),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="2",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            YOLO_CONFIG_DIR=str(root / "settings"),
        )
        work = [("parity", None)] + [("worker", index) for index in range(len(request["cases"]))]
        for mode, case_id in work:
            guard_memory(memory_snapshot(), 24 * GIB)
            command = [
                sys.executable,
                "-u",
                "-m",
                "scripts.d1.benchmark_cache_layout_gpu",
                mode,
                "--request",
                str(root / "request.json"),
            ]
            if case_id is not None:
                command += ["--case", str(case_id)]
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    "--nproc_per_node=2",
                    "--max_restarts=0",
                    "--no-python",
                    *command,
                ]
            label = "parity" if case_id is None else f"case-{case_id:02d}"
            publish(phase=mode.upper(), case=case_id, cases_total=len(request["cases"]))
            with (root / f"{label}.log").open("w") as stream:
                current = subprocess.Popen(
                    command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                )
            case_started = time.monotonic()
            while current.poll() is None:
                guard_memory(memory_snapshot(), 24 * GIB)
                if time.monotonic() - case_started > 600 or time.monotonic() - started > 1800:
                    raise TimeoutError("Bounded GPU test exceeded its time limit")
                publish(child_pid=current.pid)
                time.sleep(5)
            if current.returncode != 0:
                raise RuntimeError(f"{label} exited with status {current.returncode}; see {label}.log")
            stop_group(current)
            current = None
            if case_id is not None:
                case = request["cases"][case_id]
                folder = root / f"case-{case_id:02d}-{case['backend']}-w{case['workers']}"
                reports = [json.loads((folder / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
                write_json(folder / "aggregate.json", aggregate_case(reports))
        cases = []
        for index, case in enumerate(request["cases"]):
            path = root / f"case-{index:02d}-{case['backend']}-w{case['workers']}" / "aggregate.json"
            cases.append(json.loads(path.read_text()))
        write_json(
            root / "summary.json",
            {
                "status": "COMPLETED",
                "comparison": compare_cases(cases),
                "cases": cases,
                "full_training_eta_available": False,
                "request": str(root / "request.json"),
                "notes": request["notes"],
            },
        )
        publish(status="COMPLETED", phase="FINISHED")
    except BaseException as exc:
        publish(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if current is not None:
            stop_group(current)
        if keeper_stopped:
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            restored = subprocess.run(
                request["keeper_command"], env=env, capture_output=True, text=True, timeout=30, check=False
            )
            if restored.returncode != 0:
                status["status"] = "FAILED"
            publish(
                keeper_restore_returncode=restored.returncode,
                keeper_restore_output=(restored.stdout + restored.stderr)[-1000:],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "parity", "worker"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request["physical_gpu_indices"] != [4, 5] or len(request["gpu_uuids"]) != 2:
        raise ValueError("Only the explicitly authorized fifth and sixth GPUs are allowed")
    if args.mode == "supervise":
        supervise(request)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(request["gpu_uuids"]):
            raise RuntimeError("GPU isolation environment mismatch")
        from ultralytics.utils import SETTINGS

        SETTINGS.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "comet", "clearml", "neptune")})
        if args.mode == "parity":
            parity(request)
        else:
            worker(request, args.case)


if __name__ == "__main__":
    main()

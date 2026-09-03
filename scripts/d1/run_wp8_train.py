#!/usr/bin/env python3
"""Prepare, run, and summarize the D1 WP8 formal COCO 2017 training."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from scripts.d1.run_wp7 import capture_initial_state, compact_routing, read_results_csv, routing_deltas
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/d1.
    from run_wp7 import capture_initial_state, compact_routing, read_results_csv, routing_deltas

from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.foundation.cache import FeatureCacheReader, canonical_json_bytes, sha256_bytes, sha256_file
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import unwrap_model


SCHEMA_VERSION = "d1-wp8-train-v1"
SPLIT_COUNTS = {"train2017": 118_287, "val2017": 5_000}
FEATURE_NAMES = ("block4", "block8", "block12")
OUTPUT_LAYERS = (4, 8, 12)
EXPECTED_SHAPE = (384, 40, 40)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode() + b"\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()


def load_contract(path: Path) -> dict[str, Any]:
    contract = YAML.load(path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"WP8 contract must use schema {SCHEMA_VERSION!r}.")
    if type(contract.get("seed")) is not int or contract["seed"] < 0:
        raise ValueError("WP8 seed must be a non-negative integer.")
    if contract.get("deterministic") is not True:
        raise ValueError("WP8 requires deterministic=true.")
    hardware = contract.get("hardware", {})
    devices = tuple(str(hardware.get("devices", "")).split(","))
    world_size = hardware.get("world_size")
    if world_size != 6 or devices != ("0", "1", "2", "3", "4", "5"):
        raise ValueError("WP8 requires exactly CUDA devices 0,1,2,3,4,5.")
    cache_io = contract.get("cache_io", {})
    if cache_io.get("trusted") is not True:
        raise ValueError("WP8 requires a preflight-verified trusted feature cache.")
    max_open_shards = cache_io.get("max_open_shards_per_worker")
    if type(max_open_shards) is not int or max_open_shards <= 0:
        raise ValueError("WP8 max_open_shards_per_worker must be a positive integer.")
    prefetch_factor = cache_io.get("prefetch_factor")
    if type(prefetch_factor) is not int or prefetch_factor <= 0:
        raise ValueError("WP8 cache prefetch_factor must be a positive integer.")
    runtime = contract.get("runtime", {})
    amp_init_scale = runtime.get("amp_init_scale")
    if not isinstance(amp_init_scale, (int, float)) or amp_init_scale <= 0:
        raise ValueError("WP8 amp_init_scale must be a positive number.")
    amp_growth_interval = runtime.get("amp_growth_interval")
    if type(amp_growth_interval) is not int or amp_growth_interval <= 0:
        raise ValueError("WP8 amp_growth_interval must be a positive integer.")
    train = contract.get("train", {})
    required = {
        "epochs": 100,
        "batch": 48,
        "nbs": 48,
        "amp": True,
        "optimizer": "AdamW",
        "pretrained": False,
        "fraction": 1.0,
        "cache": False,
        "compile": False,
        "val": True,
    }
    mismatches = [name for name, value in required.items() if train.get(name) != value]
    if mismatches:
        raise ValueError("WP8 formal training contract mismatch: " + ", ".join(mismatches))
    if train["batch"] % world_size:
        raise ValueError("WP8 global batch must be divisible by world_size.")
    acceptance = contract.get("acceptance", {})
    if acceptance.get("train_samples") != SPLIT_COUNTS["train2017"]:
        raise ValueError("WP8 train sample count is not locked to official COCO 2017.")
    if acceptance.get("val_samples") != SPLIT_COUNTS["val2017"]:
        raise ValueError("WP8 val sample count is not locked to official COCO 2017.")
    return contract


def _report_candidates(workspace: Path, split: str) -> list[Path]:
    return sorted((workspace / "manifests").glob(f"wp8-full-{split}-*.json"))


def discover_cache_report(workspace: Path, split: str, explicit: Path | None) -> Path:
    candidates = [explicit.resolve()] if explicit else _report_candidates(workspace, split)
    passed = []
    for path in candidates:
        if not path or not path.is_file():
            continue
        report = load_json(path)
        final = report.get("finalization", {})
        if report.get("status") == "passed" and final.get("split") == split:
            passed.append(path)
    if len(passed) != 1:
        raise ValueError(f"expected exactly one passed full-cache report for {split}, got {passed}")
    return passed[0]


def validate_cache_evidence(cache_dir: Path, report_path: Path, split: str) -> dict[str, Any]:
    reader = FeatureCacheReader(cache_dir)
    report = load_json(report_path)
    final = report.get("finalization", {})
    verification = final.get("verification", {})
    index = reader.index
    expected = {
        "status": "passed",
        "split": split,
        "sample_count": SPLIT_COUNTS[split],
        "contract_sha256": index.get("contract_sha256"),
        "content_sha256": index.get("content_sha256"),
        "shard_count": len(index.get("shards", [])),
    }
    actual = {
        "status": final.get("status"),
        "split": final.get("split"),
        "sample_count": verification.get("sample_count"),
        "contract_sha256": verification.get("contract_sha256"),
        "content_sha256": verification.get("content_sha256"),
        "shard_count": verification.get("shard_count"),
    }
    if actual != expected:
        raise ValueError(f"{split} cache report and index disagree: expected={expected}, actual={actual}")
    if len(reader.records) != SPLIT_COUNTS[split] or index.get("split_counts") != {split: SPLIT_COUNTS[split]}:
        raise ValueError(f"{split} cache index has an invalid sample count.")
    contract = reader.contract
    if tuple(contract["feature_names"]) != FEATURE_NAMES:
        raise ValueError(f"{split} cache has unexpected feature names.")
    if tuple(contract["output_layers"]) != OUTPUT_LAYERS or tuple(contract["expected_shape"]) != EXPECTED_SHAPE:
        raise ValueError(f"{split} cache has an unexpected DINOv3 output contract.")
    if contract["dtype"] != "float16":
        raise ValueError(f"{split} cache must contain float16 features.")
    indexed = {entry["filename"]: entry for entry in index["shards"]}
    actual_names = {path.name for path in cache_dir.glob("*.safetensors")}
    if actual_names != set(indexed):
        raise ValueError(f"{split} cache shard set differs from its index.")
    for name, shard in indexed.items():
        if (cache_dir / name).stat().st_size != shard["bytes"]:
            raise ValueError(f"{split} cache shard size changed: {name}")
    part_files = sorted(path.name for path in cache_dir.glob("*.part"))
    if part_files:
        raise ValueError(f"{split} cache contains unfinished files: {part_files[:3]}")
    return {
        "split": split,
        "report": report_path.name,
        "sample_count": len(reader.records),
        "shard_count": len(indexed),
        "cache_bytes": sum(entry["bytes"] for entry in indexed.values()),
        "contract_sha256": index["contract_sha256"],
        "content_sha256": index["content_sha256"],
        "teacher_weights_sha256": contract["teacher_weights_sha256"],
    }


def training_overrides(repo_root: Path, contract: Mapping[str, Any], data_yaml: Path, run_root: Path) -> dict[str, Any]:
    experiment = YAML.load(repo_root / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml")
    train = dict(contract["train"])
    overrides = {
        **experiment,
        **train,
        "data": str(data_yaml),
        "model": str((repo_root / experiment["model"]).resolve()),
        "seed": int(contract["seed"]),
        "deterministic": bool(contract["deterministic"]),
        "device": contract["hardware"]["devices"],
        "project": str(run_root.parent),
        "name": run_root.name,
        "exist_ok": True,
        "verbose": False,
        "rect": False,
        "multi_scale": 0.0,
    }
    return overrides


def resolved_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.repo_root = args.repo_root.resolve()
    args.workspace = args.workspace.resolve()
    commit = git_commit(args.repo_root)
    short = commit[:7]
    args.config = (args.config or args.repo_root / "ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml").resolve()
    args.data_root = (args.data_root or args.workspace / "datasets/coco").resolve()
    args.train_cache = (
        args.train_cache or args.workspace / "feature_cache/coco2017-train2017-d1-cache-v1"
    ).resolve()
    args.val_cache = (args.val_cache or args.workspace / "feature_cache/coco2017-val2017-d1-cache-v1").resolve()
    args.train_cache_report = discover_cache_report(args.workspace, "train2017", args.train_cache_report)
    args.val_cache_report = discover_cache_report(args.workspace, "val2017", args.val_cache_report)
    args.run_root = (args.run_root or args.workspace / f"runs/wp8-formal-{short}").resolve()
    args.report_dir = (args.report_dir or args.workspace / f"manifests/wp8-formal-{short}").resolve()
    return args


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.config)
    train_cache = validate_cache_evidence(args.train_cache, args.train_cache_report, "train2017")
    val_cache = validate_cache_evidence(args.val_cache, args.val_cache_report, "val2017")
    if train_cache["contract_sha256"] != val_cache["contract_sha256"]:
        raise ValueError("train and val cache contracts differ.")
    if train_cache["teacher_weights_sha256"] != val_cache["teacher_weights_sha256"]:
        raise ValueError("train and val caches use different Teacher weights.")
    for split, count in SPLIT_COUNTS.items():
        split_dir = args.data_root / "images" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        manifest = args.repo_root / "experiments/d1/manifests" / f"coco2017-{split}.txt"
        entries = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if len(entries) != count:
            raise ValueError(f"tracked {split} list has {len(entries)} entries instead of {count}.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < contract["hardware"]["world_size"]:
        raise RuntimeError("WP8 preflight requires six visible CUDA devices.")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(contract["hardware"]["world_size"])]
    if any(name != "NVIDIA A40" for name in gpu_names):
        raise RuntimeError(f"WP8 preflight expected six NVIDIA A40 GPUs, got {gpu_names}.")
    data_template = YAML.load(args.repo_root / "ultralytics/cfg/datasets/coco.yaml")
    data_yaml = args.run_root / "inputs/coco2017.yaml"
    YAML.save(
        data_yaml,
        {
            "path": str(args.data_root),
            "train": "images/train2017",
            "val": "images/val2017",
            "names": data_template["names"],
        },
    )
    overrides = training_overrides(args.repo_root, contract, data_yaml, args.run_root)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": git_commit(args.repo_root),
        "config_sha256": sha256_file(args.config),
        "model_config_sha256": sha256_file(Path(overrides["model"])),
        "train_cache": train_cache,
        "val_cache": val_cache,
        "global_batch": contract["train"]["batch"],
        "per_gpu_batch": contract["train"]["batch"] // contract["hardware"]["world_size"],
        "world_size": contract["hardware"]["world_size"],
        "cache_io": dict(contract["cache_io"]),
        "runtime": dict(contract["runtime"]),
    }
    identity_path = args.run_root / "inputs/identity.json"
    if identity_path.is_file() and load_json(identity_path) != identity:
        raise RuntimeError(f"run directory belongs to another WP8 identity: {args.run_root}")
    write_json(identity_path, identity)
    write_json(args.run_root / "inputs/resolved-overrides.json", overrides)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "preflight",
        "status": "passed",
        "identity": identity,
        "run_directory": str(args.run_root),
        "data_yaml": str(data_yaml),
        "trainable_model_parameters": 3_542_567,
        "teacher_loaded_during_training": False,
        "gpu_names": gpu_names,
        "approval_required_before_training": True,
    }
    write_json(args.report_dir / "preflight.json", report)
    return report


class EpochTelemetry:
    """Persist lightweight per-rank timing and routing evidence without per-batch logs."""

    def __init__(self, report_dir: Path) -> None:
        self.report_dir = report_dir
        self.rank = int(os.environ.get("RANK", "0"))
        self.batch_started: float | None = None
        self.previous_batch_ended: float | None = None
        self.data_wait_seconds = 0.0
        self.step_seconds = 0.0
        self.batch_count = 0

    def on_pretrain_routine_end(self, trainer) -> None:
        if self.rank != 0:
            return
        path = trainer.save_dir / "inputs/initial-routing-state.pt"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(capture_initial_state(unwrap_model(trainer.model)), path)

    def on_train_batch_start(self, _trainer) -> None:
        now = time.perf_counter()
        if self.previous_batch_ended is not None:
            self.data_wait_seconds += now - self.previous_batch_ended
        self.batch_started = now

    def on_train_batch_end(self, _trainer) -> None:
        now = time.perf_counter()
        if self.batch_started is not None:
            self.step_seconds += now - self.batch_started
        self.previous_batch_ended = now
        self.batch_count += 1

    def on_train_epoch_end(self, trainer) -> None:
        model = unwrap_model(trainer.model)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "epoch": int(trainer.epoch),
            "batch_count": self.batch_count,
            "data_wait_seconds": self.data_wait_seconds,
            "step_seconds": self.step_seconds,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else 0,
            "optimizer_steps": int(trainer.optimizer_steps),
            "routing": compact_routing(model),
        }
        write_json(self.report_dir / "epochs" / f"rank-{self.rank:02d}-epoch-{trainer.epoch:03d}.json", payload)
        self.batch_started = None
        self.previous_batch_ended = None
        self.data_wait_seconds = 0.0
        self.step_seconds = 0.0
        self.batch_count = 0

    def on_fit_epoch_end(self, trainer) -> None:
        if self.rank != 0:
            return
        payload = {
            "schema_version": SCHEMA_VERSION,
            "epoch": int(trainer.epoch),
            "validator_seen": int(getattr(trainer.validator, "seen", 0)),
            "metrics": dict(trainer.metrics or {}),
        }
        write_json(self.report_dir / "validation" / f"epoch-{trainer.epoch:03d}.json", payload)


def train(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    contract = load_contract(args.config)
    if world_size != contract["hardware"]["world_size"] or rank < 0 or local_rank < 0:
        raise RuntimeError("WP8 train must run under torchrun with exactly six processes.")
    preflight = load_json(args.report_dir / "preflight.json")
    if preflight.get("status") != "passed" or preflight["identity"]["code_commit"] != git_commit(args.repo_root):
        raise RuntimeError("WP8 preflight is missing, failed, or belongs to another commit.")
    data_yaml = args.run_root / "inputs/coco2017.yaml"
    overrides = training_overrides(args.repo_root, contract, data_yaml, args.run_root)
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        overrides["resume"] = str(args.resume.resolve())
    telemetry = EpochTelemetry(args.report_dir)
    cache_io = contract["cache_io"]
    trainer = D1FoundationDetectionTrainer(
        overrides=overrides,
        feature_caches={"train": args.train_cache, "val": args.val_cache},
        trusted_feature_cache=cache_io["trusted"],
        max_open_feature_shards=cache_io["max_open_shards_per_worker"],
        feature_prefetch_factor=cache_io["prefetch_factor"],
        amp_init_scale=contract["runtime"]["amp_init_scale"],
        amp_growth_interval=contract["runtime"]["amp_growth_interval"],
    )
    trainer.add_callback("on_pretrain_routine_end", telemetry.on_pretrain_routine_end)
    trainer.add_callback("on_train_batch_start", telemetry.on_train_batch_start)
    trainer.add_callback("on_train_batch_end", telemetry.on_train_batch_end)
    trainer.add_callback("on_train_epoch_end", telemetry.on_train_epoch_end)
    trainer.add_callback("on_fit_epoch_end", telemetry.on_fit_epoch_end)
    trainer.train()
    return {"status": "passed", "rank": rank}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.config)
    preflight = load_json(args.report_dir / "preflight.json")
    results_path = args.run_root / "results.csv"
    rows = read_results_csv(results_path)
    failures = []
    if len(rows) != contract["acceptance"]["require_epochs"]:
        failures.append(f"results.csv contains {len(rows)} epochs instead of {contract['acceptance']['require_epochs']}")
    numeric = [value for row in rows for value in row.values()]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        failures.append("training results contain NaN or Inf")
    validation_path = args.report_dir / "validation" / f"epoch-{len(rows) - 1:03d}.json"
    validation = load_json(validation_path) if validation_path.is_file() else {}
    if validation.get("validator_seen") != SPLIT_COUNTS["val2017"]:
        failures.append("final validation did not process all 5,000 COCO val images")
    checkpoint_path = args.run_root / "weights/best.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = args.run_root / "weights/last.pt"
    checkpoint_model, checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    if not isinstance(checkpoint_model, D1FoundationDetectionModel):
        failures.append("checkpoint does not contain D1FoundationDetectionModel")
    restored = D1FoundationDetectionModel(checkpoint_model.config_dict()).eval()
    initialize_mixture_loss_ema_buffer(restored)
    restored.load_state_dict(checkpoint_model.float().state_dict(), strict=True)
    teacher_keys = [key for key in restored.state_dict() if "teacher" in key.lower() or "dinov3" in key.lower()]
    if teacher_keys:
        failures.append("checkpoint contains Teacher parameters")
    initial_path = args.run_root / "inputs/initial-routing-state.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    deltas = routing_deltas(initial, restored)
    epoch_reports = [load_json(path) for path in sorted((args.report_dir / "epochs").glob("rank-*.json"))]
    wait = sum(report["data_wait_seconds"] for report in epoch_reports)
    step = sum(report["step_seconds"] for report in epoch_reports)
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "formal-training-summary",
        "status": "passed" if not failures else "failed",
        "identity": preflight["identity"],
        "epochs": len(rows),
        "final_metrics": rows[-1] if rows else {},
        "final_validator_seen": validation.get("validator_seen"),
        "data_wait_ratio": wait / (wait + step) if wait + step else None,
        "peak_gpu_bytes_by_rank": {
            str(rank): max(
                (item["peak_gpu_bytes"] for item in epoch_reports if item["rank"] == rank), default=0
            )
            for rank in range(contract["hardware"]["world_size"])
        },
        "routing_deltas": deltas,
        "checkpoint": {
            "filename": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "strict_reload": True,
            "teacher_parameter_count": len(teacher_keys),
            "epoch": int(checkpoint.get("epoch", -1)),
        },
        "failures": failures,
    }
    write_json(args.report_dir / "summary.json", report)
    if failures:
        raise RuntimeError("WP8 formal training failed acceptance: " + "; ".join(failures))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--train-cache", type=Path)
    result.add_argument("--val-cache", type=Path)
    result.add_argument("--train-cache-report", type=Path)
    result.add_argument("--val-cache-report", type=Path)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--report-dir", type=Path)
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--resume", type=Path)
    subparsers.add_parser("summarize")
    return result


def main() -> None:
    args = resolved_paths(parser().parse_args())
    if args.command == "prepare":
        report = prepare(args)
    elif args.command == "train":
        report = train(args)
    else:
        report = summarize(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
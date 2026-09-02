#!/usr/bin/env python3
"""Run the D1 WP7 online-parity and minimal-training acceptance gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch

try:
    from scripts.d1.cache_features import EXPECTED_SHAPE, FEATURE_NAMES, load_image, make_letterbox
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/d1.
    from cache_features import EXPECTED_SHAPE, FEATURE_NAMES, load_image, make_letterbox

from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.foundation import DINOv3Teacher
from ultralytics.nn.foundation.cache import FeatureCacheReader, sha256_file, verify_feature_cache
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import unwrap_model


SCHEMA_VERSION = "d1-wp7-v1"
PROFILE_NAMES = ("overfit32", "coco8")
REPORT_NAMES = {
    "parity": "wp7-parity.json",
    "overfit32": "wp7-overfit32.json",
    "coco8": "wp7-coco8.json",
    "summary": "wp7-summary.json",
}


def write_json(path: Path, payload: Any) -> None:
    """Atomically write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()


def load_contract(path: Path) -> dict[str, Any]:
    contract = YAML.load(path)
    if not isinstance(contract, dict) or contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"WP7 contract must use schema_version={SCHEMA_VERSION!r}.")
    if contract.get("seed") != 0 or contract.get("deterministic") is not True:
        raise ValueError("WP7 requires seed=0 and deterministic=true.")
    common = contract.get("common")
    if not isinstance(common, dict):
        raise TypeError("WP7 common configuration must be a mapping.")
    required_common = {
        "imgsz": 640,
        "workers": 0,
        "amp": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0,
        "warmup_epochs": 0.0,
    }
    for name, expected in required_common.items():
        if common.get(name) != expected:
            raise ValueError(f"WP7 common.{name} must be {expected!r}.")
    profiles = contract.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(PROFILE_NAMES):
        raise ValueError(f"WP7 profiles must be exactly {PROFILE_NAMES}.")
    overfit_ids = tuple(profiles["overfit32"].get("train_ids", ()))
    coco_train = tuple(profiles["coco8"].get("train_ids", ()))
    coco_val = tuple(profiles["coco8"].get("val_ids", ()))
    if len(overfit_ids) != 32 or len(set(overfit_ids)) != 32:
        raise ValueError("overfit32 must contain exactly 32 unique train IDs.")
    if len(coco_train) != 4 or len(coco_val) != 4 or set(coco_train) & set(coco_val):
        raise ValueError("coco8 must contain disjoint four-image train and val splits.")
    if overfit_ids[:8] != coco_train + coco_val:
        raise ValueError("coco8 IDs must be the first eight overfit/cache IDs in train-then-val order.")
    return contract


def runtime_identity(repo_root: Path, reader: FeatureCacheReader, weights_dir: Path) -> dict[str, Any]:
    weights = weights_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    index = load_json(reader.root / "index.json")
    expected_weight_hash = reader.contract["teacher_weights_sha256"]
    actual_weight_hash = sha256_file(weights)
    if actual_weight_hash != expected_weight_hash:
        raise ValueError("DINOv3 weights do not match the feature-cache contract.")
    return {
        "schema_version": SCHEMA_VERSION,
        "code_commit": git_commit(repo_root),
        "cache_id": reader.root.name,
        "cache_contract_sha256": index["contract_sha256"],
        "cache_content_sha256": index["content_sha256"],
        "teacher_weights_sha256": actual_weight_hash,
    }


def report_matches(report: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    return report.get("status") == "passed" and report.get("identity") == dict(identity)


def _record_image_id(record: Mapping[str, Any]) -> str:
    sample_id = str(record.get("sample_id", ""))
    parts = sample_id.split("/")
    if len(parts) != 2 or parts[0] != "train2017" or not parts[1].isdigit() or len(parts[1]) != 12:
        raise ValueError(f"invalid WP7 train2017 sample ID: {sample_id!r}")
    return parts[1]


def select_records(
    records: Mapping[str, Mapping[str, Any]],
    image_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Select records in contract order and reject aliases or missing samples."""
    if isinstance(image_ids, (str, bytes)) or not image_ids:
        raise ValueError("image_ids must be a non-empty ordered sequence.")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_ids must not contain duplicates.")
    selected = []
    for image_id in image_ids:
        if not isinstance(image_id, str) or len(image_id) != 12 or not image_id.isdigit():
            raise ValueError(f"invalid COCO image ID: {image_id!r}")
        sample_id = f"train2017/{image_id}"
        if sample_id not in records:
            raise FileNotFoundError(f"cache is missing WP7 sample {sample_id}")
        record = dict(records[sample_id])
        if _record_image_id(record) != image_id:
            raise ValueError(f"cache record identity mismatch for {sample_id}")
        selected.append(record)
    return selected


def validate_source_files(
    data_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    require_nonempty_labels: bool,
) -> list[Path]:
    paths = []
    for record in records:
        relative = PurePosixPath(str(record["image_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"cache contains a non-portable image path: {relative}")
        image_path = data_root.joinpath(*relative.parts)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if relative.parts[:2] != ("images", "train2017"):
            raise ValueError(f"WP7 only accepts COCO train2017 cache records, got {relative}")
        label_relative = Path("labels", "train2017", f"{image_path.stem}.txt")
        label_path = data_root / label_relative
        if require_nonempty_labels and (not label_path.is_file() or label_path.stat().st_size == 0):
            raise ValueError(f"WP7 requires a non-empty label file for {record['sample_id']}")
        paths.append(image_path.resolve())
    return paths


def _write_lines(path: Path, values: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(value) for value in values) + "\n", encoding="utf-8")


def prepare_profile_files(
    repo_root: Path,
    data_root: Path,
    cache_reader: FeatureCacheReader,
    profile_name: str,
    profile: Mapping[str, Any],
    run_dir: Path,
) -> Path:
    train_records = select_records(cache_reader.records, profile["train_ids"])
    train_paths = validate_source_files(
        data_root,
        train_records,
        require_nonempty_labels=bool(profile.get("require_nonempty_labels", False)),
    )
    if profile.get("val_source") == "train":
        val_paths = train_paths
    else:
        val_records = select_records(cache_reader.records, profile["val_ids"])
        val_paths = validate_source_files(
            data_root,
            val_records,
            require_nonempty_labels=bool(profile.get("require_nonempty_labels", False)),
        )
    input_dir = run_dir / "inputs"
    train_file, val_file = input_dir / "train.txt", input_dir / "val.txt"
    _write_lines(train_file, train_paths)
    _write_lines(val_file, val_paths)
    coco = YAML.load(repo_root / "ultralytics/cfg/datasets/coco.yaml")
    data_yaml = input_dir / f"{profile_name}.yaml"
    YAML.save(
        data_yaml,
        {
            "path": str(data_root.resolve()),
            "train": str(train_file.resolve()),
            "val": str(val_file.resolve()),
            "names": coco["names"],
            "channels": 3,
        },
    )
    return data_yaml


def _tensor_list(value: Any) -> list[float] | None:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().reshape(-1).tolist()
    return None


def compact_routing(model: D1FoundationDetectionModel) -> dict[str, Any]:
    result = {}
    for level, mixture in model.mixtures.items():
        snapshot = mixture.routing_snapshot()
        result[level] = {
            "mean_router_probs": _tensor_list(snapshot.get("mean_router_probs")),
            "entropy": float(snapshot.get("entropy", 0.0)),
            "balance_loss": float(snapshot.get("balance_loss", 0.0)),
            "z_loss": float(snapshot.get("z_loss", 0.0)),
            "aux_loss": float(snapshot.get("aux_loss", 0.0)),
            "residual_gain": float(mixture.residual_gain.detach().float().cpu()),
            "finite": bool(snapshot.get("finite", False)),
        }
    return result


def capture_initial_state(model: D1FoundationDetectionModel) -> dict[str, torch.Tensor]:
    state = {}
    for level, mixture in model.mixtures.items():
        state[f"{level}.residual_gain"] = mixture.residual_gain.detach().float().cpu().clone()
        for name, parameter in mixture.router.expert_head.named_parameters():
            state[f"{level}.router.{name}"] = parameter.detach().float().cpu().clone()
    return state


def routing_deltas(
    initial: Mapping[str, torch.Tensor],
    model: D1FoundationDetectionModel,
) -> dict[str, dict[str, float]]:
    result = {}
    for level, mixture in model.mixtures.items():
        residual = mixture.residual_gain.detach().float().cpu()
        residual_delta = float((residual - initial[f"{level}.residual_gain"]).abs().max())
        router_delta = 0.0
        for name, parameter in mixture.router.expert_head.named_parameters():
            key = f"{level}.router.{name}"
            if key not in initial:
                raise ValueError(f"initial state is missing {key}")
            router_delta = max(
                router_delta,
                float((parameter.detach().float().cpu() - initial[key]).abs().max()),
            )
        result[level] = {
            "router_max_abs_delta": router_delta,
            "residual_gain_max_abs_delta": residual_delta,
            "residual_gain": float(residual),
        }
    return result


def read_results_csv(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                key, value = key.strip(), value.strip()
                if value:
                    row[key] = float(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"training results are empty: {path}")
    return rows


def _metric(row: Mapping[str, float], *names: str) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(f"none of the required metrics are present: {names}")


def evaluate_training_rows(
    rows: Sequence[Mapping[str, float]],
    profile_name: str,
    profile: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    losses = [
        sum(_metric(row, name) for name in ("train/box_loss", "train/cls_loss", "train/dfl_loss"))
        for row in rows
    ]
    values = [value for row in rows for value in row.values()]
    failures = []
    if not all(math.isfinite(value) for value in values + losses):
        failures.append("training CSV contains NaN or Inf")
    final = rows[-1]
    map50 = _metric(final, "metrics/mAP50(B)")
    map50_95 = _metric(final, "metrics/mAP50-95(B)")
    metrics: dict[str, Any] = {
        "epochs_recorded": len(rows),
        "all_finite": not failures,
        "final_map50": map50,
        "final_map50_95": map50_95,
        "initial_detection_loss": losses[0],
        "final_detection_loss": losses[-1],
    }
    if profile_name == "overfit32":
        window = int(profile["loss_window"])
        if len(losses) < 2 * window:
            failures.append(f"overfit32 requires at least {2 * window} recorded epochs")
        else:
            first = statistics.median(losses[:window])
            last = statistics.median(losses[-window:])
            ratio = last / first if first else math.inf
            metrics.update(
                first_window_detection_loss_median=first,
                last_window_detection_loss_median=last,
                final_to_initial_loss_ratio=ratio,
            )
            if ratio > float(profile["max_final_to_initial_loss"]):
                failures.append("overfit32 detection-loss reduction did not meet the contract")
        if map50 < float(profile["min_map50"]):
            failures.append("overfit32 mAP50 did not meet the contract")
        if map50_95 < float(profile["min_map50_95"]):
            failures.append("overfit32 mAP50-95 did not meet the contract")
    return metrics, failures


def validate_checkpoint(
    checkpoint_path: Path,
    initial_state_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    checkpoint_model, checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    if not isinstance(checkpoint_model, D1FoundationDetectionModel):
        raise TypeError("WP7 checkpoint does not contain a D1FoundationDetectionModel.")
    restored = D1FoundationDetectionModel(checkpoint_model.config_dict()).eval()
    initialize_mixture_loss_ema_buffer(restored)
    restored.load_state_dict(checkpoint_model.float().state_dict(), strict=True)
    state_keys = tuple(restored.state_dict())
    teacher_keys = [key for key in state_keys if "teacher" in key.lower() or "dinov3" in key.lower()]
    if teacher_keys:
        raise ValueError(f"WP7 downstream checkpoint unexpectedly contains Teacher keys: {teacher_keys[:3]}")
    initial = torch.load(initial_state_path, map_location="cpu", weights_only=True)
    deltas = routing_deltas(initial, restored)
    report = {
        "filename": checkpoint_path.name,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "strict_reload": True,
        "teacher_parameter_count": 0,
        "downstream_state_key_count": len(state_keys),
        "epoch": int(checkpoint.get("epoch", -1)),
        "optimizer_steps": int(checkpoint.get("optimizer_steps", 0)),
    }
    return report, deltas


def run_parity(args: argparse.Namespace, contract: Mapping[str, Any], reader: FeatureCacheReader, identity: dict) -> dict:
    report_path = args.report_dir / REPORT_NAMES["parity"]
    if report_path.is_file():
        existing = load_json(report_path)
        if report_matches(existing, identity):
            return existing
    config = contract["parity"]
    count = int(config["sample_count"])
    records = [dict(reader.records[sample_id]) for sample_id in sorted(reader.records)[:count]]
    paths = validate_source_files(args.data_root, records, require_nonempty_labels=False)
    verification = verify_feature_cache(args.cache_dir)
    teacher = DINOv3Teacher(
        model_id=reader.contract["model_id"],
        weights_path=args.weights_dir,
        local_files_only=True,
        dtype="fp16",
        device=args.device,
        output_layers=(4, 8, 12),
    )
    layer_sums = {name: 0.0 for name in FEATURE_NAMES}
    layer_counts = {name: 0 for name in FEATURE_NAMES}
    layer_max = {name: 0.0 for name in FEATURE_NAMES}
    rtol, atol = float(config["rtol"]), float(config["atol"])
    failures = []
    letterbox = make_letterbox()
    batch_size = int(config["batch_size"])
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images = torch.stack([load_image(path, letterbox) for path in batch_paths])
        online = teacher.encode(images)
        for batch_index, record in enumerate(records[offset : offset + batch_size]):
            cached = reader.get(record["sample_id"])
            for name in FEATURE_NAMES:
                candidate = online.dense[name][batch_index].detach().cpu().to(torch.float16)
                if tuple(candidate.shape) != EXPECTED_SHAPE or not torch.isfinite(candidate).all():
                    failures.append(f"{record['sample_id']} {name} has invalid shape or values")
                    continue
                delta = (candidate.float() - cached[name].float()).abs()
                layer_sums[name] += float(delta.sum())
                layer_counts[name] += delta.numel()
                layer_max[name] = max(layer_max[name], float(delta.max()))
                if not torch.allclose(candidate, cached[name], rtol=rtol, atol=atol):
                    failures.append(f"{record['sample_id']} {name} exceeds FP16 tolerance")
    if teacher.training or teacher.model.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        failures.append("Teacher did not remain frozen in eval mode")
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": "online_cache_parity",
        "status": "passed" if not failures else "failed",
        "identity": identity,
        "sample_count": len(records),
        "rtol": rtol,
        "atol": atol,
        "cache_verification": {
            "sample_count": verification["sample_count"],
            "tensor_count": verification["tensor_count"],
            "content_sha256": verification["content_sha256"],
        },
        "layers": {
            name: {
                "shape": list(EXPECTED_SHAPE),
                "dtype": "float16",
                "max_abs_error": layer_max[name],
                "mean_abs_error": layer_sums[name] / layer_counts[name],
            }
            for name in FEATURE_NAMES
        },
        "teacher_frozen_eval": not any("Teacher" in failure for failure in failures),
        "failures": failures,
    }
    write_json(report_path, report)
    if failures:
        raise RuntimeError("WP7 parity failed: " + "; ".join(failures[:5]))
    return report


def _training_overrides(
    repo_root: Path,
    contract: Mapping[str, Any],
    profile: Mapping[str, Any],
    data_yaml: Path,
    run_root: Path,
    profile_name: str,
    resume: Path | None,
) -> dict[str, Any]:
    experiment = YAML.load(repo_root / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml")
    overrides = {
        **experiment,
        **contract["common"],
        "data": str(data_yaml),
        "epochs": int(profile["epochs"]),
        "batch": int(profile["batch"]),
        "nbs": int(profile["nbs"]),
        "seed": int(contract["seed"]),
        "deterministic": bool(contract["deterministic"]),
        "project": str(run_root),
        "name": profile_name,
        "exist_ok": True,
        "save_period": -1,
        "patience": int(profile["epochs"]),
        "verbose": False,
    }
    if resume is not None:
        overrides["resume"] = str(resume)
    return overrides


def run_training(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    reader: FeatureCacheReader,
    identity: dict,
    profile_name: str,
) -> dict[str, Any]:
    profile = contract["profiles"][profile_name]
    report_path = args.report_dir / REPORT_NAMES[profile_name]
    if report_path.is_file():
        existing = load_json(report_path)
        if report_matches(existing, identity):
            return existing
        raise RuntimeError(f"stale or failed report exists: {report_path}")
    run_dir = args.run_root / profile_name
    run_identity_path = run_dir / "inputs" / "identity.json"
    if run_identity_path.is_file() and load_json(run_identity_path) != identity:
        raise RuntimeError(f"run directory belongs to another code/cache identity: {run_dir}")
    write_json(run_identity_path, identity)
    data_yaml = prepare_profile_files(args.repo_root, args.data_root, reader, profile_name, profile, run_dir)
    initial_state_path = run_dir / "inputs" / "initial-routing-state.pt"
    trace_path = run_dir / "routing-trace.jsonl"
    trainer_run_dir = args.run_root / profile_name
    last_checkpoint = trainer_run_dir / "weights" / "last.pt"
    resume = last_checkpoint if last_checkpoint.is_file() and not report_path.exists() else None
    overrides = _training_overrides(
        args.repo_root,
        contract,
        profile,
        data_yaml,
        args.run_root,
        profile_name,
        resume,
    )
    overrides["device"] = args.device
    trainer = D1FoundationDetectionTrainer(
        overrides=overrides,
        feature_caches={"train": args.cache_dir, "val": args.cache_dir},
    )

    def capture_initial(trainer_instance) -> None:
        model = unwrap_model(trainer_instance.model)
        if not isinstance(model, D1FoundationDetectionModel):
            raise TypeError("WP7 Trainer did not construct D1FoundationDetectionModel.")
        if not initial_state_path.is_file():
            torch.save(capture_initial_state(model), initial_state_path)

    def capture_batch(trainer_instance) -> None:
        model = unwrap_model(trainer_instance.model)
        payload = {
            "epoch": int(trainer_instance.epoch),
            "optimizer_steps": int(trainer_instance.optimizer_steps),
            "routing": compact_routing(model),
        }
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")

    trainer.add_callback("on_pretrain_routine_end", capture_initial)
    trainer.add_callback("on_train_batch_end", capture_batch)
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    rows = read_results_csv(trainer.csv)
    metrics, failures = evaluate_training_rows(rows, profile_name, profile)
    if trainer.optimizer_steps < int(profile["min_optimizer_steps"]):
        failures.append(
            f"optimizer_steps={trainer.optimizer_steps} is below {int(profile['min_optimizer_steps'])}"
        )
    expected_val = len(profile["train_ids"] if profile.get("val_source") == "train" else profile["val_ids"])
    validator_seen = int(getattr(trainer.validator, "seen", 0))
    if validator_seen != expected_val:
        failures.append(f"validator saw {validator_seen} images instead of {expected_val}")
    checkpoint_path = trainer.best if trainer.best.is_file() else trainer.last
    if not checkpoint_path.is_file():
        failures.append("training did not produce a checkpoint")
        checkpoint_report, deltas = {}, {}
    else:
        checkpoint_report, deltas = validate_checkpoint(checkpoint_path, initial_state_path)
    if profile_name == "overfit32":
        for level in ("p3", "p4", "p5"):
            values = deltas.get(level, {})
            if values.get("router_max_abs_delta", 0.0) <= 0.0:
                failures.append(f"{level} Router parameters did not change")
            if values.get("residual_gain_max_abs_delta", 0.0) <= 0.0:
                failures.append(f"{level} residual gain did not change")
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": profile_name,
        "status": "passed" if not failures else "failed",
        "identity": identity,
        "train_sample_count": len(profile["train_ids"]),
        "val_sample_count": expected_val,
        "epochs": int(profile["epochs"]),
        "batch": int(profile["batch"]),
        "optimizer": contract["common"]["optimizer"],
        "optimizer_steps": int(trainer.optimizer_steps),
        "validator_seen": validator_seen,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "routing_deltas": deltas,
        "routing_final": compact_routing(unwrap_model(trainer.model)),
        "checkpoint": checkpoint_report,
        "failures": failures,
    }
    write_json(report_path, report)
    if failures:
        raise RuntimeError(f"WP7 {profile_name} failed: " + "; ".join(failures))
    return report


def run_summary(args: argparse.Namespace, identity: dict) -> dict[str, Any]:
    gates = {}
    failures = []
    for name in ("parity", "coco8", "overfit32"):
        path = args.report_dir / REPORT_NAMES[name]
        if not path.is_file():
            failures.append(f"missing report {path.name}")
            continue
        report = load_json(path)
        gates[name] = report["status"]
        if not report_matches(report, identity):
            failures.append(f"{path.name} is failed or belongs to another runtime identity")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "gate": "wp7_summary",
        "status": "passed" if not failures else "failed",
        "identity": identity,
        "gates": gates,
        "failures": failures,
    }
    write_json(args.report_dir / REPORT_NAMES["summary"], summary)
    if failures:
        raise RuntimeError("WP7 summary failed: " + "; ".join(failures))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--weights-dir", type=Path)
    result.add_argument("--run-root", type=Path)
    result.add_argument("--report-dir", type=Path)
    result.add_argument("--device", default="cuda:0")
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("parity")
    train = subparsers.add_parser("train")
    train.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    subparsers.add_parser("summarize")
    subparsers.add_parser("all")
    return result


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.repo_root = args.repo_root.resolve()
    args.workspace = args.workspace.resolve()
    args.config = (args.config or args.repo_root / "ultralytics/cfg/experiments/d1/wp7-minimal-tests.yaml").resolve()
    args.cache_dir = (args.cache_dir or args.workspace / "feature_cache/wp2-train100-a").resolve()
    args.data_root = (args.data_root or args.workspace / "datasets/coco").resolve()
    args.weights_dir = (
        args.weights_dir or args.workspace / "weights/teachers/dinov3-vits16-pretrain-lvd1689m"
    ).resolve()
    args.run_root = (args.run_root or args.workspace / "runs/wp7").resolve()
    args.report_dir = (args.report_dir or args.workspace / "manifests/wp7").resolve()
    return args


def main() -> None:
    args = resolve_args(parser().parse_args())
    contract = load_contract(args.config)
    reader = FeatureCacheReader(args.cache_dir)
    identity = runtime_identity(args.repo_root, reader, args.weights_dir)
    if args.command == "parity":
        report = run_parity(args, contract, reader, identity)
    elif args.command == "train":
        report = run_training(args, contract, reader, identity, args.profile)
    elif args.command == "summarize":
        report = run_summary(args, identity)
    else:
        run_parity(args, contract, reader, identity)
        run_training(args, contract, reader, identity, "coco8")
        run_training(args, contract, reader, identity, "overfit32")
        report = run_summary(args, identity)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()

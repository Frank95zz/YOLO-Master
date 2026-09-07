#!/usr/bin/env python3
"""Run read-only diagnostics for a stopped D1 WP8 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ultralytics.data import converter
from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
from ultralytics.models.yolo.detect.foundation_val import D1FoundationDetectionValidator
from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.foundation.cache import sha256_file
from ultralytics.nn.foundation.npy_cache import NPY_SCHEMA_VERSION, open_feature_cache
from ultralytics.nn.foundation_detection_model import D1_AUX_REPORT_NAMES
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML

SCHEMA_VERSION = "d1-wp8-diagnostic-v1"
EXPECTED_SPLIT_COUNTS = {"train2017": 118_287, "val2017": 5_000}
ROUTING_LEVELS = ("p3", "p4", "p5")
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def write_json(path: Path, payload: Any) -> None:
    """Atomically write stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode() + b"\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write stable JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def sha256_lines(lines: Sequence[str]) -> str:
    payload = (("\n".join(lines) + "\n") if lines else "").encode()
    return hashlib.sha256(payload).hexdigest()


def git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo_root), "status", "--short"], text=True).splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def select_split_paths(
    repo_root: Path,
    data_root: Path,
    split: str,
    limit: int | None,
) -> tuple[list[str], list[Path], str]:
    """Select a deterministic prefix of an official tracked COCO split."""
    if split not in EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"unsupported split {split!r}")
    manifest = repo_root / "experiments/d1/manifests" / f"coco2017-{split}.txt"
    relative = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    expected = EXPECTED_SPLIT_COUNTS[split]
    if len(relative) != expected or len(relative) != len(set(relative)):
        raise ValueError(f"{manifest} must contain {expected} unique paths")
    if relative != sorted(relative):
        raise ValueError(f"{manifest} must remain sorted")
    if limit is not None:
        if type(limit) is not int or not 0 < limit <= expected:
            raise ValueError(f"limit must be in [1, {expected}]")
        relative = relative[:limit]
    absolute = [(data_root / path).resolve() for path in relative]
    missing = [str(path) for path in absolute if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} selected images; first={missing[0]}")
    return relative, absolute, sha256_lines(relative)


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return stable distribution statistics for a finite numeric sequence."""
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    if not np.isfinite(array).all():
        raise FloatingPointError("diagnostic values contain NaN or Inf")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def prediction_summary(predictions: Sequence[Mapping[str, Any]], image_ids: Sequence[str]) -> dict[str, Any]:
    """Summarize confidence, class usage, and detections per image."""
    counts = {str(int(image_id)): 0 for image_id in image_ids}
    class_histogram: dict[str, int] = {}
    confidences = []
    for prediction in predictions:
        image_id = str(int(prediction["image_id"]))
        counts[image_id] = counts.get(image_id, 0) + 1
        category = str(int(prediction["category_id"]))
        class_histogram[category] = class_histogram.get(category, 0) + 1
        confidences.append(float(prediction["score"]))
    ordered_histogram = dict(sorted(class_histogram.items(), key=lambda item: int(item[0])))
    return {
        "total_detections": len(predictions),
        "images": len(image_ids),
        "zero_detection_images": sum(value == 0 for value in counts.values()),
        "detections_per_image": numeric_summary(list(counts.values())),
        "confidence": numeric_summary(confidences),
        "coco_category_histogram": ordered_histogram,
    }


def _bbox_context(dataset: Any) -> dict[str, dict[str, Any]]:
    """Describe each image using original-space bbox area bins."""
    result = {}
    for im_file, label in zip(dataset.im_files, dataset.labels):
        height, width = (int(value) for value in label["shape"])
        boxes = np.asarray(label["bboxes"], dtype=np.float64).reshape(-1, 4)
        areas = boxes[:, 2] * width * boxes[:, 3] * height if boxes.size else np.zeros(0)
        counts = {
            "small": int((areas < 32**2).sum()),
            "medium": int(((areas >= 32**2) & (areas < 96**2)).sum()),
            "large": int((areas >= 96**2).sum()),
        }
        largest = "none"
        if counts["small"]:
            largest = "small"
        if counts["medium"]:
            largest = "medium"
        if counts["large"]:
            largest = "large"
        result[Path(im_file).stem] = {
            "object_count": len(areas),
            "bbox_area_counts": counts,
            "largest_bbox_scale": largest,
        }
    return result


def router_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image routing probabilities globally and by object scale."""
    result: dict[str, Any] = {}
    for level in ROUTING_LEVELS:
        rows = [record["routing"][level] for record in records]
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
        entropy = np.asarray([row["entropy"] for row in rows], dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(f"{level} routing probabilities must be [N,E]")
        if not np.isfinite(probabilities).all() or not np.isfinite(entropy).all():
            raise FloatingPointError(f"{level} routing diagnostics contain NaN or Inf")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError(f"{level} routing probabilities are not normalized")
        top1 = probabilities.argmax(axis=1)
        experts = probabilities.shape[1]
        by_scale = {}
        for scale in ("none", "small", "medium", "large"):
            indices = [index for index, record in enumerate(records) if record["largest_bbox_scale"] == scale]
            if not indices:
                continue
            subset = probabilities[indices]
            subset_entropy = entropy[indices]
            by_scale[scale] = {
                "images": len(indices),
                "mean_probabilities": subset.mean(axis=0).tolist(),
                "mean_entropy": float(subset_entropy.mean()),
            }
        result[level] = {
            "images": len(records),
            "num_experts": experts,
            "mean_probabilities": probabilities.mean(axis=0).tolist(),
            "std_probabilities": probabilities.std(axis=0).tolist(),
            "entropy": numeric_summary(entropy.tolist()),
            "mean_normalized_entropy": float(entropy.mean() / math.log(experts)),
            "top1_counts": np.bincount(top1, minlength=experts).astype(int).tolist(),
            "top1_fractions": (np.bincount(top1, minlength=experts) / max(len(top1), 1)).tolist(),
            "by_largest_bbox_scale": by_scale,
        }
    return result


class DiagnosticValidator(D1FoundationDetectionValidator):
    """D1 cache validator that always emits correctly mapped COCO JSON."""

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.diagnostic_sample_ids = tuple(batch["sample_id"])
        return super().preprocess(batch)

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.is_coco = True
        self.is_lvis = False
        self.class_map = converter.coco80_to_coco91_class()
        self.args.save_json = True


class RouterCollector:
    """Collect full image-level Router probabilities after each validation batch."""

    def __init__(self, model: D1FoundationDetectionModel, context: Mapping[str, Mapping[str, Any]]) -> None:
        self.model = model
        self.context = context
        self.records: list[dict[str, Any]] = []

    def on_val_batch_end(self, validator: DiagnosticValidator) -> None:
        sample_ids = tuple(getattr(validator, "diagnostic_sample_ids", ()))
        if not sample_ids:
            raise RuntimeError("diagnostic validator did not expose sample IDs")
        batch_records = []
        for sample_id in sample_ids:
            stem = str(sample_id).split("/")[-1]
            batch_records.append(
                {
                    "sample_id": str(sample_id),
                    **self.context[stem],
                    "routing": {},
                }
            )
        for level in ROUTING_LEVELS:
            probabilities = self.model.mixtures[level].routing_probs
            if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2:
                raise RuntimeError(f"{level} did not expose [B,E] routing probabilities")
            probabilities = probabilities.detach().float().cpu()
            if probabilities.shape[0] != len(sample_ids):
                raise RuntimeError(f"{level} Router batch size differs from input batch")
            if not bool(torch.isfinite(probabilities).all()):
                raise FloatingPointError(f"{level} Router probabilities contain NaN or Inf")
            entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            for index, record in enumerate(batch_records):
                record["routing"][level] = {
                    "probabilities": probabilities[index].tolist(),
                    "entropy": float(entropy[index]),
                    "top1_expert": int(probabilities[index].argmax()),
                }
        self.records.extend(batch_records)


def _strict_checkpoint_model(checkpoint_path: Path) -> tuple[D1FoundationDetectionModel, dict[str, Any]]:
    loaded, checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    if not isinstance(loaded, D1FoundationDetectionModel):
        raise TypeError("checkpoint does not contain D1FoundationDetectionModel")
    restored = D1FoundationDetectionModel(loaded.config_dict()).eval()
    initialize_mixture_loss_ema_buffer(restored)
    restored.load_state_dict(loaded.float().state_dict(), strict=True)
    teacher_keys = [name for name in restored.state_dict() if "teacher" in name.lower() or "dinov3" in name.lower()]
    if teacher_keys:
        raise ValueError(f"checkpoint unexpectedly contains Teacher parameters: {teacher_keys[:3]}")
    return restored, {
        "filename": checkpoint_path.name,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "strict_reload": True,
        "teacher_parameter_count": 0,
    }


def _cache_identity(cache_dir: Path) -> dict[str, Any]:
    reader = open_feature_cache(cache_dir, max_open_shards=1)
    is_npy = reader.index.get("schema_version") == NPY_SCHEMA_VERSION
    return {
        "path": str(cache_dir),
        "sample_count": len(reader.records),
        "contract_sha256": reader.index["contract_sha256"],
        "content_sha256": reader.index["source_content_sha256" if is_npy else "content_sha256"],
        "shard_count": 0 if is_npy else len(reader.index["shards"]),
        "format": reader.index.get("schema_version"),
    }


def _build_inputs(args: argparse.Namespace, split: str, limit: int) -> tuple[Path, list[str], str]:
    relative, _absolute, paths_sha256 = select_split_paths(args.repo_root, args.data_root, split, limit)
    inputs_dir = args.output_dir / "inputs"
    list_path = inputs_dir / f"{split}-{len(relative)}.txt"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    view_root = args.output_dir / "dataset-view"
    for kind in ("images", "labels"):
        target = (args.data_root / kind / split).resolve()
        link = view_root / kind / split
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError(f"diagnostic dataset link points to the wrong target: {link}")
        elif link.exists():
            raise RuntimeError(f"diagnostic dataset link path is occupied: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    view_paths = [view_root / path for path in relative]
    content = "".join(f"{path}\n" for path in view_paths)
    if not list_path.is_file() or list_path.read_text(encoding="utf-8") != content:
        temporary = list_path.with_name(f".{list_path.name}.part")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, list_path)
    template = YAML.load(args.repo_root / "ultralytics/cfg/datasets/coco.yaml")
    data_yaml = inputs_dir / f"{split}-{len(relative)}.yaml"
    YAML.save(
        data_yaml,
        {
            "path": str(view_root),
            "train": str(list_path),
            "val": str(list_path),
            "names": template["names"],
        },
    )
    return data_yaml, relative, paths_sha256


def _trainer_overrides(args: argparse.Namespace, data_yaml: Path, split: str) -> dict[str, Any]:
    overrides = YAML.load(args.repo_root / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml")
    overrides.update(
        {
            "data": str(data_yaml),
            "model": str((args.repo_root / "ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml").resolve()),
            "device": str(args.device),
            "batch": args.batch,
            "nbs": args.batch,
            "workers": args.workers,
            "epochs": 1,
            "val": True,
            "save": False,
            "save_json": True,
            "plots": False,
            "pretrained": False,
            "project": str(args.output_dir / "runtime"),
            "name": split,
            "exist_ok": True,
            "verbose": False,
        }
    )
    return overrides


def evaluate_split(args: argparse.Namespace, split: str, limit: int, cache_dir: Path) -> dict[str, Any]:
    report_dir = args.output_dir / split
    report_path = report_dir / "report.json"
    if report_path.is_file() and not args.force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") == "passed" and existing.get("checkpoint", {}).get("sha256") == sha256_file(
            args.checkpoint
        ):
            return existing
        raise RuntimeError(f"{report_path} exists but is not reusable; pass --force for a new evaluation")
    if importlib.util.find_spec("faster_coco_eval") is None:
        raise RuntimeError("faster-coco-eval>=1.6.7 is required for official COCO diagnostics")

    data_yaml, relative_paths, paths_sha256 = _build_inputs(args, split, limit)
    selected_ids = [Path(path).stem for path in relative_paths]
    cache_identity = _cache_identity(cache_dir)
    if cache_identity["sample_count"] < limit:
        raise ValueError(f"{split} cache has fewer than {limit} records")
    overrides = _trainer_overrides(args, data_yaml, split)
    trainer = D1FoundationDetectionTrainer(
        overrides=overrides,
        feature_caches={"train": cache_dir, "val": cache_dir},
        trusted_feature_cache=True,
        max_open_feature_shards=args.max_open_shards,
        feature_prefetch_factor=args.prefetch_factor,
    )
    model, checkpoint = _strict_checkpoint_model(args.checkpoint)
    trainer.model = model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.model.eval()
    trainer.amp = True
    trainer.world_size = 1
    trainer.epoch = 0
    trainer.epochs = 1
    trainer.stopper = SimpleNamespace(possible_stop=True)
    trainer.ema = SimpleNamespace(ema=None)
    trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", *D1_AUX_REPORT_NAMES, "mixture_aux_loss")
    trainer.loss_items = torch.zeros(len(trainer.loss_names), device=trainer.device)

    validator = DiagnosticValidator(
        dataloader=None,
        save_dir=report_dir,
        args=copy(trainer.args),
        _callbacks=trainer.callbacks,
        feature_cache=cache_dir,
        trusted_cache=True,
        max_open_shards=args.max_open_shards,
        prefetch_factor=args.prefetch_factor,
    )
    validator.data = trainer.data
    validator.dataloader = validator.get_dataloader(trainer.data["val"], args.batch)
    context = _bbox_context(validator.dataloader.dataset)
    collector = RouterCollector(trainer.model, context)
    validator.callbacks["on_val_batch_end"].append(collector.on_val_batch_end)
    torch.cuda.reset_peak_memory_stats(trainer.device)
    started = time.perf_counter()
    internal_metrics = validator(trainer=trainer)
    elapsed = time.perf_counter() - started
    if validator.seen != limit or len(collector.records) != limit:
        raise RuntimeError(
            f"{split} processed an incomplete sample set: seen={validator.seen}, router={len(collector.records)}"
        )

    predictions_path = report_dir / "predictions.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_part = predictions_path.with_name(f".{predictions_path.name}.part")
    with predictions_part.open("w", encoding="utf-8") as stream:
        json.dump(validator.jdict, stream, separators=(",", ":"), ensure_ascii=True)
    os.replace(predictions_part, predictions_path)
    annotation_path = args.data_root / "annotations" / f"instances_{split}.json"
    official_metrics = validator.coco_evaluate(dict(internal_metrics), predictions_path, annotation_path)
    required_official = (
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP_small(B)",
        "metrics/mAP_medium(B)",
        "metrics/mAP_large(B)",
    )
    missing_metrics = [name for name in required_official if name not in official_metrics]
    if missing_metrics:
        raise RuntimeError(f"official COCO evaluator did not return {missing_metrics}")
    numeric_metrics = [float(value) for value in official_metrics.values() if isinstance(value, (int, float))]
    if not all(math.isfinite(value) for value in numeric_metrics):
        raise FloatingPointError("validation metrics contain NaN or Inf")

    router_samples_path = report_dir / "router-samples.jsonl"
    write_jsonl(router_samples_path, collector.records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "split": split,
        "selected_sample_count": limit,
        "selected_paths_sha256": paths_sha256,
        "code": git_state(args.repo_root),
        "checkpoint": checkpoint,
        "cache": cache_identity,
        "runtime": {
            "device": str(trainer.device),
            "gpu_name": torch.cuda.get_device_name(trainer.device),
            "batch": args.batch,
            "workers": args.workers,
            "elapsed_seconds": elapsed,
            "images_per_second": limit / elapsed,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(trainer.device),
        },
        "internal_metrics": internal_metrics,
        "official_coco_metrics": official_metrics,
        "predictions": {
            **prediction_summary(validator.jdict, selected_ids),
            "file": predictions_path.name,
            "bytes": predictions_path.stat().st_size,
            "sha256": sha256_file(predictions_path),
        },
        "routing": router_summary(collector.records),
        "router_samples": {
            "file": router_samples_path.name,
            "bytes": router_samples_path.stat().st_size,
            "sha256": sha256_file(router_samples_path),
        },
        "bbox_area_note": "Router groups use original-space bbox area thresholds, not COCO segmentation area.",
    }
    write_json(report_path, report)
    return report


def _last_training_row(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no result rows")
    return {str(key).strip(): float(value) for key, value in rows[-1].items() if value not in (None, "")}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    val = json.loads((args.output_dir / "val2017/report.json").read_text(encoding="utf-8"))
    train = json.loads((args.output_dir / "train2017/report.json").read_text(encoding="utf-8"))
    baseline_path = args.training_results or args.checkpoint.parent.parent / "results.csv"
    baseline = _last_training_row(baseline_path)
    comparison = {}
    for key in METRIC_KEYS:
        observed = float(val["internal_metrics"][key])
        expected = float(baseline[key])
        comparison[key] = {
            "training_log": expected,
            "independent_validation": observed,
            "absolute_difference": abs(observed - expected),
        }
    reproduced = all(item["absolute_difference"] <= args.reproduction_tolerance for item in comparison.values())
    val_map = float(val["official_coco_metrics"]["metrics/mAP50-95(B)"])
    train_map = float(train["official_coco_metrics"]["metrics/mAP50-95(B)"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "checkpoint_sha256": val["checkpoint"]["sha256"],
        "independent_validation_reproduces_training_log": reproduced,
        "reproduction_tolerance": args.reproduction_tolerance,
        "metric_comparison": comparison,
        "official_coco": {
            "val_mAP50_95": val_map,
            "train5000_mAP50_95": train_map,
            "absolute_generalization_gap": train_map - val_map,
            "train_to_val_ratio": train_map / val_map if val_map else None,
            "val_AP_small": val["official_coco_metrics"]["metrics/mAP_small(B)"],
            "val_AP_medium": val["official_coco_metrics"]["metrics/mAP_medium(B)"],
            "val_AP_large": val["official_coco_metrics"]["metrics/mAP_large(B)"],
        },
        "interpretation_contract": {
            "low_train_and_low_val": "underfitting, optimization, or feature/spatial bottleneck",
            "high_train_and_low_val": "generalization gap; augmentation and regularization become primary suspects",
            "small_AP_far_below_medium_large": "consistent with missing native stride-8 detail; not proof by itself",
            "router_entropy_near_log4": "weak expert specialization; compare against a fixed-uniform ablation next",
        },
    }
    write_json(args.output_dir / "summary.json", report)
    return report


def resolved_args(args: argparse.Namespace) -> argparse.Namespace:
    args.repo_root = args.repo_root.resolve()
    args.workspace = args.workspace.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.data_root = (args.data_root or args.workspace / "datasets/coco").resolve()
    args.train_cache = (args.train_cache or args.workspace / "feature_cache/coco2017-train2017-d1-cache-v1").resolve()
    args.val_cache = (args.val_cache or args.workspace / "feature_cache/coco2017-val2017-d1-cache-v1").resolve()
    args.output_dir = args.output_dir.resolve()
    if args.training_results is not None:
        args.training_results = args.training_results.resolve()
    for path in (args.repo_root, args.workspace, args.data_root, args.train_cache, args.val_cache):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("WP8 checkpoint diagnostics require CUDA")
    return args


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("val", "train-subset", "all", "summarize"))
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--train-cache", type=Path)
    result.add_argument("--val-cache", type=Path)
    result.add_argument("--training-results", type=Path)
    result.add_argument("--device", default="0")
    result.add_argument("--batch", type=int, default=128)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--max-open-shards", type=int, default=8)
    result.add_argument("--prefetch-factor", type=int, default=1)
    result.add_argument("--train-limit", type=int, default=5_000)
    result.add_argument("--reproduction-tolerance", type=float, default=1e-4)
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = resolved_args(parser().parse_args())
    if args.command == "val":
        report = evaluate_split(args, "val2017", EXPECTED_SPLIT_COUNTS["val2017"], args.val_cache)
    elif args.command == "train-subset":
        report = evaluate_split(args, "train2017", args.train_limit, args.train_cache)
    elif args.command == "summarize":
        report = summarize(args)
    else:
        evaluate_split(args, "val2017", EXPECTED_SPLIT_COUNTS["val2017"], args.val_cache)
        evaluate_split(args, "train2017", args.train_limit, args.train_cache)
        report = summarize(args)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()

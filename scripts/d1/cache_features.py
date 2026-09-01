#!/usr/bin/env python3
"""Build, verify, and compare D1 DINOv3 feature caches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.nn.foundation import DINOv3Teacher
from ultralytics.nn.foundation.cache import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_TARGET_SHARD_BYTES,
    FeatureCacheReader,
    FeatureCacheWriter,
    canonical_json_bytes,
    compare_feature_caches,
    sha256_bytes,
    sha256_file,
    verify_feature_cache,
)


DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
OUTPUT_LAYERS = (4, 8, 12)
FEATURE_NAMES = ("block4", "block8", "block12")
EXPECTED_SHAPE = (384, 40, 40)


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a stable JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode() + b"\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()


def cache_contract(repo_root: Path) -> dict[str, Any]:
    """Derive the cache contract only from tracked WP0 manifests."""
    manifests = repo_root / "experiments" / "d1" / "manifests"
    p0 = load_json(manifests / "p0-experiment-contract.json")
    teacher = load_json(manifests / "dinov3-vits16.json")
    if p0["cache"]["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("WP0 and cache implementation schema versions differ.")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_id": teacher["model_id"],
        "teacher_weights_sha256": teacher["files"]["model.safetensors"]["sha256"],
        "preprocessing_sha256": sha256_bytes(canonical_json_bytes(p0["input"])),
        "output_layers": list(OUTPUT_LAYERS),
        "feature_names": list(FEATURE_NAMES),
        "dtype": p0["cache"]["dtype"],
        "expected_shape": list(EXPECTED_SHAPE),
    }


def split_paths(repo_root: Path, split: str, limit: int | None) -> tuple[list[str], str]:
    path = repo_root / "experiments" / "d1" / "manifests" / f"coco2017-{split}.txt"
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if entries != sorted(entries):
        raise ValueError(f"split manifest is not sorted: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"split manifest contains duplicate paths: {path}")
    if any(Path(entry).is_absolute() or ".." in Path(entry).parts for entry in entries):
        raise ValueError(f"split manifest contains a non-portable path: {path}")
    selected = entries if limit is None else entries[:limit]
    data = ("\n".join(selected) + "\n").encode()
    return selected, sha256_bytes(data)


def make_letterbox() -> LetterBox:
    return LetterBox(
        new_shape=(640, 640),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        stride=32,
        padding_value=114,
        interpolation=cv2.INTER_LINEAR,
    )


def load_image(path: Path, letterbox: LetterBox) -> torch.Tensor:
    """Read BGR input, apply the WP0 letterbox, and return RGB CHW in [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {path}")
    image = letterbox(image=image)
    if image.shape != (640, 640, 3):
        raise ValueError(f"letterbox returned {image.shape} for {path}")
    rgb_chw = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb_chw).float().div_(255.0)


def normalize_device(value: str) -> str:
    if value.isdigit():
        return f"cuda:{value}"
    return value


def selected_samples(
    data_root: Path,
    split: str,
    paths: list[str],
    writer: FeatureCacheWriter,
) -> tuple[list[dict[str, Any]], int]:
    samples = []
    resumed = 0
    for relative_path in paths:
        image_path = data_root / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_sha256 = sha256_file(image_path)
        sample_id = f"{split}/{image_path.stem}"
        if writer.is_cached(sample_id, image_sha256):
            resumed += 1
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "split": split,
                "image_path": relative_path,
                "path": image_path,
                "image_sha256": image_sha256,
            }
        )
    return samples, resumed


def benchmark_reader(cache_dir: Path, sample_ids: list[str]) -> dict[str, Any]:
    reader = FeatureCacheReader(cache_dir)
    start = time.perf_counter()
    tensor_bytes = 0
    for sample_id in sample_ids:
        features = reader.get(sample_id)
        for tensor in features.values():
            tensor_bytes += tensor.numel() * tensor.element_size()
    elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "tensor_bytes": tensor_bytes,
        "mib_per_second": tensor_bytes / 1024**2 / elapsed if elapsed else None,
        "images_per_second": len(sample_ids) / elapsed if elapsed else None,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    workspace = args.workspace.resolve()
    data_root = (args.data_root or workspace / "datasets" / "coco").resolve()
    weights_dir = (
        args.weights_dir or workspace / "weights" / "teachers" / "dinov3-vits16-pretrain-lvd1689m"
    ).resolve()
    cache_dir = args.cache_dir.resolve()
    contract = cache_contract(repo_root)
    paths, selected_paths_sha256 = split_paths(repo_root, args.split, args.limit)
    if not paths:
        raise ValueError("selected split is empty.")
    if not weights_dir.joinpath("model.safetensors").is_file():
        raise FileNotFoundError(weights_dir / "model.safetensors")
    if sha256_file(weights_dir / "model.safetensors") != contract["teacher_weights_sha256"]:
        raise ValueError("teacher weights do not match the tracked WP0 manifest.")

    writer = FeatureCacheWriter(
        cache_dir,
        split=args.split,
        contract=contract,
        target_shard_bytes=args.target_shard_bytes,
    )
    pending, resumed = selected_samples(data_root, args.split, paths, writer)
    device = normalize_device(args.device)
    peak_gpu_bytes = 0
    extraction_start = time.perf_counter()
    if pending:
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable.")
            torch.cuda.set_device(torch.device(device))
            torch.cuda.reset_peak_memory_stats()
        teacher = DINOv3Teacher(
            model_id=DEFAULT_MODEL_ID,
            weights_path=weights_dir,
            local_files_only=True,
            dtype="fp16",
            device=device,
            output_layers=OUTPUT_LAYERS,
        )
        letterbox = make_letterbox()
        for offset in range(0, len(pending), args.batch_size):
            batch_samples = pending[offset : offset + args.batch_size]
            images = torch.stack([load_image(sample["path"], letterbox) for sample in batch_samples])
            features = teacher.encode(images)
            if tuple(features.dense) != FEATURE_NAMES:
                raise ValueError(f"teacher returned unexpected features: {tuple(features.dense)}")
            for batch_index, sample in enumerate(batch_samples):
                writer.add(
                    sample_id=sample["sample_id"],
                    split=sample["split"],
                    image_path=sample["image_path"],
                    image_sha256=sample["image_sha256"],
                    features={name: features.dense[name][batch_index] for name in FEATURE_NAMES},
                )
        writer.close()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_gpu_bytes = torch.cuda.max_memory_allocated()
        del teacher
    else:
        writer.close()
    extraction_seconds = time.perf_counter() - extraction_start

    verification = verify_feature_cache(cache_dir)
    sample_ids = [f"{args.split}/{Path(path).stem}" for path in paths]
    missing = sorted(set(sample_ids) - set(FeatureCacheReader(cache_dir).records))
    if missing:
        raise ValueError(f"cache build is missing selected samples: {missing[:5]}")
    read_benchmark = benchmark_reader(cache_dir, sample_ids)
    report = {
        "schema_version": "d1-wp2-build-report-v1",
        "code_commit": git_commit(repo_root),
        "cache_id": cache_dir.name,
        "split": args.split,
        "selected_sample_count": len(paths),
        "selected_paths_sha256": selected_paths_sha256,
        "new_sample_count": len(pending),
        "resumed_sample_count": resumed,
        "batch_size": args.batch_size,
        "target_shard_bytes": args.target_shard_bytes,
        "contract": contract,
        "verification": verification,
        "metrics": {
            "extraction_seconds": extraction_seconds,
            "new_images_per_second": len(pending) / extraction_seconds if extraction_seconds else None,
            "peak_gpu_bytes": peak_gpu_bytes,
            "read": read_benchmark,
        },
        "online_cache_validation": "exact FP16 tensor SHA256 verified after safetensors reload",
    }
    if args.report:
        write_json(args.report, report)
    return report


def verify(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "schema_version": "d1-wp2-verify-report-v1",
        "cache_id": args.cache_dir.resolve().name,
        "verification": verify_feature_cache(args.cache_dir, full_tensor_hash=not args.metadata_only),
    }
    if args.report:
        write_json(args.report, report)
    return report


def compare(args: argparse.Namespace) -> dict[str, Any]:
    comparison = compare_feature_caches(args.cache_dir, args.other_cache_dir)
    report = {
        "schema_version": "d1-wp2-reproducibility-v1",
        "code_commit": git_commit(args.repo_root.resolve()),
        "first_cache_id": args.cache_dir.resolve().name,
        "second_cache_id": args.other_cache_dir.resolve().name,
        "comparison": comparison,
    }
    if args.first_report:
        report["first_build"] = load_json(args.first_report)
    if args.second_report:
        report["second_build"] = load_json(args.second_report)
    if args.report:
        write_json(args.report, report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build or resume one deterministic cache")
    build_parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    build_parser.add_argument("--workspace", type=Path, required=True)
    build_parser.add_argument("--data-root", type=Path)
    build_parser.add_argument("--weights-dir", type=Path)
    build_parser.add_argument("--cache-dir", type=Path, required=True)
    build_parser.add_argument("--split", choices=("train2017", "val2017"), default="train2017")
    build_parser.add_argument("--limit", type=int)
    build_parser.add_argument("--batch-size", type=int, default=8)
    build_parser.add_argument("--device", default="0")
    build_parser.add_argument("--target-shard-bytes", type=int, default=DEFAULT_TARGET_SHARD_BYTES)
    build_parser.add_argument("--report", type=Path)
    build_parser.set_defaults(handler=build)

    verify_parser = subparsers.add_parser("verify", help="verify an existing cache without model inference")
    verify_parser.add_argument("--cache-dir", type=Path, required=True)
    verify_parser.add_argument("--metadata-only", action="store_true")
    verify_parser.add_argument("--report", type=Path)
    verify_parser.set_defaults(handler=verify)

    compare_parser = subparsers.add_parser("compare", help="compare two independent verified cache builds")
    compare_parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    compare_parser.add_argument("--cache-dir", type=Path, required=True)
    compare_parser.add_argument("--other-cache-dir", type=Path, required=True)
    compare_parser.add_argument("--first-report", type=Path)
    compare_parser.add_argument("--second-report", type=Path)
    compare_parser.add_argument("--report", type=Path)
    compare_parser.set_defaults(handler=compare)
    return result


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("--batch-size must be positive.")
    report = args.handler(args)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()

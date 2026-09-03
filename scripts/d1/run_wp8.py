#!/usr/bin/env python3
"""Run deterministic multi-GPU cache preparation and benchmarks for D1 WP8."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.d1.cache_features import (
        DEFAULT_MODEL_ID,
        EXPECTED_SHAPE,
        FEATURE_NAMES,
        OUTPUT_LAYERS,
        cache_contract,
        load_image,
        make_letterbox,
        normalize_device,
        selected_samples,
        split_paths,
        write_json,
    )
except ModuleNotFoundError:  # Direct execution sets sys.path to scripts/d1.
    from cache_features import (
        DEFAULT_MODEL_ID,
        EXPECTED_SHAPE,
        FEATURE_NAMES,
        OUTPUT_LAYERS,
        cache_contract,
        load_image,
        make_letterbox,
        normalize_device,
        selected_samples,
        split_paths,
        write_json,
    )
from ultralytics.nn.foundation import DINOv3Teacher
from ultralytics.nn.foundation.cache import (
    DEFAULT_TARGET_SHARD_BYTES,
    FeatureCacheReader,
    FeatureCacheWriter,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_feature_cache,
)


SCHEMA_VERSION = "d1-wp8-cache-v1"
DEFAULT_DEVICES = "0,1,2,3,4,5"
DEFAULT_BATCH_CANDIDATES = (8, 16, 32)
DEFAULT_BENCHMARK_LIMIT = 1200
MIN_AGGREGATE_IMAGES_PER_SECOND = 30.0
MAX_ESTIMATED_TOTAL_SECONDS = 2 * 3600
FULL_COCO_COUNTS = {"train2017": 118_287, "val2017": 5_000}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def partition_paths(paths: list[str], rank: int, world_size: int) -> list[str]:
    """Return one deterministic, disjoint rank partition."""
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return paths[rank::world_size]


def paths_sha256(paths: list[str]) -> str:
    return sha256_bytes((("\n".join(paths) + "\n") if paths else "").encode())


def validate_split_membership(
    train_paths: list[str],
    val_paths: list[str],
    expected_counts: dict[str, int] = FULL_COCO_COUNTS,
) -> None:
    """Fail closed when official split membership is incomplete or overlapping."""
    actual_counts = {"train2017": len(train_paths), "val2017": len(val_paths)}
    if actual_counts != expected_counts:
        raise ValueError(f"COCO split counts differ: expected={expected_counts}, actual={actual_counts}")
    train_ids = {Path(path).stem for path in train_paths}
    val_ids = {Path(path).stem for path in val_paths}
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise ValueError(f"COCO train/val sample IDs overlap: {sorted(overlap)[:3]}")


def validate_official_splits(repo_root: Path) -> None:
    train_paths, _train_sha = split_paths(repo_root, "train2017", None)
    val_paths, _val_sha = split_paths(repo_root, "val2017", None)
    validate_split_membership(train_paths, val_paths)


def parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("--devices must contain unique comma-separated CUDA indices")
    if any(not device.isdigit() for device in devices):
        raise ValueError("WP8 multi-GPU cache workers require numeric CUDA device indices")
    return devices


def rank_parts_root(cache_dir: Path) -> Path:
    return cache_dir.parent / f".{cache_dir.name}.parts"


def rank_cache_dir(cache_dir: Path, rank: int) -> Path:
    return rank_parts_root(cache_dir) / f"rank-{rank:02d}"


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    workspace = args.workspace.resolve()
    data_root = (args.data_root or workspace / "datasets" / "coco").resolve()
    weights_dir = (
        args.weights_dir or workspace / "weights" / "teachers" / "dinov3-vits16-pretrain-lvd1689m"
    ).resolve()
    contract = cache_contract(repo_root)
    weight_path = weights_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    if sha256_file(weight_path) != contract["teacher_weights_sha256"]:
        raise ValueError("teacher weights do not match the tracked WP0 manifest")
    return repo_root, data_root, weights_dir, contract


def _load_teacher(weights_dir: Path, device: str) -> DINOv3Teacher:
    return DINOv3Teacher(
        model_id=DEFAULT_MODEL_ID,
        weights_path=weights_dir,
        local_files_only=True,
        dtype="fp16",
        device=device,
        output_layers=OUTPUT_LAYERS,
    )


def _encode_paths(
    teacher: DINOv3Teacher,
    paths: list[Path],
    batch_size: int,
    *,
    on_batch=None,
) -> int:
    letterbox = make_letterbox()
    processed = 0
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images = torch.stack([load_image(path, letterbox) for path in batch_paths])
        output = teacher.encode(images)
        if tuple(output.dense) != FEATURE_NAMES:
            raise ValueError(f"teacher returned unexpected features: {tuple(output.dense)}")
        if any(tuple(value.shape[1:]) != EXPECTED_SHAPE for value in output.dense.values()):
            raise ValueError("teacher returned an unexpected feature shape")
        if on_batch is not None:
            on_batch(offset, output.dense)
        processed += len(batch_paths)
    return processed


def worker_compute(args: argparse.Namespace) -> dict[str, Any]:
    repo_root, data_root, weights_dir, contract = _resolve_inputs(args)
    all_paths, selected_sha = split_paths(repo_root, args.split, args.limit)
    paths = partition_paths(all_paths, args.rank, args.world_size)
    device = normalize_device(args.device)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("WP8 compute benchmark requires CUDA")
    torch.cuda.set_device(torch.device(device))
    load_started = time.perf_counter()
    teacher = _load_teacher(weights_dir, device)
    model_load_seconds = time.perf_counter() - load_started
    absolute_paths = [data_root / path for path in paths]
    if any(not path.is_file() for path in absolute_paths):
        raise FileNotFoundError("benchmark input image is missing")
    warmup_paths = absolute_paths[: min(args.batch_size, len(absolute_paths))]
    if warmup_paths:
        _encode_paths(teacher, warmup_paths, args.batch_size)
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processed = _encode_paths(teacher, absolute_paths, args.batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": SCHEMA_VERSION,
        "operation": "compute",
        "status": "passed",
        "code_commit": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "split": args.split,
        "selected_sample_count": len(all_paths),
        "selected_paths_sha256": selected_sha,
        "rank": args.rank,
        "world_size": args.world_size,
        "partition_sample_count": len(paths),
        "partition_paths_sha256": paths_sha256(paths),
        "batch_size": args.batch_size,
        "device_index": str(args.device),
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
        "metrics": {
            "model_load_seconds": model_load_seconds,
            "extraction_seconds": elapsed,
            "images_per_second": processed / elapsed if elapsed else None,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    write_json(args.report, report)
    return report


def worker_build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root, data_root, weights_dir, contract = _resolve_inputs(args)
    all_paths, selected_sha = split_paths(repo_root, args.split, args.limit)
    paths = partition_paths(all_paths, args.rank, args.world_size)
    cache_dir = args.cache_dir.resolve()
    writer = FeatureCacheWriter(
        cache_dir,
        split=args.split,
        contract=contract,
        target_shard_bytes=args.target_shard_bytes,
        shard_prefix=f"{args.split}-r{args.rank:02d}",
    )
    pending, resumed = selected_samples(data_root, args.split, paths, writer)
    device = normalize_device(args.device)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("WP8 cache build requires CUDA")
    torch.cuda.set_device(torch.device(device))
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    teacher = _load_teacher(weights_dir, device) if pending else None
    model_load_seconds = time.perf_counter() - load_started
    started = time.perf_counter()
    if pending:
        absolute_paths = [sample["path"] for sample in pending]

        def add_batch(offset: int, dense: dict[str, torch.Tensor]) -> None:
            batch_samples = pending[offset : offset + args.batch_size]
            for batch_index, sample in enumerate(batch_samples):
                writer.add(
                    sample_id=sample["sample_id"],
                    split=sample["split"],
                    image_path=sample["image_path"],
                    image_sha256=sample["image_sha256"],
                    features={name: dense[name][batch_index] for name in FEATURE_NAMES},
                )

        _encode_paths(teacher, absolute_paths, args.batch_size, on_batch=add_batch)
        torch.cuda.synchronize()
    writer.close()
    elapsed = time.perf_counter() - started
    verification = verify_feature_cache(cache_dir)
    records = FeatureCacheReader(cache_dir).records
    expected_ids = {f"{args.split}/{Path(path).stem}" for path in paths}
    if set(records) != expected_ids:
        raise ValueError("rank cache records do not match its deterministic partition")
    report = {
        "schema_version": SCHEMA_VERSION,
        "operation": "build-part",
        "status": "passed",
        "code_commit": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "split": args.split,
        "selected_sample_count": len(all_paths),
        "selected_paths_sha256": selected_sha,
        "rank": args.rank,
        "world_size": args.world_size,
        "partition_sample_count": len(paths),
        "partition_paths_sha256": paths_sha256(paths),
        "batch_size": args.batch_size,
        "target_shard_bytes": args.target_shard_bytes,
        "device_index": str(args.device),
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "contract": contract,
        "contract_sha256": sha256_bytes(canonical_json_bytes(contract)),
        "new_sample_count": len(pending),
        "resumed_sample_count": resumed,
        "verification": verification,
        "metrics": {
            "model_load_seconds": model_load_seconds,
            "extraction_seconds": elapsed,
            "images_per_second": len(pending) / elapsed if elapsed else None,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    write_json(args.report, report)
    return report


def validate_rank_reports(
    reports: list[dict[str, Any]],
    paths: list[str],
    *,
    split: str,
    world_size: int,
    code_commit: str,
    operation: str = "build-part",
    target_shard_bytes: int | None = None,
) -> None:
    if len(reports) != world_size:
        raise ValueError(f"expected {world_size} rank reports, got {len(reports)}")
    selected_sha = paths_sha256(paths)
    for rank, report in enumerate(reports):
        expected_partition = partition_paths(paths, rank, world_size)
        expected = {
            "schema_version": SCHEMA_VERSION,
            "operation": operation,
            "status": "passed",
            "code_commit": code_commit,
            "split": split,
            "selected_sample_count": len(paths),
            "selected_paths_sha256": selected_sha,
            "rank": rank,
            "world_size": world_size,
            "partition_sample_count": len(expected_partition),
            "partition_paths_sha256": paths_sha256(expected_partition),
        }
        if target_shard_bytes is not None:
            expected["target_shard_bytes"] = target_shard_bytes
        mismatches = [key for key, value in expected.items() if report.get(key) != value]
        if mismatches:
            raise ValueError(f"rank {rank} report mismatch: {', '.join(mismatches)}")


def finalize_rank_caches(
    *,
    repo_root: Path,
    cache_dir: Path,
    split: str,
    limit: int | None,
    world_size: int,
    reports: list[dict[str, Any]],
    target_shard_bytes: int,
) -> dict[str, Any]:
    paths, _selected_sha = split_paths(repo_root, split, limit)
    code_commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    validate_rank_reports(
        reports,
        paths,
        split=split,
        world_size=world_size,
        code_commit=code_commit,
        target_shard_bytes=target_shard_bytes,
    )
    part_root = rank_parts_root(cache_dir)
    part_files = sorted(path.as_posix() for path in part_root.rglob("*.part")) if part_root.exists() else []
    if part_files:
        raise ValueError(f"rank caches contain unfinished .part files: {part_files[:3]}")
    expected_ids = {f"{split}/{Path(path).stem}" for path in paths}
    seen_ids: set[str] = set()
    shard_sources: dict[str, Path] = {}
    contract = None
    for rank, report in enumerate(reports):
        part_dir = rank_cache_dir(cache_dir, rank)
        verification = verify_feature_cache(part_dir)
        if verification != report["verification"]:
            raise ValueError(f"rank {rank} cache verification changed after its report was written")
        reader = FeatureCacheReader(part_dir)
        rank_ids = set(reader.records)
        if seen_ids.intersection(rank_ids):
            raise ValueError(f"rank {rank} cache duplicates samples from another rank")
        seen_ids.update(rank_ids)
        contract = reader.contract if contract is None else contract
        if reader.contract != contract:
            raise ValueError(f"rank {rank} cache uses a different contract")
        for shard in reader.index["shards"]:
            name = shard["filename"]
            if name in shard_sources:
                raise ValueError(f"duplicate shard filename {name}")
            shard_sources[name] = part_dir / name
    if seen_ids != expected_ids:
        raise ValueError(
            f"rank cache union mismatch: missing={len(expected_ids - seen_ids)}, extra={len(seen_ids - expected_ids)}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    actual_names = {path.name for path in cache_dir.glob("*.safetensors")}
    unexpected = actual_names - set(shard_sources)
    if unexpected:
        raise ValueError(f"final cache contains unexpected shards: {sorted(unexpected)[:3]}")
    link_started = time.perf_counter()
    for name, source in sorted(shard_sources.items()):
        destination = cache_dir / name
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size or sha256_file(destination) != sha256_file(source):
                raise ValueError(f"existing final shard differs from rank shard: {name}")
            continue
        try:
            os.link(source, destination)
        except OSError as exc:
            raise OSError(
                f"hard-linking {name} into the final cache failed; refusing to duplicate cache bytes"
            ) from exc
    link_seconds = time.perf_counter() - link_started
    writer = FeatureCacheWriter(
        cache_dir,
        split=split,
        contract=contract,
        target_shard_bytes=target_shard_bytes,
    )
    writer.close()
    verify_started = time.perf_counter()
    verification = verify_feature_cache(cache_dir)
    verify_seconds = time.perf_counter() - verify_started
    if set(FeatureCacheReader(cache_dir).records) != expected_ids:
        raise ValueError("final cache index does not contain the expected sample set")
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": "finalize",
        "status": "passed",
        "code_commit": code_commit,
        "split": split,
        "world_size": world_size,
        "selected_sample_count": len(paths),
        "selected_paths_sha256": paths_sha256(paths),
        "rank_reports_sha256": sha256_bytes(canonical_json_bytes(reports)),
        "link_seconds": link_seconds,
        "verification_seconds": verify_seconds,
        "verification": verification,
    }


def _worker_command(
    args: argparse.Namespace,
    operation: str,
    rank: int,
    device: str,
    batch_size: int,
    report: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        operation,
        "--repo-root",
        str(args.repo_root.resolve()),
        "--workspace",
        str(args.workspace.resolve()),
        "--split",
        args.split,
        "--rank",
        str(rank),
        "--world-size",
        str(len(parse_devices(args.devices))),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--report",
        str(report),
    ]
    if args.data_root:
        command.extend(("--data-root", str(args.data_root.resolve())))
    if args.weights_dir:
        command.extend(("--weights-dir", str(args.weights_dir.resolve())))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if operation == "worker-build":
        command.extend(("--cache-dir", str(rank_cache_dir(args.cache_dir.resolve(), rank))))
        command.extend(("--target-shard-bytes", str(args.target_shard_bytes)))
    return command


def launch_workers(
    args: argparse.Namespace,
    operation: str,
    batch_size: int,
    label: str,
) -> tuple[list[dict[str, Any]], float]:
    devices = parse_devices(args.devices)
    runtime_root = args.workspace.resolve() / "manifests" / "wp8-runtime" / label
    log_root = args.workspace.resolve() / "logs" / "wp8" / label
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    started = time.perf_counter()
    try:
        for rank, device in enumerate(devices):
            report = runtime_root / f"rank-{rank:02d}.json"
            log_path = log_root / f"rank-{rank:02d}.log"
            stream = log_path.open("w", encoding="utf-8")
            streams.append(stream)
            env = dict(os.environ)
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            process = subprocess.Popen(
                _worker_command(args, operation, rank, device, batch_size, report),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
            )
            processes.append((rank, process, report, log_path))
        write_json(
            runtime_root / "pids.json",
            {"operation": operation, "pids": {str(rank): process.pid for rank, process, _, _ in processes}},
        )
        utilization_samples: dict[str, list[int]] = {device: [] for device in devices}
        while any(process.poll() is None for _rank, process, _report, _log in processes):
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                for line in output.splitlines():
                    device, utilization = (part.strip() for part in line.split(",", maxsplit=1))
                    if device in utilization_samples:
                        utilization_samples[device].append(int(utilization))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(1)
        failures = []
        for rank, process, _report, log_path in processes:
            if process.returncode:
                failures.append(f"rank {rank} exited {process.returncode}; see {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    finally:
        for stream in streams:
            stream.close()
    elapsed = time.perf_counter() - started
    reports = [load_json(report) for _rank, _process, report, _log in processes]
    for report, device in zip(reports, devices):
        samples = utilization_samples[device]
        report["metrics"]["gpu_utilization_percent"] = {
            "mean": sum(samples) / len(samples) if samples else None,
            "max": max(samples) if samples else None,
            "sample_count": len(samples),
        }
    return reports, elapsed


def aggregate_workers(reports: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    sample_count = sum(report["partition_sample_count"] for report in reports)
    extraction_seconds = max(report["metrics"]["extraction_seconds"] for report in reports)
    return {
        "sample_count": sample_count,
        "wall_seconds": wall_seconds,
        "parallel_extraction_seconds": extraction_seconds,
        "aggregate_images_per_second": sample_count / extraction_seconds,
        "end_to_end_images_per_second": sample_count / wall_seconds,
        "peak_gpu_bytes_by_rank": [report["metrics"]["peak_gpu_bytes"] for report in reports],
        "model_load_seconds_by_rank": [report["metrics"]["model_load_seconds"] for report in reports],
        "gpu_utilization_percent_by_rank": [
            report["metrics"].get("gpu_utilization_percent") for report in reports
        ],
    }


def benchmark_cache(args: argparse.Namespace) -> dict[str, Any]:
    validate_official_splits(args.repo_root.resolve())
    devices = parse_devices(args.devices)
    if len(devices) != 6:
        raise ValueError("the WP8 benchmark gate requires exactly six devices")
    candidates = []
    for batch_size in args.batch_candidates:
        try:
            reports, wall = launch_workers(args, "worker-compute", batch_size, f"compute-b{batch_size}")
            paths, _selected_sha = split_paths(args.repo_root.resolve(), args.split, args.limit)
            code_commit = subprocess.check_output(
                ["git", "-C", str(args.repo_root.resolve()), "rev-parse", "HEAD"], text=True
            ).strip()
            validate_rank_reports(
                reports,
                paths,
                split=args.split,
                world_size=len(devices),
                code_commit=code_commit,
                operation="compute",
            )
            aggregate = aggregate_workers(reports, wall)
            candidates.append({"batch_size": batch_size, "status": "passed", **aggregate})
        except Exception as exc:
            candidates.append(
                {"batch_size": batch_size, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
    passed = [candidate for candidate in candidates if candidate["status"] == "passed"]
    if not passed:
        raise RuntimeError("all six-GPU batch candidates failed")
    selected = max(passed, key=lambda item: item["aggregate_images_per_second"])
    build_reports, build_wall = launch_workers(
        args, "worker-build", selected["batch_size"], f"write-b{selected['batch_size']}"
    )
    build_aggregate = aggregate_workers(build_reports, build_wall)
    finalize_started = time.perf_counter()
    finalization = finalize_rank_caches(
        repo_root=args.repo_root.resolve(),
        cache_dir=args.cache_dir.resolve(),
        split=args.split,
        limit=args.limit,
        world_size=len(devices),
        reports=build_reports,
        target_shard_bytes=args.target_shard_bytes,
    )
    finalize_wall = time.perf_counter() - finalize_started
    full_samples = sum(FULL_COCO_COUNTS.values())
    images_per_second = build_aggregate["aggregate_images_per_second"]
    extraction_eta = full_samples / images_per_second
    verification_eta = finalization["verification_seconds"] * full_samples / args.limit
    total_eta = extraction_eta + verification_eta
    bytes_per_image = math.prod(EXPECTED_SHAPE) * 2 * len(FEATURE_NAMES)
    estimated_cache_bytes = bytes_per_image * full_samples
    free_bytes = os.statvfs(args.workspace.resolve()).f_bavail * os.statvfs(args.workspace.resolve()).f_frsize
    failures = []
    if images_per_second < MIN_AGGREGATE_IMAGES_PER_SECOND:
        failures.append(
            f"aggregate throughput {images_per_second:.3f} images/s is below {MIN_AGGREGATE_IMAGES_PER_SECOND:.1f}"
        )
    if total_eta > MAX_ESTIMATED_TOTAL_SECONDS:
        failures.append(f"estimated total time {total_eta:.1f}s exceeds {MAX_ESTIMATED_TOTAL_SECONDS}s")
    if free_bytes < int(estimated_cache_bytes * 1.4):
        failures.append("free storage is below the 1.4x cache safety margin")
    cache_bytes = finalization["verification"]["cache_bytes"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "operation": "benchmark-gate",
        "status": "passed" if not failures else "failed",
        "code_commit": finalization["code_commit"],
        "devices": list(devices),
        "world_size": len(devices),
        "split": args.split,
        "benchmark_sample_count": args.limit,
        "batch_candidates": candidates,
        "selected_batch_size": selected["batch_size"],
        "write_benchmark": {
            **build_aggregate,
            "cache_bytes": cache_bytes,
            "effective_cache_mib_per_second": cache_bytes / 1024**2 / build_aggregate["parallel_extraction_seconds"],
        },
        "finalization": finalization,
        "estimate": {
            "full_sample_count": full_samples,
            "cache_bytes": estimated_cache_bytes,
            "extraction_seconds": extraction_eta,
            "verification_seconds": verification_eta,
            "total_seconds": total_eta,
            "free_storage_bytes": free_bytes,
        },
        "thresholds": {
            "min_aggregate_images_per_second": MIN_AGGREGATE_IMAGES_PER_SECOND,
            "max_estimated_total_seconds": MAX_ESTIMATED_TOTAL_SECONDS,
            "storage_safety_factor": 1.4,
        },
        "failures": failures,
    }
    write_json(args.report, report)
    return report


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    validate_official_splits(args.repo_root.resolve())
    devices = parse_devices(args.devices)
    reports, wall = launch_workers(args, "worker-build", args.batch_size, f"full-{args.split}")
    finalization = finalize_rank_caches(
        repo_root=args.repo_root.resolve(),
        cache_dir=args.cache_dir.resolve(),
        split=args.split,
        limit=args.limit,
        world_size=len(devices),
        reports=reports,
        target_shard_bytes=args.target_shard_bytes,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "operation": "build-cache",
        "status": "passed",
        "batch_size": args.batch_size,
        "workers": aggregate_workers(reports, wall),
        "finalization": finalization,
    }
    write_json(args.report, result)
    return result


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--split", choices=("train2017", "val2017"), default="train2017")
    parser.add_argument("--limit", type=int)


def _add_worker(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--report", type=Path, required=True)


def _add_orchestrator(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--devices", default=DEFAULT_DEVICES)
    parser.add_argument("--target-shard-bytes", type=int, default=DEFAULT_TARGET_SHARD_BYTES)
    parser.add_argument("--report", type=Path, required=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compute = subparsers.add_parser("worker-compute")
    _add_worker(compute)
    compute.set_defaults(handler=worker_compute)
    worker = subparsers.add_parser("worker-build")
    _add_worker(worker)
    worker.add_argument("--cache-dir", type=Path, required=True)
    worker.add_argument("--target-shard-bytes", type=int, default=DEFAULT_TARGET_SHARD_BYTES)
    worker.set_defaults(handler=worker_build)
    benchmark = subparsers.add_parser("benchmark-cache")
    _add_orchestrator(benchmark)
    benchmark.set_defaults(limit=DEFAULT_BENCHMARK_LIMIT)
    benchmark.add_argument("--batch-candidates", type=int, nargs="+", default=DEFAULT_BATCH_CANDIDATES)
    benchmark.set_defaults(handler=benchmark_cache)
    build = subparsers.add_parser("build-cache")
    _add_orchestrator(build)
    build.add_argument("--batch-size", type=int, required=True)
    build.set_defaults(handler=build_cache)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if hasattr(args, "batch_candidates") and any(batch <= 0 for batch in args.batch_candidates):
        raise ValueError("--batch-candidates must be positive")
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()

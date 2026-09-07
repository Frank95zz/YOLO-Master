#!/usr/bin/env python3
"""Bounded CPU-only comparison of existing D1 shards and lossless per-sample NPY files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NAMES = ("block4", "block8", "block12")
SHAPE = (384, 40, 40)
MIB = 1024**2
GIB = 1024**3


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sample_path(root, sample_id):
    if not re.fullmatch(r"train2017/[0-9]{12}", sample_id):
        raise ValueError(f"Invalid train sample ID: {sample_id!r}")
    return Path(root) / (sample_id + ".npy")


def select_ids(records, count, seed):
    available = sorted(sid for sid in records if sid.startswith("train2017/"))
    if type(count) is not int or not 0 < count <= len(available):
        raise ValueError("Sample count must be positive and cannot exceed available train samples")
    selected = sorted(random.Random(seed).sample(available, count))
    for sid in selected:
        sample_path(Path("."), sid)
    return selected


def validate_array(value, record, shape=SHAPE):
    if value.dtype != np.dtype("float16") or value.shape != (3, *shape):
        raise ValueError(f"Wrong NPY dtype/shape: {value.dtype} {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite feature data")
    for index, name in enumerate(NAMES):
        if digest(value[index]) != record["tensors"][name]["sha256"]:
            raise ValueError(f"Tensor checksum mismatch: {record['sample_id']} {name}")


def pack_features(features):
    if set(features) != set(NAMES):
        raise ValueError("Feature names must be block4/block8/block12")
    return np.stack([np.asarray(features[name]) for name in NAMES])


def read_npy(path, shape=SHAPE):
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float16") or value.shape != (3, *shape):
        raise ValueError("Invalid per-sample NPY dtype/shape")
    return {name: value[index] for index, name in enumerate(NAMES)}


def export_sample(root, record, features, shape=SHAPE):
    """Commit only fully read-back and bitwise-verified arrays; never replace an existing sample."""
    path = sample_path(root, record["sample_id"])
    if path.exists():
        validate_array(np.load(path, allow_pickle=False), record, shape)
        return False
    value = pack_features(features)
    validate_array(value, record, shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    validate_array(np.load(temporary, allow_pickle=False), record, shape)
    os.replace(temporary, path)
    return True


def memory_snapshot():
    root = Path("/sys/fs/cgroup/memory")
    if not root.is_dir():
        raise RuntimeError("This bounded server probe requires cgroup-v1 memory counters")
    stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
    usage = int((root / "memory.usage_in_bytes").read_text())
    limit = int((root / "memory.limit_in_bytes").read_text())
    rss = int(stats.get("total_rss", stats.get("rss", 0)))
    shmem = int(stats.get("total_shmem", stats.get("shmem", 0)))
    return {"usage_bytes": usage, "limit_bytes": limit, "rss_bytes": rss, "shmem_bytes": shmem}


def guard_memory(snapshot, reserve_bytes):
    # File cache is reclaimable; shared memory is not equivalent to clean file cache.
    remaining = snapshot["limit_bytes"] - snapshot["rss_bytes"] - snapshot["shmem_bytes"]
    if remaining < reserve_bytes:
        raise RuntimeError("Memory guard: insufficient headroom after anonymous/shared memory")


def proc_counters():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":")
        io[key] = int(value)
    return {
        "read_bytes": io["read_bytes"],
        "write_bytes": io["write_bytes"],
        "major_faults": usage.ru_majflt,
        "minor_faults": usage.ru_minflt,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss * 1024,
    }


def benchmark_pass(kind, ids, reader, npy_root, batch_size, guard, progress):
    """Materialize identical [B,C,H,W] batches; mmap page faults are included in total time."""
    before = proc_counters()
    started = time.perf_counter()
    fetch_seconds = stack_seconds = 0.0
    batch_seconds = []
    signature = hashlib.sha256()
    reader.close()
    for offset in range(0, len(ids), batch_size):
        guard()
        batch_started = time.perf_counter()
        values = []
        for sid in ids[offset : offset + batch_size]:
            if kind == "shard":
                sample = {name: tensor.numpy() for name, tensor in reader.get(sid).items()}
            else:
                sample = read_npy(sample_path(npy_root, sid))
            values.append(sample)
        fetched = time.perf_counter()
        batch = {name: np.stack([sample[name] for sample in values]) for name in NAMES}
        stacked = time.perf_counter()
        for name in NAMES:
            signature.update(batch[name][:, 0, 0, 0].tobytes())
        del values, batch, sample
        fetch_seconds += fetched - batch_started
        stack_seconds += stacked - fetched
        batch_seconds.append(time.perf_counter() - batch_started)
        progress(min(offset + batch_size, len(ids)))
    reader.close()
    elapsed = time.perf_counter() - started
    after = proc_counters()
    logical_bytes = len(ids) * 3 * int(np.prod(SHAPE)) * 2
    return {
        "backend": kind,
        "sample_count": len(ids),
        "batch_size": batch_size,
        "order_sha256": json_digest(ids),
        "elapsed_seconds": elapsed,
        "images_per_second": len(ids) / elapsed,
        "logical_mib_per_second": logical_bytes / MIB / elapsed,
        "fetch_seconds": fetch_seconds,
        "stack_seconds": stack_seconds,
        "batch_p50_seconds": float(np.percentile(batch_seconds, 50)),
        "batch_p95_seconds": float(np.percentile(batch_seconds, 95)),
        "sentinel_sha256": signature.hexdigest(),
        "process_counters_delta": {key: after[key] - before[key] for key in before if key != "peak_rss_bytes"},
        "process_peak_rss_bytes": after["peak_rss_bytes"],
    }


def summarize(passes):
    if [p["backend"] for p in passes] != ["shard", "npy", "npy", "shard"]:
        raise ValueError("Expected complete ABBA comparison")
    for key in ("sample_count", "batch_size", "order_sha256", "sentinel_sha256"):
        if len({p[key] for p in passes}) != 1:
            raise ValueError(f"Comparison inputs/output mismatch: {key}")
    medians = {
        kind: statistics.median(p["elapsed_seconds"] for p in passes if p["backend"] == kind)
        for kind in ("shard", "npy")
    }
    return {
        "median_pass_seconds": medians,
        "npy_speedup_vs_shard": medians["shard"] / medians["npy"],
        "full_training_eta_available": False,
        "decision": "Reader microbenchmark only; require sustained full-working-set and training validation",
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--count", type=int, default=2048)
    result.add_argument("--batch", type=int, default=64)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-open-shards", type=int, default=4)
    result.add_argument("--conversion-mib-s", type=float, default=64.0)
    result.add_argument("--max-seconds", type=float, default=1200.0)
    result.add_argument("--reserve-gib", type=float, default=24.0)
    return result


def main():
    args = parser().parse_args()
    if not 1 <= args.count <= 5000 or not 1 <= args.batch <= 64:
        raise ValueError("Probe is bounded to 1..5000 samples and batch 1..64")
    if args.max_open_shards < 0 or min(args.conversion_mib_s, args.max_seconds, args.reserve_gib) <= 0:
        raise ValueError("Invalid resource limits")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' to isolate this probe from training GPUs")
    root = args.output.resolve()
    cache = args.cache.resolve()
    if root == cache or root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError("Output must be separate from the source cache")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, root, cache)


def run(args, root, cache):
    started = time.monotonic()
    state = {"status": "RUNNING", "phase": "INITIALIZING", "pid": os.getpid()}

    def status(**updates):
        state.update(updates)
        state["elapsed_seconds"] = time.monotonic() - started
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(root / "status.json", state)
        print(json.dumps(state, sort_keys=True), flush=True)

    def guard():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Probe time limit exceeded")
        guard_memory(memory_snapshot(), args.reserve_gib * GIB)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 3 * GIB:
            raise RuntimeError("Probe process exceeded 3 GiB RSS guard")

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    status()
    reader = None
    try:
        guard()
        import torch

        from ultralytics.nn.foundation.cache import FeatureCacheReader

        torch.set_num_threads(1)
        reader = FeatureCacheReader(cache, max_open_shards=args.max_open_shards)
        if tuple(reader.contract["expected_shape"]) != SHAPE or tuple(reader.contract["feature_names"]) != NAMES:
            raise ValueError("Expected formal D1 block4/8/12 features at [384,40,40]")
        if reader.contract["dtype"] != "float16":
            raise ValueError("Expected FP16 cache")
        ids = select_ids(reader.records, args.count, args.seed)
        order = list(ids)
        random.Random(args.seed + 1).shuffle(order)
        npy_root = root / "npy"
        needed = len(ids) * (3 * int(np.prod(SHAPE)) * 2 + 4096)
        if shutil.disk_usage(root).free < needed + 32 * GIB:
            raise RuntimeError("Insufficient disk space including a 32 GiB reserve")
        repo = Path(__file__).resolve().parents[2]
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        request = {
            "source_cache": str(cache),
            "source_content_sha256": reader.index["content_sha256"],
            "source_contract_sha256": reader.index["contract_sha256"],
            "source_samples_manifest_sha256": reader.index["samples_manifest_sha256"],
            "code_commit": head,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_status": subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True),
            "sample_count": len(ids),
            "sample_ids": ids,
            "read_order": order,
            "order_sha256": json_digest(order),
            "selected_shard_count": len({reader.records[sid]["shard"] for sid in ids}),
            "batch_size": args.batch,
            "max_open_shards": args.max_open_shards,
            "feature_names": NAMES,
            "dtype": "float16",
            "shape": (3, *SHAPE),
            "seed": args.seed,
            "workers": 0,
            "gpu_used": False,
            "pin_memory": False,
            "pass_order": ["shard", "npy", "npy", "shard"],
            "conversion_mib_s": args.conversion_mib_s,
            "max_seconds": args.max_seconds,
            "reserve_gib": args.reserve_gib,
            "memory_before": memory_snapshot(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "notes": [
                "No cache eviction, no GPU/DDP/model/loss/optimizer, no training loader modification",
                "OS cache is uncontrolled and conversion warms data; this is not a cold-disk benchmark",
                "One CPU reader and common NumPy batch materialization; excludes DataLoader IPC/pinning/H2D",
                "mmap faults can be charged to stack time; compare total elapsed, not fetch time alone",
                "Low scheduling priority and concurrent official training may affect both measurements",
            ],
        }
        old_request = root / "request.json"
        if old_request.exists():
            old = json.loads(old_request.read_text())
            for key in (
                "source_content_sha256",
                "source_contract_sha256",
                "order_sha256",
                "script_sha256",
                "batch_size",
            ):
                if old[key] != request[key]:
                    raise ValueError(f"Cannot resume a different benchmark request: {key}")
        write_json(old_request, request)
        conversion_started = time.monotonic()
        manifests = []
        by_shard = sorted(ids, key=lambda sid: (reader.records[sid]["shard"], sid))
        for index, sid in enumerate(by_shard, 1):
            guard()
            record = reader.records[sid]
            features = {name: tensor.numpy() for name, tensor in reader.get(sid).items()}
            export_sample(npy_root, record, features)
            del features
            manifests.append(
                {
                    "sample_id": sid,
                    "path": sample_path(Path("npy"), sid).as_posix(),
                    "source_cache_key": record["cache_key"],
                    "tensor_sha256": {name: record["tensors"][name]["sha256"] for name in NAMES},
                }
            )
            ideal_seconds = index * 3 * int(np.prod(SHAPE)) * 2 / (args.conversion_mib_s * MIB)
            time.sleep(max(0.0, ideal_seconds - (time.monotonic() - conversion_started)))
            if index == 1 or index % 64 == 0 or index == len(ids):
                status(phase="CONVERTING_AND_VERIFYING", completed=index, total=len(ids))
        reader.close()
        conversion_seconds = time.monotonic() - conversion_started
        write_json(
            root / "npy-manifest.json",
            {
                "schema": "d1-npy-layout-probe-v1",
                "verified_tensor_count": len(ids) * 3,
                "feature_names": NAMES,
                "dtype": "float16",
                "shape": (3, *SHAPE),
                "records": sorted(manifests, key=lambda item: item["sample_id"]),
                "source_contract_sha256": request["source_contract_sha256"],
            },
        )
        passes = []
        for index, kind in enumerate(request["pass_order"], 1):
            status(phase="BENCHMARKING", backend=kind, pass_number=index, completed=0, total=len(ids))
            result = benchmark_pass(
                kind,
                order,
                reader,
                npy_root,
                args.batch,
                guard,
                lambda completed: status(completed=completed),
            )
            passes.append(result)
            write_json(root / f"pass-{index:02d}-{kind}.json", result)
        summary = summarize(passes)
        summary.update(
            {
                "status": "COMPLETED",
                "sample_count": len(ids),
                "verified_tensor_count": len(ids) * 3,
                "conversion_seconds": conversion_seconds,
                "passes": passes,
                "script_sha256": request["script_sha256"],
                "source_content_sha256": request["source_content_sha256"],
                "memory_after": memory_snapshot(),
                "notes": request["notes"],
            }
        )
        write_json(root / "summary.json", summary)
        status(status="COMPLETED", phase="FINISHED", npy_speedup_vs_shard=summary["npy_speedup_vs_shard"])
    except BaseException as exc:
        status(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    main()

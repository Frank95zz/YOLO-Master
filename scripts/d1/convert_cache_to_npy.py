#!/usr/bin/env python3
"""Migrate local D1 shards to lossless NPY files, retiring only verified shards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import signal
import stat
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from safetensors import safe_open

NAMES = ("block4", "block8", "block12")
SHAPE = (384, 40, 40)
GIB = 1024**3
SCHEMA = "d1-npy-cache-v1"
WORKER_STOP = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    root = Path(root).absolute()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path escapes the selected directory")
    target = root / relative
    for item in (target, *target.parents):
        if item.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {item}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Resolved path escapes the selected directory")
    return target


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    path = safe_path(Path(path).parent, Path(path).name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(path.parent, path.name + ".part")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def discard_pages(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def npy_path(root, sample_id):
    if not re.fullmatch(r"(train2017|val2017)/[0-9]{12}", sample_id):
        raise ValueError(f"Invalid sample ID: {sample_id}")
    return safe_path(root, sample_id + ".npy")


def validate_array(value, record, shape=SHAPE):
    if value.shape != (3, *shape) or value.dtype != np.dtype("float16"):
        raise ValueError("NPY shape/dtype mismatch")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite features")
    if set(record["tensors"]) != set(NAMES):
        raise ValueError("Feature names mismatch")
    for i, name in enumerate(NAMES):
        info = record["tensors"][name]
        if info["shape"] != list(shape) or info["dtype"] != "float16" or info["nbytes"] != value[i].nbytes:
            raise ValueError("Source tensor metadata mismatch")
        if sha(value[i].tobytes(order="C")) != info["sha256"]:
            raise ValueError(f"Tensor checksum mismatch: {record['sample_id']} {name}")


def check_npy(path, record, shape=SHAPE):
    path = safe_path(Path(path).parent, Path(path).name)
    value = np.load(path, allow_pickle=False)
    validate_array(value, record, shape)
    return {
        "sample_id": record["sample_id"],
        "path": record["sample_id"] + ".npy",
        "bytes": path.stat().st_size,
        "sha256": file_sha(path),
    }


def export_sample(output, record, handle, shape=SHAPE):
    path = npy_path(output, record["sample_id"])
    if path.exists():
        return check_npy(path, record, shape)
    value = np.stack([handle.get_tensor(record["tensors"][name]["key"]) for name in NAMES])
    validate_array(value, record, shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_path(path.parent, path.name + ".part")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    checked = check_npy(temporary, record, shape)
    os.replace(temporary, path)
    sync_dir(path.parent)
    return checked


def convert_shard(
    source,
    output,
    shard,
    records,
    source_index_sha,
    *,
    shape=SHAPE,
    progress=lambda _: None,
    before_retire=lambda: None,
):
    """Durably verify every replacement before committing the source-file deletion."""
    name = shard["filename"]
    if not re.fullmatch(r"(train2017|val2017)-r[0-9]{2}-[0-9]{5}\.safetensors", name):
        raise ValueError("Unexpected shard filename")
    records = sorted(records, key=lambda item: item["sample_id"])
    if len(records) != shard["sample_count"] or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Shard record count/uniqueness mismatch")
    if any(r["shard"] != name or r["split"] != shard["split"] for r in records):
        raise ValueError("Record belongs to another shard/split")
    path = safe_path(source, name)
    receipt_path = safe_path(output, "receipts/" + name + ".json")
    identity = {"source_index_sha256": source_index_sha, "source_shard": shard}
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if old and any(old.get(k) != v for k, v in identity.items()):
        raise ValueError("Receipt belongs to another source")
    if old and old.get("state") not in ("VERIFIED", "SOURCE_REMOVED"):
        raise ValueError("Invalid receipt state")
    if not path.exists() and old is None:
        raise FileNotFoundError(f"Source missing without a verified receipt: {path}")
    before = None
    if path.exists():
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != shard["bytes"] or file_sha(path) != shard["sha256"]:
            raise ValueError("Source shard size/checksum mismatch")
        with safe_open(path, framework="numpy") as handle:
            expected = {r["tensors"][n]["key"] for r in records for n in NAMES}
            if set(handle.keys()) != expected:
                raise ValueError("Shard contains missing or extra tensor keys")
            for i, record in enumerate(records, 1):
                export_sample(output, record, handle, shape)
                if i == 1 or i % 64 == 0 or i == len(records):
                    progress(i)
    # A second independent pass protects the destructive commit, including interrupted resumes.
    checked = []
    for record in records:
        destination = npy_path(output, record["sample_id"])
        checked.append(check_npy(destination, record, shape))
        discard_pages(destination)
    if old and old.get("files") != checked:
        raise ValueError("Verified replacement no longer matches the durable receipt")
    receipt = {
        **identity,
        "schema_version": SCHEMA,
        "state": "VERIFIED",
        "files": checked,
        "sample_count": len(records),
        "verified_tensor_count": 3 * len(records),
    }
    atomic_json(receipt_path, receipt)
    before_retire()
    if before is not None:
        after = path.stat()
        if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_dev,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError("Source changed during conversion; refusing deletion")
        safe_path(source, name)
        discard_pages(path)
        path.unlink()
        sync_dir(path.parent)
    receipt["state"] = "SOURCE_REMOVED"
    atomic_json(receipt_path, receipt)
    return receipt


def _init_worker(stop):
    global WORKER_STOP
    WORKER_STOP = stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _check_stop():
    if WORKER_STOP is not None and WORKER_STOP.is_set():
        raise InterruptedError("Coordinator requested stop; source retirement cancelled")


def _worker_job(job):
    source, output, shard, records, identity, shape = job
    started = time.monotonic()
    status_path = safe_path(output, f"workers/pid-{os.getpid()}.json")
    existed = safe_path(source, shard["filename"]).exists()

    def progress(completed=0, status="RUNNING"):
        _check_stop()
        atomic_json(
            status_path,
            {
                "pid": os.getpid(),
                "status": status,
                "shard": shard["filename"],
                "mode": "CONVERT" if existed else "REVALIDATE_RETIRED",
                "samples_written": completed,
                "sample_count": len(records),
                "elapsed_seconds": time.monotonic() - started,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    try:
        progress()
        receipt = convert_shard(
            source, output, shard, records, identity, shape=shape, progress=progress, before_retire=_check_stop
        )
        progress(len(records), "COMPLETED")
        return {
            "receipt": receipt,
            "pid": os.getpid(),
            "source_existed": existed,
            "elapsed_seconds": time.monotonic() - started,
        }
    except BaseException as exc:
        atomic_json(
            status_path,
            {
                "pid": os.getpid(),
                "status": "FAILED",
                "shard": shard["filename"],
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise


def _resource_gate(output, reserved_bytes):
    if shutil.disk_usage(output).free < reserved_bytes:
        raise RuntimeError("Insufficient space for all concurrent shards plus reserve")
    root = Path("/sys/fs/cgroup/memory")
    if (root / "memory.limit_in_bytes").is_file():
        stats = {k: int(v) for k, v in (line.split() for line in (root / "memory.stat").read_text().splitlines())}
        limit = int((root / "memory.limit_in_bytes").read_text())
        kernel_path = root / "memory.kmem.usage_in_bytes"
        kernel = int(kernel_path.read_text()) if kernel_path.exists() else 0
        if limit - stats.get("rss", 0) - stats.get("shmem", 0) - kernel < 16 * GIB:
            raise RuntimeError("Insufficient anonymous-memory headroom; parallel migration stopped")


def parallel_shards(jobs, output, workers, reserve_bytes, on_result, heartbeat=lambda _: None):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer in 1..8")
    names = [job[2]["filename"] for job in jobs]
    ids = [record["sample_id"] for job in jobs for record in job[3]]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Parallel jobs contain duplicate shard names or sample IDs")
    if not jobs:
        return
    maximum_staging = max(job[2]["bytes"] for job in jobs) * workers + reserve_bytes
    _resource_gate(output, maximum_staging)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(stop,))
    pending = {}
    iterator = iter(jobs)

    def submit():
        job = next(iterator, None)
        if job is None:
            return
        _resource_gate(output, maximum_staging)
        pending[executor.submit(_worker_job, job)] = job[2]["filename"]

    try:
        for _ in range(workers):
            submit()
        while pending:
            done, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
            _resource_gate(output, maximum_staging)
            # Observe all failures in a completed group before dispatching additional destructive work.
            results = [(future, future.result()) for future in done]
            for future, result in results:
                del pending[future]
                on_result(result)
            for _ in results:
                submit()
            heartbeat(sorted(pending.values()))
    except BaseException:
        stop.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def local_root(raw):
    path = safe_path(Path(raw).absolute().parent, Path(raw).name)
    if not path.resolve().is_relative_to(Path("/root")):
        raise ValueError("This migration is restricted to /root")
    parent = path
    while not parent.exists():
        parent = parent.parent
    if parent.stat().st_dev != Path("/root").stat().st_dev:
        raise ValueError("Source/output must be on the /root filesystem, not another mount")
    return path


def load_sources(base):
    from ultralytics.nn.foundation.cache import FeatureCacheReader

    result = []
    numeric_ids = set()
    for split, count in (("train2017", 118287), ("val2017", 5000)):
        source = local_root(base / f"coco2017-{split}-d1-cache-v1")
        reader = FeatureCacheReader(source)
        if len(reader.records) != count or reader.index["split_counts"] != {split: count}:
            raise ValueError("Formal COCO split count mismatch")
        if reader.contract["expected_shape"] != list(SHAPE) or reader.contract["feature_names"] != list(NAMES):
            raise ValueError("Wrong formal D1 feature contract")
        if reader.contract["dtype"] != "float16":
            raise ValueError("Wrong formal D1 dtype")
        groups = {}
        for record in reader.records.values():
            npy_path(Path("/root"), record["sample_id"])
            if record["split"] != split or not record["sample_id"].startswith(split + "/"):
                raise ValueError("Wrong sample split")
            numeric = record["sample_id"].split("/")[1]
            if numeric in numeric_ids:
                raise ValueError("Train/val duplicate image ID")
            numeric_ids.add(numeric)
            groups.setdefault(record["shard"], []).append(record)
        shards = reader.index["shards"]
        if len({s["filename"] for s in shards}) != len(shards) or set(groups) != {s["filename"] for s in shards}:
            raise ValueError("Shard index coverage mismatch")
        for shard in shards:
            if len(groups[shard["filename"]]) != shard["sample_count"]:
                raise ValueError("Shard sample count mismatch")
            safe_path(source, shard["filename"])
        actual = {p.name for p in source.glob("*.safetensors")}
        if actual - set(groups) or list(source.glob("*.part")):
            raise ValueError("Unindexed source shards or unfinished source files")
        result.append((split, source, reader, groups))
    if result[0][2].index["contract_sha256"] != result[1][2].index["contract_sha256"]:
        raise ValueError("Train/val contracts differ")
    return result


def immutable_copy(source, target):
    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("Existing provenance differs from source")
    else:
        atomic_bytes(target, data)


def execute(args, base, output):
    started = time.monotonic()
    state = {
        "status": "RUNNING",
        "phase": "PREFLIGHT",
        "pid": os.getpid(),
        "total_samples": 123287,
        "verified_samples": 0,
        "shards_completed": 0,
        "source_bytes_removed": 0,
        "workers": args.workers,
        "session_samples_checked": 0,
        "session_tasks_completed": 0,
        "newly_converted_samples": 0,
        "revalidated_prior_samples": 0,
    }

    def status(**updates):
        state.update(updates)
        elapsed = time.monotonic() - started
        state.update(elapsed_seconds=elapsed, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        if state["session_samples_checked"]:
            state["images_per_second"] = state["session_samples_checked"] / elapsed
            state["rough_remaining_seconds"] = (123287 - state["session_samples_checked"]) / state["images_per_second"]
            state["eta_note"] = (
                "Includes revalidation of previously retired shards; pending shards are converted first."
            )
        atomic_json(output / "status.json", state)
        print(json.dumps(state, sort_keys=True), flush=True)

    status()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Conversion must run CPU-only with CUDA_VISIBLE_DEVICES=''")
        sources = load_sources(base)
        source_identity = {split: file_sha(source / "index.json") for split, source, _, _ in sources}
        identity = {
            "schema_version": SCHEMA,
            "source_base": str(base),
            "output": str(output),
            "source_indices_sha256": source_identity,
            "delete_verified_source": True,
        }
        for split, source, reader, _ in sources:
            for shard in reader.index["shards"]:
                path = safe_path(source, shard["filename"])
                receipt_path = safe_path(output, "receipts/" + shard["filename"] + ".json")
                if not path.exists() and not receipt_path.exists():
                    raise FileNotFoundError(f"Preflight: missing source and receipt: {path}")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if (
                        receipt.get("source_index_sha256") != source_identity[split]
                        or receipt.get("source_shard") != shard
                    ):
                        raise ValueError("Preflight receipt identity mismatch")
        for split, source, reader, _ in sources:
            for name in ("index.json", reader.index["samples_manifest"]):
                immutable_copy(safe_path(source, name), safe_path(output, f"provenance/{split}/{name}"))
        intent_path = base / "NPY_MIGRATION.json"
        if intent_path.exists() and json.loads(intent_path.read_text()).get("identity") != identity:
            raise ValueError("This source is already bound to another migration")
        atomic_json(
            intent_path,
            {
                "identity": identity,
                "warning": "Source shards are retired incrementally; do not train from this directory.",
            },
        )
        script_path = Path(__file__).resolve()
        atomic_json(
            output / "run-provenance.json",
            {
                **identity,
                "script_sha256": file_sha(script_path),
                "workers": args.workers,
                "process_start_method": "spawn" if args.workers > 1 else "serial",
                "shape": [3, *SHAPE],
                "feature_names": NAMES,
                "dtype": "float16",
                "source_preservation": "Metadata retained; each local shard deleted only after durable lossless verification",
                "nfs_used": False,
                "no_training_started": True,
            },
        )
        state["total_shards"] = sum(len(reader.index["shards"]) for _, _, reader, _ in sources)
        jobs = []
        previously_retired = set()
        for split, source, reader, groups in sources:
            for shard in sorted(reader.index["shards"], key=lambda item: item["filename"]):
                if not safe_path(source, shard["filename"]).exists():
                    previously_retired.add(shard["filename"])
                    state["verified_samples"] += shard["sample_count"]
                    state["shards_completed"] += 1
                    state["source_bytes_removed"] += shard["bytes"]
                jobs.append((source, output, shard, groups[shard["filename"]], source_identity[split], SHAPE))
        state["previously_converted_samples"] = state["verified_samples"]
        state["previously_retired_shards"] = len(previously_retired)
        jobs.sort(key=lambda job: (job[2]["filename"] in previously_retired, job[2]["filename"]))

        def committed(result):
            receipt = result["receipt"]
            shard = receipt["source_shard"]
            count = receipt["sample_count"]
            state["session_samples_checked"] += count
            state["session_tasks_completed"] += 1
            if shard["filename"] in previously_retired:
                state["revalidated_prior_samples"] += count
            else:
                state["newly_converted_samples"] += count
                state["verified_samples"] += count
                state["shards_completed"] += 1
                state["source_bytes_removed"] += shard["bytes"]
            status(
                phase="SHARD_COMMITTED",
                last_committed_shard=shard["filename"],
                last_worker_pid=result["pid"],
                last_shard_seconds=result["elapsed_seconds"],
            )

        if args.workers > 1:
            parallel_shards(
                jobs,
                output,
                args.workers,
                args.reserve_gib * GIB,
                committed,
                heartbeat=lambda names: status(
                    phase="PARALLEL_CONVERT_OR_REVALIDATE", active_shards=names, active_tasks=len(names)
                ),
            )
        else:
            for job in jobs:
                source, _, shard, records, index_sha, shape = job
                _resource_gate(output, shard["bytes"] + args.reserve_gib * GIB)
                status(phase="CONVERT_VERIFY_RETIRE", current_shard=shard["filename"], current_shard_samples=0)
                step_started = time.monotonic()
                receipt = convert_shard(
                    source,
                    output,
                    shard,
                    records,
                    index_sha,
                    shape=shape,
                    progress=lambda n: status(current_shard_samples=n),
                )
                committed({"receipt": receipt, "pid": os.getpid(), "elapsed_seconds": time.monotonic() - step_started})
        status(phase="PUBLISHING_INDEX")
        for split, source, reader, groups in sources:
            files = {}
            for shard in reader.index["shards"]:
                receipt_path = safe_path(output, "receipts/" + shard["filename"] + ".json")
                receipt = json.loads(receipt_path.read_text())
                if receipt["state"] != "SOURCE_REMOVED":
                    raise ValueError("Unfinished shard during finalization")
                for entry in receipt["files"]:
                    if entry["sample_id"] in files:
                        raise ValueError("Duplicate replacement sample")
                    files[entry["sample_id"]] = entry
            if set(files) != set(reader.records):
                raise ValueError("Final replacement coverage mismatch")
            expected = {entry["path"] for entry in files.values()}
            actual = {p.relative_to(output).as_posix() for p in (output / split).glob("*.npy")}
            if expected != actual or list((output / split).glob("*.part")):
                raise ValueError("Unexpected output files or unfinished NPY files")
            lines = []
            for sid in sorted(files):
                entry = files[sid]
                path = npy_path(output, sid)
                if path.stat().st_size != entry["bytes"]:
                    raise ValueError("Replacement size changed before publication")
                record = dict(reader.records[sid])
                record["source_shard"] = record.pop("shard")
                record.update(npy_path=entry["path"], npy_sha256=entry["sha256"], npy_bytes=entry["bytes"])
                lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            data = "".join(lines).encode()
            atomic_bytes(output / f"{split}-samples.jsonl", data)
            atomic_json(
                output / f"{split}-index.json",
                {
                    "schema_version": SCHEMA,
                    "sample_count": len(files),
                    "split": split,
                    "contract": reader.contract,
                    "contract_sha256": reader.index["contract_sha256"],
                    "source_content_sha256": reader.index["content_sha256"],
                    "source_index_sha256": source_identity[split],
                    "feature_names": NAMES,
                    "dtype": "float16",
                    "shape": [3, *SHAPE],
                    "samples_manifest": f"{split}-samples.jsonl",
                    "samples_manifest_sha256": sha(data),
                    "npy_bytes": sum(f["bytes"] for f in files.values()),
                },
            )
            if list(source.glob("*.safetensors")):
                raise ValueError("Source retirement is incomplete")
        atomic_json(
            output / "summary.json",
            {
                **identity,
                "status": "COMPLETED",
                "sample_count": 123287,
                "tensor_count": 369861,
                "shards_retired": state["shards_completed"],
                "elapsed_seconds": time.monotonic() - started,
                "npy_shape": [3, *SHAPE],
                "validation": "Source shard SHA256 plus per-layer SHA256 on write and independent readback before retirement",
                "training_integration": "NPY storage complete; production D1 loader integration is a separate change",
            },
        )
        status(status="COMPLETED", phase="FINISHED")
    except BaseException as exc:
        status(
            status="STOPPED" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "FAILED",
            phase="STOPPED_RESUMABLE",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retire-verified-source", action="store_true", required=True)
    parser.add_argument("--reserve-gib", type=float, default=24)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    if args.reserve_gib < 8:
        raise ValueError("At least 8 GiB free-space reserve is required")
    base, output = local_root(args.source_base), local_root(args.output)
    if base == output or base.is_relative_to(output) or output.is_relative_to(base):
        raise ValueError("Source and output directories must be disjoint")
    output.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, _frame):
        raise InterruptedError(f"Signal {signum}; resume with the same command")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with safe_path(base, ".npy-migration.lock").open("a") as source_lock, safe_path(output, ".lock").open("a") as lock:
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, base, output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build and losslessly convert VisDrone train/val caches on local storage only."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from scripts.d1.cache_features import FEATURE_NAMES, cache_contract, load_image, make_letterbox, write_json
from scripts.d1.convert_cache_to_npy import check_npy, validate_array
from scripts.d1.prepare_visdrone import COUNTS, encoded, file_sha, immutable
from ultralytics.nn.foundation import DINOv3Teacher
from ultralytics.nn.foundation.cache import FeatureCacheReader, FeatureCacheWriter, sha256_bytes, verify_feature_cache
from ultralytics.nn.foundation.npy_cache import NPY_SCHEMA_VERSION, NpyFeatureCacheReader, validate_npy_evidence

SCHEMA = "d1-e2-cache-run-v1"
SHARD_BYTES = 2 * 1024**3


def batches(records, rank, world_size, batch):
    if world_size < 1 or not 0 <= rank < world_size or batch < 1:
        raise ValueError("Invalid deterministic partition")
    part = records[rank::world_size]
    return [part[i:i + batch] for i in range(0, len(part), batch)]


def rows(data, split):
    manifest = json.loads((data / "manifest.json").read_text())
    raw = (data / "samples.jsonl").read_bytes()
    if sha256_bytes(raw) != manifest["samples_sha256"]:
        raise ValueError("Data manifest checksum changed")
    records = [json.loads(line) for line in raw.splitlines()]
    selected = [r for r in records if r["split"] == f"visdrone-{split}"]
    ids = [r["sample_id"] for r in selected]
    if len(ids) != COUNTS[split] or ids != sorted(set(ids)):
        raise ValueError("Wrong split coverage or ordering")
    return selected


def identity(args):
    return {"schema_version": SCHEMA,
            "code_commit": subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
            "data_manifest_sha256": file_sha(args.data / "manifest.json"),
            "contract": cache_contract(args.repo), "world_size": len(args.devices.split(",")),
            "devices": args.devices, "batch": args.batch, "target_shard_bytes": SHARD_BYTES,
            "batch_policy": "sorted indices modulo world_size; replay original batches including tails on resume",
            "seed": 0, "tf32": False}


def worker(args):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    own = identity(args)
    immutable(args.output / "run-identity.json", encoded(own), verify=True)
    split = f"visdrone-{args.split}"
    records = rows(args.data, args.split)
    selected_batches = batches(records, args.rank, own["world_size"], args.batch)
    part_dir = args.output / "parts" / split / f"rank{args.rank:02d}"
    part_dir.mkdir(parents=True, exist_ok=True)
    with (part_dir / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _worker(args, own, split, selected_batches, part_dir)


def _worker(args, own, split, selected_batches, part_dir):
    started = time.monotonic()
    report_path = args.output / "reports" / f"{split}-r{args.rank:02d}.json"
    if list(part_dir.glob("*.part")):
        raise ValueError("Uncommitted temporary shards remain; inspect and quarantine them before resuming")
    if (part_dir / "index.json").exists():
        verify_feature_cache(part_dir)
    writer = FeatureCacheWriter(part_dir, split=split, contract=own["contract"],
                                target_shard_bytes=SHARD_BYTES, shard_prefix=f"{split}-r{args.rank:02d}")
    weights = args.weights / "model.safetensors"
    if file_sha(weights) != own["contract"]["teacher_weights_sha256"]:
        raise ValueError("Teacher weights differ from WP0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    teacher = DINOv3Teacher(model_id=own["contract"]["model_id"], weights_path=args.weights,
                           local_files_only=True, dtype="fp16", device="cuda:0", output_layers=(4, 8, 12))
    letterbox = make_letterbox()
    contexts, done = [], 0
    seen_hashes = set()
    for number, batch_rows in enumerate(selected_batches):
        for row in batch_rows:
            if file_sha(args.data / row["image_path"]) != row["image_sha256"]:
                raise ValueError("Source image changed")
        contexts.append([r["sample_id"] for r in batch_rows])
        if not all(writer.is_cached(r["sample_id"], r["image_sha256"]) for r in batch_rows):
            images = torch.stack([load_image(args.data / r["image_path"], letterbox) for r in batch_rows])
            dense = teacher.encode(images).dense
            for i, row in enumerate(batch_rows):
                # Official duplicate images keep separate IDs. Avoid sharing a tensor key
                # inside a shard when FP16 outputs came from different batch contexts.
                if row["image_sha256"] in seen_hashes:
                    writer.flush()
                writer.add(sample_id=row["sample_id"], split=split, image_path=row["image_path"],
                           image_sha256=row["image_sha256"], features={n: dense[n][i] for n in FEATURE_NAMES})
                seen_hashes.add(row["image_sha256"])
        done += len(batch_rows)
        if number % 8 == 0:
            write_json(report_path, {"status": "RUNNING", "rank": args.rank, "split": split,
                                   "processed": done, "total": sum(map(len, selected_batches)),
                                   "elapsed_seconds": time.monotonic() - started})
            print(f"{split} rank={args.rank} processed={done}", flush=True)
    writer.close()
    reader = FeatureCacheReader(part_dir)
    expected = {r["sample_id"] for batch in selected_batches for r in batch}
    if set(reader.records) != expected:
        raise ValueError("Rank cache contains missing or extra samples")
    verification = verify_feature_cache(part_dir)
    parity = []
    # Recreate the first and tail batch, not an arbitrary regrouping of FP16 inputs.
    for batch_index in sorted({0, len(selected_batches) - 1}):
        batch_rows = selected_batches[batch_index]
        dense = teacher.encode(torch.stack([load_image(args.data / r["image_path"], letterbox)
                                           for r in batch_rows])).dense
        for i, row in enumerate(batch_rows):
            cached = reader.get(row["sample_id"])
            for name in FEATURE_NAMES:
                online, saved = dense[name][i].cpu(), cached[name]
                if not torch.isfinite(online).all() or not torch.allclose(online, saved, rtol=1e-3, atol=1e-3):
                    raise ValueError(f"Original-batch parity failed: {row['sample_id']}/{name}")
                error = (online.float() - saved.float()).abs()
                parity.append({"sample_id": row["sample_id"], "layer": name, "batch_index": batch_index,
                               "max_abs": error.max().item(), "mean_abs": error.mean().item()})
    if teacher.model.training or any(p.requires_grad for p in teacher.model.parameters()):
        raise ValueError("Teacher lost frozen/eval constraints")
    report = {"status": "PASSED", "identity": own, "rank": args.rank, "split": split,
              "sample_count": len(expected), "batch_contexts": contexts, "verification": verification,
              "parity": parity, "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    write_json(report_path, report)
    return report


def finalize(args, split, own):
    name = f"visdrone-{split}"
    target = args.output / "safetensors" / name
    target.mkdir(parents=True, exist_ok=True)
    expected_rows = rows(args.data, split)
    expected = {r["sample_id"]: r for r in expected_rows}
    seen, sources = set(), {}
    for rank in range(own["world_size"]):
        report = json.loads((args.output / "reports" / f"{name}-r{rank:02d}.json").read_text())
        expected_contexts = [[r["sample_id"] for r in b]
                             for b in batches(expected_rows, rank, own["world_size"], args.batch)]
        if (report.get("status") != "PASSED" or report.get("identity") != own
                or report.get("rank") != rank or report.get("split") != name
                or report.get("batch_contexts") != expected_contexts):
            raise ValueError("Rank receipt or batch context mismatch")
        part = args.output / "parts" / name / f"rank{rank:02d}"
        if list(part.glob("*.part")) or verify_feature_cache(part) != report["verification"]:
            raise ValueError("Rank cache changed or unfinished")
        reader = FeatureCacheReader(part)
        rank_ids = {sid for batch in expected_contexts for sid in batch}
        if reader.contract != own["contract"] or set(reader.records) != rank_ids or seen & rank_ids:
            raise ValueError("Rank contract or membership mismatch")
        seen.update(rank_ids)
        for sid, record in reader.records.items():
            if any(record[k] != expected[sid][k] for k in ("image_path", "image_sha256", "split")):
                raise ValueError("Image-feature identity mismatch")
        for shard in reader.index["shards"]:
            filename = shard["filename"]
            if filename in sources:
                raise ValueError("Shard collision")
            sources[filename] = part / filename
    if seen != set(expected) or list(target.glob("*.part")):
        raise ValueError("Final split coverage or temporary-file mismatch")
    if {p.name for p in target.glob("*.safetensors")} - set(sources):
        raise ValueError("Unindexed final shard")
    for filename, source in sources.items():
        dest = target / filename
        if dest.exists():
            if file_sha(dest) != file_sha(source):
                raise ValueError("Final shard changed")
        else:
            os.link(source, dest)
    FeatureCacheWriter(target, split=name, contract=own["contract"], target_shard_bytes=SHARD_BYTES).close()
    return verify_feature_cache(target)


def convert_preserving_source(source, output):
    """Non-destructive sibling of the legacy /root-only retirement utility."""
    reader = FeatureCacheReader(source)
    split = source.name
    if split not in ("visdrone-train", "visdrone-val"):
        raise ValueError("This entry only converts E2 train/val")
    source_sha = file_sha(source / "index.json")
    for filename in ("index.json", "samples.jsonl"):
        immutable(output / "provenance" / split / filename, (source / filename).read_bytes())
    records, total_bytes = [], 0
    for shard in reader.index["shards"]:
        shard_path = source / shard["filename"]
        if file_sha(shard_path) != shard["sha256"]:
            raise ValueError("Source shard checksum failed")
        members = sorted((r for r in reader.records.values() if r["shard"] == shard["filename"]),
                         key=lambda r: r["sample_id"])
        checked = []
        with safe_open(shard_path, framework="numpy") as handle:
            for record in members:
                # The generic reader enforces portable split/image paths on readback.
                if record["sample_id"] != f"{split}/{Path(record['image_path']).stem}":
                    raise ValueError("Unsafe NPY identity")
                dest = output / (record["sample_id"] + ".npy")
                if not dest.exists():
                    array = np.stack([handle.get_tensor(record["tensors"][n]["key"]) for n in FEATURE_NAMES])
                    validate_array(array, record)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + ".part")
                    if tmp.is_symlink():
                        raise ValueError("Unsafe temporary NPY")
                    with tmp.open("wb") as stream:
                        np.save(stream, array, allow_pickle=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    check_npy(tmp, record)
                    os.replace(tmp, dest)
                checked.append(check_npy(dest, record))
                item = dict(record)
                item["source_shard"] = item.pop("shard")
                item.update(npy_path=checked[-1]["path"], npy_sha256=checked[-1]["sha256"],
                            npy_bytes=checked[-1]["bytes"])
                total_bytes += item["npy_bytes"]
                records.append(item)
        receipt = {"schema_version": NPY_SCHEMA_VERSION, "state": "VERIFIED", "source_index_sha256": source_sha,
                   "source_shard": shard, "sample_count": len(members), "verified_tensor_count": len(members) * 3,
                   "files": checked}
        immutable(output / "receipts" / f"{shard['filename']}.json", encoded(receipt))
    records.sort(key=lambda r: r["sample_id"])
    data = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in records)
    immutable(output / f"{split}-samples.jsonl", data)
    index = {"schema_version": NPY_SCHEMA_VERSION, "sample_count": len(records), "split": split,
             "contract": reader.contract, "contract_sha256": reader.index["contract_sha256"],
             "source_content_sha256": reader.index["content_sha256"], "source_index_sha256": source_sha,
             "feature_names": list(FEATURE_NAMES), "dtype": "float16", "shape": [3, 384, 40, 40],
             "samples_manifest": f"{split}-samples.jsonl", "samples_manifest_sha256": sha256_bytes(data),
             "npy_bytes": total_bytes}
    immutable(output / f"{split}-index.json", encoded(index))
    converted = NpyFeatureCacheReader(output / split)
    for sid in converted.records:
        converted.verify_sample(sid)
    return index


def execute(args):
    own = identity(args)
    if subprocess.check_output(["git", "-C", str(args.repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit source before running E2 cache")
    if shutil.disk_usage(args.output).free < 200 * 1024**3:
        raise ValueError("E2 requires 200 GiB free incremental capacity")
    immutable(args.output / "run-identity.json", encoded(own))
    started = time.monotonic()
    results = {}
    for split in ("train", "val"):
        write_json(args.output / "status.json", {"status": "RUNNING", "phase": f"extract-{split}", "pid": os.getpid()})
        children = []
        try:
            for rank, device in enumerate(args.devices.split(",")):
                cmd = [sys.executable, "-u", "-m", "scripts.d1.cache_visdrone", "worker", "--repo", str(args.repo),
                       "--data", str(args.data), "--weights", str(args.weights), "--output", str(args.output),
                       "--devices", args.devices, "--batch", str(args.batch), "--split", split, "--rank", str(rank)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8")
                log = (args.output / "logs" / f"{split}-r{rank:02d}.log").open("a")
                try:
                    child = subprocess.Popen(cmd, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                finally:
                    log.close()
                children.append(child)
            write_json(args.output / f"{split}-pids.json", {"pids": [p.pid for p in children]})
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None, 0) for p in children):
                    raise RuntimeError("Cache worker failed; see rank logs")
                time.sleep(2)
            if any(p.returncode != 0 for p in children):
                raise RuntimeError("Cache worker failed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        write_json(args.output / "status.json", {"status": "RUNNING", "phase": f"verify-convert-{split}", "pid": os.getpid()})
        results[split] = {"safetensors": finalize(args, split, own),
                          "npy": convert_preserving_source(args.output / "safetensors" / f"visdrone-{split}",
                                                           args.output / "npy")}
    npy_root = args.output / "npy"
    summary = {"schema_version": NPY_SCHEMA_VERSION, "status": "COMPLETED", "source_preserved": True,
               "source_indices_sha256": {f"visdrone-{s}": r["npy"]["source_index_sha256"] for s, r in results.items()},
               "sample_count": 7019, "tensor_count": 21057}
    immutable(npy_root / "summary.json", encoded(summary))
    for split in ("train", "val"):
        name = f"visdrone-{split}"
        results[split]["readiness"] = validate_npy_evidence(npy_root / name, npy_root / "summary.json", name, COUNTS[split])
    write_json(args.output / "cache-summary.json", {"status": "PASSED", "identity": own, "splits": results,
               "elapsed_seconds": time.monotonic() - started, "test_dev_extracted": False,
               "training_started": False, "e2_overall_complete": False})
    write_json(args.output / "status.json", {"status": "COMPLETED", "phase": "CACHE_VERIFIED", "pid": os.getpid()})


def main():
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "worker"))
    for name in ("repo", "data", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1 or not args.devices or len(set(args.devices.split(","))) != len(args.devices.split(",")):
        parser.error("Invalid batch or duplicate devices")
    for name in ("repo", "data", "weights", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.stat().st_dev != args.data.stat().st_dev or args.output.stat().st_dev != args.weights.stat().st_dev:
        raise ValueError("Data, weights and cache must share the selected local filesystem")
    fs = subprocess.check_output(["findmnt", "-n", "-o", "FSTYPE", "-T", str(args.output)], text=True).strip()
    if fs in ("nfs", "nfs4", "cifs"):
        raise ValueError("E2 may not extract on network storage")
    if args.mode == "worker":
        worker(args)
        return
    (args.output / "logs").mkdir(exist_ok=True)
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except BaseException as exc:
            write_json(args.output / "status.json", {"status": "FAILED", "error": str(exc), "pid": os.getpid()})
            raise


if __name__ == "__main__":
    main()

"""Non-destructive NVMe staging and content verification for D1 E0/E1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from scripts.d1.cache_features import cache_contract, split_paths
from scripts.d1.run_wp8_p1_control import sha256_file, write_json
from ultralytics.nn.foundation.cache import canonical_json_bytes, sha256_bytes
from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {"train2017": 118287, "val2017": 5000}


@lru_cache(maxsize=8)
def _nvme_device(root: Path) -> int:
    """Probe mount type once; each data path still gets a device/containment check."""
    mount = json.loads(
        subprocess.check_output(
            ["findmnt", "-J", "-T", str(root), "-o", "SOURCE,FSTYPE,TARGET"],
            text=True,
        )
    )["filesystems"][0]
    if not str(mount["source"]).startswith("/dev/nvme") or mount["fstype"] in {"nfs", "nfs4"}:
        raise ValueError(f"Not an NVMe filesystem: {mount}")
    return root.stat().st_dev


def nvme_path(path: Path, nvme_root: Path) -> Path:
    """Resolve containment and verify the actual backing device, not a directory name."""
    path, root = path.resolve(), nvme_root.resolve(strict=True)
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes approved NVMe root: {path}")
    existing = path
    while not existing.exists():
        existing = existing.parent
    if existing.stat().st_dev != _nvme_device(root):
        raise ValueError(f"Data path uses another filesystem: {path}")
    return path


def copy_verified(source: Path | None, target: Path, nvme_root: Path) -> dict:
    """Commit a byte-identical copy; missing source means a verified empty-label case."""
    nvme_path(target, nvme_root)
    if target.is_symlink() or (source is not None and source.is_symlink()):
        raise ValueError("Symlink copies are forbidden")
    size = source.stat().st_size if source is not None else 0
    digest = sha256_file(source) if source is not None else sha256_bytes(b"")
    if target.exists():
        if target.stat().st_size != size or sha256_file(target) != digest:
            raise ValueError(f"Existing destination differs; refusing overwrite: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".d1-copy.part")
        if temporary.is_symlink():
            raise ValueError("Copy temporary must not be a symlink")
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > size:
            raise ValueError(f"Oversized partial copy: {temporary}")
        if source is None:
            with temporary.open("ab"):
                pass
        else:
            with source.open("rb") as src, temporary.open("ab") as dst:
                src.seek(offset)
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
        if temporary.stat().st_size != size or sha256_file(temporary) != digest:
            raise ValueError(f"Partial copy checksum mismatch (preserved): {temporary}")
        os.replace(temporary, target)
    return {"bytes": size, "sha256": digest, "empty_label_materialized": source is None}


def prepare_data(source: Path, target: Path, cache_root: Path, nvme_root: Path, output: Path, workers: int = 8) -> dict:
    """Validate all COCO IDs, copy RGB inputs, and verify every immutable NPY file."""
    if output.resolve() == ROOT or ROOT in output.resolve().parents:
        raise ValueError("Runtime evidence must be outside Git")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in 1..16")
    source = source.resolve(strict=True)
    target = nvme_path(target, nvme_root)
    nvme_path(cache_root, nvme_root)
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Copy roots must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target).free < 30 * 2**30:
        raise ValueError("At least 30 GiB free space is required for RGB staging")
    started = time.time()
    files, summaries, all_ids = {}, {}, set()
    expected_contract = cache_contract(ROOT)
    for split, count in COUNTS.items():
        relative, list_hash = split_paths(ROOT, split, None)
        ids = {Path(p).stem for p in relative}
        if len(relative) != count or len(ids) != count or ids & all_ids:
            raise ValueError("Official COCO splits must be complete, unique and disjoint")
        all_ids.update(ids)
        annotation_rel = f"annotations/instances_{split}.json"
        annotation = json.loads((source / annotation_rel).read_text())
        if {str(row["id"]).zfill(12) for row in annotation["images"]} != ids:
            raise ValueError("COCO annotation image IDs differ from split list")
        positive_ids = {
            str(row["image_id"]).zfill(12)
            for row in annotation["annotations"]
            if not row.get("iscrowd", 0) and row["bbox"][2] > 0 and row["bbox"][3] > 0
        }
        reader = NpyFeatureCacheReader(cache_root / split)
        if reader.contract != expected_contract or set(reader.records) != {f"{split}/{sid}" for sid in ids}:
            raise ValueError("Cache identity does not match the original COCO/ViT-S contract")
        jobs = [(annotation_rel, source / annotation_rel)]
        empty_count = 0
        for image in relative:
            sid = Path(image).stem
            label = f"labels/{split}/{sid}.txt"
            origin = source / label
            if not origin.exists():
                if sid in positive_ids:
                    raise FileNotFoundError(f"Missing labels for positive image {sid}")
                origin = None
                empty_count += 1
            jobs.extend(((image, source / image), (label, origin)))

        def stage(job, split=split, reader=reader):
            name, origin = job
            if origin is not None:
                resolved = origin.resolve(strict=True)
                if source not in resolved.parents:
                    raise ValueError("Source data escaped its root")
            record = copy_verified(origin, target / name, nvme_root)
            if name.startswith("images/"):
                sid = f"{split}/{Path(name).stem}"
                if record["sha256"] != reader.records[sid]["image_sha256"]:
                    raise ValueError(f"RGB and feature-cache image disagree: {sid}")
            return name, record

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (name, record) in enumerate(pool.map(stage, jobs), 1):
                files[name] = record
                if index % 5000 == 0:
                    write_json(
                        output / "status.json",
                        {
                            "status": "running",
                            "phase": "copy_rgb",
                            "split": split,
                            "done": index,
                            "total": len(jobs),
                            "updated_unix": time.time(),
                        },
                    )

        def verify_npy(record):
            path = cache_root / record["npy_path"]
            nvme_path(path, nvme_root)
            if (
                path.is_symlink()
                or path.stat().st_size != record["npy_bytes"]
                or sha256_file(path) != record["npy_sha256"]
            ):
                raise ValueError(f"NPY checksum mismatch: {record['sample_id']}")
            return record["sample_id"]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, _ in enumerate(pool.map(verify_npy, reader.records.values()), 1):
                if index % 1000 == 0:
                    write_json(
                        output / "status.json",
                        {
                            "status": "running",
                            "phase": "verify_npy",
                            "split": split,
                            "done": index,
                            "total": count,
                            "updated_unix": time.time(),
                        },
                    )
        keys = list(reader.records)
        for i in sorted({0, len(keys) // 2, len(keys) - 1}):
            reader.verify_sample(keys[i])
        summaries[split] = {
            "count": count,
            "split_sha256": list_hash,
            "empty_labels": empty_count,
            "cache_index_sha256": sha256_file(reader.index_path),
            "samples_manifest_sha256": reader.index["samples_manifest_sha256"],
            "cache_content_sha256": reader.index["source_content_sha256"],
            "full_npy_sha256_verified": True,
        }
        reader.close()
    write_json(output / "files.json", files)
    report = {
        "schema_version": "d1-e0-data-v1",
        "status": "passed",
        "source": str(source),
        "data_root": str(target),
        "nvme_root": str(nvme_root.resolve()),
        "cache_root": str(cache_root.resolve()),
        "files_sha256": sha256_file(output / "files.json"),
        "splits": summaries,
        "content_sha256": sha256_bytes(canonical_json_bytes(files)),
        "elapsed_seconds": time.time() - started,
        "completed_unix": time.time(),
        "source_preserved": True,
        "storage_class": "nvme",
    }
    write_json(output / "receipt.json", report)
    write_json(output / "status.json", {"status": "completed", "phase": "data_ready", "updated_unix": time.time()})
    return report


def main():
    """Prepare external data only; never launch training."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "cache-root", "nvme-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    try:
        result = prepare_data(args.source, args.target, args.cache_root, args.nvme_root, args.output, args.workers)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        write_json(args.output / "status.json", {"status": "failed", "error": repr(exc), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()

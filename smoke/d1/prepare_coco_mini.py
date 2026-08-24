#!/usr/bin/env python3
"""Prepare a deterministic 100-image mini split from official COCO test2017."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


INFO_URL = "http://images.cocodataset.org/annotations/image_info_test2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/test2017"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "YOLO-Master-D1-admission/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.count != 100:
        raise ValueError("D1 admission split must contain exactly 100 images")

    download(INFO_URL, args.metadata_archive)
    with zipfile.ZipFile(args.metadata_archive) as archive:
        member = next(name for name in archive.namelist() if name.endswith("image_info_test2017.json"))
        image_info = json.loads(archive.read(member))
    selected = sorted(image_info["images"], key=lambda item: item["file_name"])[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"Official image-info contains only {len(selected)} selected records")

    image_dir = args.output / "images"
    jobs = [(f"{IMAGE_BASE_URL}/{item['file_name']}", image_dir / item["file_name"]) for item in selected]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda job: download(*job), jobs))

    records = []
    for item, (url, path) in zip(selected, jobs):
        with Image.open(path) as image:
            image.verify()
        records.append(
            {
                "id": item["id"],
                "file_name": item["file_name"],
                "width": item["width"],
                "height": item["height"],
                "url": url,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if len({record["sha256"] for record in records}) != args.count:
        raise RuntimeError("COCO mini source contains duplicate image content")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "source_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "result": "PASS",
        "dataset": "COCO test2017 mini-100",
        "source": IMAGE_BASE_URL,
        "metadata_source": INFO_URL,
        "metadata_archive_sha256": sha256_file(args.metadata_archive),
        "image_count": len(records),
        "unique_image_sha256_count": len({record["sha256"] for record in records}),
        "source_manifest_sha256": sha256_file(manifest),
    }
    (args.output / "source_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare and verify the immutable D1 WP0 teacher, COCO splits, and provenance manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
MODEL_REVISION = "2e601320d0545509ab03374e2f8707f303e1de7a"
MODELSCOPE_BASE = f"https://www.modelscope.cn/models/{MODEL_ID}/resolve/{MODEL_REVISION}"
MODEL_FILES = {
    "config.json": (743, "9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b"),
    "model.safetensors": (86_406_384, "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"),
    "LICENSE.md": (7_503, "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"),
    "README.md": (14_519, "e15ec258bf2fe83dd9473cb483ad0c7f49bc3ee6195e3c177f17d995fced3478"),
}
COCO_MIRROR_REVISION = "5466a7f1944225fcddb1896006508cad5be27b5b"
COCO_MIRROR_BASE = f"https://www.modelscope.cn/datasets/PAI/COCO2017/resolve/{COCO_MIRROR_REVISION}"
COCO_FILES = {
    "train2017.zip": (
        f"{COCO_MIRROR_BASE}/train2017.zip",
        "http://images.cocodataset.org/zips/train2017.zip",
        19_336_861_798,
        "69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929",
    ),
    "val2017.zip": (
        f"{COCO_MIRROR_BASE}/val2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        815_585_330,
        "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    ),
    "annotations_trainval2017.zip": (
        f"{COCO_MIRROR_BASE}/annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        252_907_541,
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    ),
    "coco2017labels.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
        48_639_045,
        "51a5175c894a7a1010f90eb4cba613473445f02633b684ed46c0292a997d0234",
    ),
}
EXPECTED_SPLITS = {"train2017": 118_287, "val2017": 5_000}
EXPECTED_MODEL_CONFIG = {
    "hidden_size": 384,
    "model_type": "dinov3_vit",
    "num_attention_heads": 6,
    "num_hidden_layers": 12,
    "num_register_tokens": 4,
    "patch_size": 16,
}
CHUNK_BYTES = 8 * 1024 * 1024
EXTRACT_WORKERS = 16
USER_AGENT = "YOLO-Master-D1-WP0/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def download_file(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    retries: int = 5,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size_matches = expected_size is None or destination.stat().st_size == expected_size
        hash_matches = expected_sha256 is None or sha256_file(destination) == expected_sha256
        if size_matches and hash_matches:
            return
        raise ValueError(f"existing file failed integrity validation: {destination}")
    part = destination.with_name(destination.name + ".part")
    if shutil.which("curl") is None:
        raise RuntimeError("D1 WP0 downloads require curl for reliable resumable transfers")
    resume_at = part.stat().st_size if part.exists() else 0
    print(f"Downloading {destination.name}: resume={resume_at} total={expected_size or 'unknown'}", flush=True)
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        str(retries),
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--user-agent",
        USER_AGENT,
        "--output",
        str(part),
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"download failed: {url}") from exc
    actual_size = part.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(f"{destination.name}: got {actual_size} bytes, expected {expected_size}")
    actual_sha256 = sha256_file(part)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(f"{destination.name}: got SHA256 {actual_sha256}, expected {expected_sha256}")
    os.replace(part, destination)


def validate_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"CRC validation failed for {path}: {bad_member}")


def _extract_members(
    path: Path,
    destination: Path,
    members: list[zipfile.ZipInfo],
    existing_sizes: dict[str, int],
) -> None:
    with zipfile.ZipFile(path) as archive:
        for member in members:
            if member.is_dir() or existing_sizes.get(member.filename) == member.file_size:
                continue
            target = destination / member.filename
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def extract_zip(path: Path, destination: Path) -> None:
    archive_sha256 = sha256_file(path)
    marker = destination / f".{path.name}.extracted"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == archive_sha256:
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe ZIP member: {member.filename}")
    directories = {
        destination / (Path(member.filename) if member.is_dir() else Path(member.filename).parent)
        for member in members
    }
    for directory in sorted(directories, key=lambda item: len(item.parts)):
        directory.mkdir(parents=True, exist_ok=True)
    existing_sizes = {}
    for current_root, _directories, filenames in os.walk(destination):
        for filename in filenames:
            target = Path(current_root) / filename
            existing_sizes[target.relative_to(destination).as_posix()] = target.stat().st_size
    pending_members = [
        member
        for member in members
        if not member.is_dir() and existing_sizes.get(member.filename) != member.file_size
    ]
    chunk_size = max(1, math.ceil(len(pending_members) / EXTRACT_WORKERS))
    chunks = [
        pending_members[index : index + chunk_size] for index in range(0, len(pending_members), chunk_size)
    ]
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
        futures = [
            executor.submit(_extract_members, path, destination, chunk, existing_sizes) for chunk in chunks if chunk
        ]
        for future in futures:
            future.result()
    for member in members:
        if member.is_dir():
            continue
        target = destination / member.filename
        if not target.is_file() or target.stat().st_size != member.file_size:
            raise RuntimeError(f"incomplete extracted member: {member.filename}")
    marker.write_text(archive_sha256 + "\n", encoding="utf-8")


def download_and_extract(workspace: Path) -> tuple[Path, Path]:
    coco_root = workspace / "datasets" / "coco"
    source_root = coco_root / "source"
    teacher_root = workspace / "weights" / "teachers" / "dinov3-vits16-pretrain-lvd1689m"

    for name, (expected_size, expected_sha256) in MODEL_FILES.items():
        download_file(
            f"{MODELSCOPE_BASE}/{name}",
            teacher_root / name,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )

    for name, (download_url, _upstream_url, expected_size, expected_sha256) in COCO_FILES.items():
        download_file(
            download_url,
            source_root / name,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        validate_zip(source_root / name)

    extract_zip(source_root / "train2017.zip", coco_root / "images")
    extract_zip(source_root / "val2017.zip", coco_root / "images")
    extract_zip(source_root / "annotations_trainval2017.zip", coco_root)
    extract_zip(source_root / "coco2017labels.zip", workspace / "datasets")
    return coco_root, teacher_root


def build_split_list(coco_root: Path, split: str) -> list[str]:
    image_dir = coco_root / "images" / split
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: path.name)
    relative = [path.relative_to(coco_root).as_posix() for path in paths]
    expected = EXPECTED_SPLITS[split]
    if len(relative) != expected:
        raise ValueError(f"{split}: found {len(relative)} images, expected {expected}")
    if len(relative) != len(set(relative)):
        raise ValueError(f"{split}: duplicate image paths")
    return relative


def assert_disjoint_splits(train: list[str], val: list[str]) -> None:
    train_names = {Path(path).name for path in train}
    val_names = {Path(path).name for path in val}
    overlap = sorted(train_names & val_names)
    if overlap:
        raise ValueError(f"COCO train2017 and val2017 overlap: {overlap[:5]}")


def write_lines(path: Path, lines: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(lines) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def verify_labels(coco_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in EXPECTED_SPLITS:
        label_dir = coco_root / "labels" / split
        if not label_dir.is_dir():
            raise FileNotFoundError(f"missing label directory: {label_dir}")
        counts[split] = sum(1 for path in label_dir.glob("*.txt") if path.is_file())
        if counts[split] <= 0:
            raise ValueError(f"{split}: no YOLO labels found")
    return counts


def verify_model(teacher_root: Path, load_model: bool) -> dict[str, Any]:
    config_path = teacher_root / "config.json"
    weights_path = teacher_root / "model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_MODEL_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"model config {key}={config.get(key)!r}, expected {expected!r}")
    if weights_path.stat().st_size != 86_406_384:
        raise ValueError(f"unexpected model.safetensors size: {weights_path.stat().st_size}")
    if load_model:
        from transformers import DINOv3ViTBackbone

        model = DINOv3ViTBackbone.from_pretrained(teacher_root, local_files_only=True)
        if int(model.config.hidden_size) != 384:
            raise ValueError("loaded DINOv3 model has an unexpected hidden size")
        del model
    files = {}
    for name in MODEL_FILES:
        path = teacher_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"config": {key: config[key] for key in sorted(EXPECTED_MODEL_CONFIG)}, "files": files}


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def environment_manifest(repo: Path) -> dict[str, Any]:
    import torch
    import transformers
    import ultralytics

    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    return {
        "code": {
            "branch": git_output(repo, "branch", "--show-current"),
            "commit": git_output(repo, "rev-parse", "HEAD"),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "runtime": torch.version.cuda,
        },
        "gpus": [line.strip() for line in gpu_lines],
        "os": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "system": platform.system(),
        },
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "python": platform.python_version(),
        "schema_version": "d1-environment-v1",
    }


def generate_manifests(workspace: Path, repo: Path, load_model: bool) -> None:
    coco_root = workspace / "datasets" / "coco"
    teacher_root = workspace / "weights" / "teachers" / "dinov3-vits16-pretrain-lvd1689m"
    output_root = repo / "experiments" / "d1" / "manifests"
    workspace_manifest_root = workspace / "manifests"

    split_lists = {split: build_split_list(coco_root, split) for split in EXPECTED_SPLITS}
    assert_disjoint_splits(split_lists["train2017"], split_lists["val2017"])
    label_counts = verify_labels(coco_root)
    list_metadata = {}
    for split, lines in split_lists.items():
        filename = f"coco2017-{split}.txt"
        digest = write_lines(output_root / filename, lines)
        write_lines(workspace_manifest_root / filename, lines)
        list_metadata[split] = {"count": len(lines), "path": filename, "sha256": digest}

    archives = {}
    for name, (download_url, upstream_url, expected_size, expected_sha256) in COCO_FILES.items():
        path = coco_root / "source" / name
        if path.stat().st_size != expected_size:
            raise ValueError(f"{name}: size changed after extraction")
        actual_sha256 = sha256_file(path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"{name}: SHA256 changed after extraction")
        archives[name] = {
            "bytes": path.stat().st_size,
            "download_url": download_url,
            "sha256": actual_sha256,
            "upstream_url": upstream_url,
        }
        if download_url != upstream_url:
            archives[name]["mirror_revision"] = COCO_MIRROR_REVISION
    split_manifest = {
        "archives": archives,
        "labels": label_counts,
        "schema_version": "d1-coco2017-splits-v1",
        "split_policy": "canonical COCO 2017 train2017/val2017; sorted by image filename; no random split",
        "splits": list_metadata,
    }
    model_details = verify_model(teacher_root, load_model=load_model)
    model_manifest = {
        **model_details,
        "license": {
            "name": "DINOv3 License",
            "url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
        },
        "model_id": MODEL_ID,
        "schema_version": "d1-teacher-v1",
        "source": f"https://www.modelscope.cn/models/{MODEL_ID}",
        "source_revision": MODEL_REVISION,
    }
    generated = {
        "coco2017-splits.json": split_manifest,
        "dinov3-vits16.json": model_manifest,
        "environment.json": environment_manifest(repo),
    }
    for filename, payload in generated.items():
        write_json(output_root / filename, payload)
        write_json(workspace_manifest_root / filename, payload)


def verify_contract(repo: Path) -> None:
    contract_path = repo / "experiments" / "d1" / "manifests" / "p0-experiment-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["teacher"]["model_id"] != MODEL_ID:
        raise ValueError("contract teacher model does not match WP0")
    if contract["features"]["output_blocks"] != [
        {"implementation_index": 3, "name": "block4", "ordinal": 4},
        {"implementation_index": 7, "name": "block8", "ordinal": 8},
        {"implementation_index": 11, "name": "block12", "ordinal": 12},
    ]:
        raise ValueError("contract output block indices changed")
    if contract["features"]["grid_size"] != [40, 40]:
        raise ValueError("contract grid size changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="External D1 data and weight workspace")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2], help="YOLO-Master repository")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="Download, extract, verify, and generate manifests")
    mode.add_argument("--verify-only", action="store_true", help="Verify existing artifacts and regenerate manifests")
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip Transformers construction; intended only for lightweight unit tests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    repo = args.repo.resolve()
    verify_contract(repo)
    if args.download:
        download_and_extract(workspace)
    generate_manifests(workspace, repo, load_model=not args.skip_model_load)
    print(f"D1 WP0 verified: workspace={workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

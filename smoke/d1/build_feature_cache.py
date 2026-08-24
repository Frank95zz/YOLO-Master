#!/usr/bin/env python3
"""Build reproducible D1 admission feature caches and resource evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LAYER_INDICES = (2, 5, 8, 11)
LAYER_NAMES = ("block03", "block06", "block09", "block12")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--output", type=Path, required=True, help="New cache directory")
    parser.add_argument("--repo", type=Path, required=True, help="YOLO-Master git repository")
    parser.add_argument("--dinov2-repo", type=Path, required=True, help="Local official DINOv2 source checkout")
    parser.add_argument("--weights", type=Path, required=True, help="Official DINOv2 ViT-S/14 weights")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-name", default="custom-mini-100")
    parser.add_argument("--dataset-source", default="local")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(dense: dict[str, torch.Tensor], pooled: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name in sorted(dense):
        tensor = dense[name].contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    digest.update(b"pooled")
    digest.update(pooled.contiguous().numpy().tobytes())
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def image_paths(root: Path, limit: int) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) < limit:
        raise RuntimeError(f"Need {limit} source images, found {len(paths)} in {root}")
    selected = paths[:limit]
    if len({path.resolve() for path in selected}) != limit:
        raise RuntimeError("Source image list contains duplicate paths")
    return selected


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_mib(value: int | float) -> str:
    return f"{value / 1024**2:.2f} MiB"


def main() -> None:
    args = parse_args()
    if args.limit != 100:
        raise ValueError("D1 admission evidence must use exactly 100 images")
    if args.imgsz % 14:
        raise ValueError("imgsz must be divisible by the DINOv2 patch size (14)")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty cache directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output / "features"
    feature_dir.mkdir()

    images = image_paths(args.images, args.limit)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    commit = git_output(args.repo, "rev-parse", "HEAD")
    status = git_output(args.repo, "status", "--short")
    if status:
        raise RuntimeError("Admission cache must be generated from a clean git worktree")

    model = torch.hub.load(
        str(args.dinov2_repo),
        "dinov2_vits14",
        source="local",
        weights=str(args.weights),
    )
    model.eval().requires_grad_(False).to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Foundation model freeze contract failed")

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((args.imgsz, args.imgsz), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN, std=STD),
        ]
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    records: list[dict[str, object]] = []

    for batch_start in range(0, len(images), args.batch):
        batch_paths = images[batch_start : batch_start + args.batch]
        tensors = []
        original_sizes = []
        source_hashes = []
        for path in batch_paths:
            source_hashes.append(sha256_file(path))
            with Image.open(path) as image:
                image = image.convert("RGB")
                original_sizes.append([image.height, image.width])
                tensors.append(transform(image))
        batch = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                layers = model.get_intermediate_layers(
                    batch,
                    n=list(LAYER_INDICES),
                    reshape=True,
                    return_class_token=True,
                    norm=True,
                )

        for offset, path in enumerate(batch_paths):
            dense = {
                name: layer[0][offset].detach().to(device="cpu", dtype=torch.float16).contiguous()
                for name, layer in zip(LAYER_NAMES, layers)
            }
            pooled = layers[-1][1][offset].detach().to(device="cpu", dtype=torch.float16).contiguous()
            payload = {
                "dense": dense,
                "pooled": pooled,
                "metadata": {
                    "source": str(path),
                    "source_sha256": source_hashes[offset],
                    "original_hw": original_sizes[offset],
                    "input_hw": [args.imgsz, args.imgsz],
                    "patch_size": 14,
                    "model": "facebookresearch/dinov2:dinov2_vits14",
                    "weights": str(args.weights),
                    "git_commit": commit,
                },
            }
            cache_path = feature_dir / f"{batch_start + offset:03d}-{path.stem}.pt"
            torch.save(payload, cache_path)
            records.append(
                {
                    "index": batch_start + offset,
                    "source": str(path),
                    "source_sha256": source_hashes[offset],
                    "cache": str(cache_path),
                    "cache_bytes": cache_path.stat().st_size,
                    "cache_sha256": sha256_file(cache_path),
                    "feature_tensor_sha256": tensor_digest(dense, pooled),
                    "dense_shapes": {name: list(tensor.shape) for name, tensor in dense.items()},
                    "pooled_shape": list(pooled.shape),
                    "dtype": str(pooled.dtype),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - started
    manifest_path = args.output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    read_started = time.perf_counter()
    for record in records:
        cached = torch.load(record["cache"], map_location="cpu", weights_only=True)
        actual = tensor_digest(cached["dense"], cached["pooled"])
        if actual != record["feature_tensor_sha256"]:
            raise RuntimeError(f"Cache reload hash mismatch: {record['cache']}")
    read_seconds = time.perf_counter() - read_started

    cache_bytes = sum(int(record["cache_bytes"]) for record in records)
    source_bytes = sum(path.stat().st_size for path in images)
    aggregate = hashlib.sha256(
        "\n".join(str(record["feature_tensor_sha256"]) for record in records).encode()
    ).hexdigest()
    gpu_peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    gpu_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    disk = os.statvfs(args.output)

    summary = {
        "result": "PASS",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "git_worktree_clean": True,
        "dataset": {
            "name": args.dataset_name,
            "source": args.dataset_source,
            "source_root": str(args.images),
            "image_count": len(records),
            "source_bytes": source_bytes,
            "source_manifest_sha256": hashlib.sha256(
                "\n".join(str(record["source_sha256"]) for record in records).encode()
            ).hexdigest(),
        },
        "foundation_model": {
            "requested": "DINOv3 ViT-S/16",
            "actual": "DINOv2 ViT-S/14",
            "downgrade_declared": True,
            "downgrade_reason": "DINOv3 weights are gated; DINOv2 is the licensed fallback allowed by D1.",
            "weights": str(args.weights),
            "weights_bytes": args.weights.stat().st_size,
            "weights_sha256": sha256_file(args.weights),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
            ),
        },
        "interface": {
            "input": ["B", 3, args.imgsz, args.imgsz],
            "patch_size": 14,
            "layers": {name: ["B", 384, args.imgsz // 14, args.imgsz // 14] for name in LAYER_NAMES},
            "pooled": ["B", 384],
            "cache_dtype": "float16",
            "latent_mixture_targets_at_imgsz_224": {
                "p3": ["B", 256, 28, 28],
                "p4": ["B", 512, 14, 14],
                "p5": ["B", 1024, 7, 7],
            },
        },
        "resources": {
            "device": str(device),
            "gpu": gpu_name,
            "batch": args.batch,
            "encode_seconds": encode_seconds,
            "images_per_second": len(records) / encode_seconds,
            "cache_bytes": cache_bytes,
            "cache_mib_per_image": cache_bytes / len(records) / 1024**2,
            "estimated_cache_gib_per_100k_images": cache_bytes / len(records) * 100000 / 1024**3,
            "cache_read_seconds": read_seconds,
            "cache_read_mib_per_second": cache_bytes / 1024**2 / read_seconds,
            "peak_cuda_allocated_bytes": gpu_peak_allocated,
            "peak_cuda_reserved_bytes": gpu_peak_reserved,
            "disk_free_bytes_after": disk.f_bavail * disk.f_frsize,
        },
        "verification": {
            "cache_reload_count": len(records),
            "cache_reload_verified": True,
            "aggregate_feature_tensor_sha256": aggregate,
            "manifest_sha256": sha256_file(manifest_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    write_json(args.output / "summary.json", summary)

    dimension_table = f"""# D1 Interface Dimension Table

| Stage | Tensor shape | Dtype | Notes |
| --- | --- | --- | --- |
| Input | `B x 3 x {args.imgsz} x {args.imgsz}` | FP32 | Resize, ImageNet normalize |
| DINOv2 block 3 | `B x 384 x {args.imgsz // 14} x {args.imgsz // 14}` | FP16 cache | Patch size 14 |
| DINOv2 block 6 | `B x 384 x {args.imgsz // 14} x {args.imgsz // 14}` | FP16 cache | Multi-layer feature |
| DINOv2 block 9 | `B x 384 x {args.imgsz // 14} x {args.imgsz // 14}` | FP16 cache | Multi-layer feature |
| DINOv2 block 12 | `B x 384 x {args.imgsz // 14} x {args.imgsz // 14}` | FP16 cache | Final dense feature |
| Global pooled | `B x 384` | FP16 cache | Final normalized CLS token |
| P3 adapter target | `B x 256 x 28 x 28` | FP32/AMP | Resize + channel projection required |
| P4 adapter target | `B x 512 x 14 x 14` | FP32/AMP | Resize + channel projection required |
| P5 adapter target | `B x 1024 x 7 x 7` | FP32/AMP | Resize + channel projection required |

The current 8.24 check validates the frozen feature/cache boundary. The P3/P4/P5 projections are the explicit next
implementation boundary for the D1 P0 train/predict route; this report does not claim that boundary is already wired.
"""
    (args.output / "dimension_table.md").write_text(dimension_table, encoding="utf-8")

    resource_report = f"""# D1 Resource Report

- Result: PASS
- Dataset: {args.dataset_name}
- Images: {len(records)} distinct images
- Encode time: {encode_seconds:.3f} s ({len(records) / encode_seconds:.2f} images/s)
- Source image size: {format_mib(source_bytes)}
- Feature cache size: {format_mib(cache_bytes)} ({cache_bytes / len(records) / 1024**2:.3f} MiB/image)
- Estimated cache size for 100k images: {cache_bytes / len(records) * 100000 / 1024**3:.2f} GiB
- Cache reload throughput: {cache_bytes / 1024**2 / read_seconds:.2f} MiB/s
- Peak CUDA allocated: {format_mib(gpu_peak_allocated)}
- Peak CUDA reserved: {format_mib(gpu_peak_reserved)}
- Disk free after run: {disk.f_bavail * disk.f_frsize / 1024**3:.2f} GiB
- Cache reload verification: 100/100 tensor hashes passed
- Aggregate tensor SHA256: `{aggregate}`
"""
    (args.output / "resource_report.md").write_text(resource_report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

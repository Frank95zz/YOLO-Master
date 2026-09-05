#!/usr/bin/env python3
"""Prepare D1 follow-up variants and run bounded checks, never epoch training."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import torch

from scripts.d1.cache_features import cache_contract, load_image, make_letterbox, split_paths
from scripts.d1.diagnose_wp8 import git_state, sha256_lines, write_json
from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.foundation.cache import (
    FeatureCacheReader,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_tensor,
)
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/experiments/d1/wp8-followup.yaml"
NAMES = ("block4", "block8", "block12")


def variants():
    """Resolve isolated A/B/C models without changing the historical default."""
    contract = YAML.load(CONFIG)
    if contract["training_authorized"] is not False or contract["checks"]["optimizer_steps"] != 0:
        raise ValueError("Follow-up preparation must not authorize training")
    base = YAML.load(ROOT / contract["base_model"])
    result = {}
    for name, override in contract["variants"].items():
        model = deepcopy(base)
        model["latent_mixture"]["value_fusion_mode"] = override["value_fusion_mode"]
        if "value_fusion_weights" in override:
            model["latent_mixture"]["value_fusion_weights"] = override["value_fusion_weights"]
        model["loss"]["latent_aux_gain"] = override["latent_aux_gain"]
        result[name] = model
    return contract, result


def identity():
    """Bind reports to source contents even before a documentation-only commit."""
    paths = (
        "scripts/d1/inspect_wp8_followup.py",
        "ultralytics/nn/mixture_loss.py",
        "ultralytics/cfg/experiments/d1/wp8-followup.yaml",
        "ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-n.yaml",
    )
    return {**git_state(ROOT), "files": {p: sha256_file(ROOT / p) for p in paths}}


def fresh_output(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Runtime output must be outside the Git repository")
    path.mkdir(parents=True, exist_ok=False)
    return path


def select_records(reader, split, count=8):
    """Spread the fixed sample across the entire canonical split, including both ends."""
    canonical, _ = split_paths(ROOT, split, None)
    expected = 118287 if split == "train2017" else 5000
    if len(canonical) != expected or len(reader.records) != expected:
        raise ValueError("Full official COCO split is required")
    indices = [i * (expected - 1) // (count - 1) for i in range(count)]
    selected = []
    for index in indices:
        relative = canonical[index]
        sample_id = f"{split}/{Path(relative).stem}"
        record = reader.records[sample_id]
        if record["sample_id"] != sample_id or record["image_path"] != relative:
            raise ValueError("Cache image identity does not match the canonical split")
        selected.append(record)
    return selected


def prepare(output):
    contract, configs = variants()
    rows = {}
    for name, config in configs.items():
        torch.manual_seed(0)
        model = D1FoundationDetectionModel(config)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if parameters != 3542567:
            raise ValueError(f"{name} parameter budget changed: {parameters}")
        YAML.save(output / f"model-{name}.yaml", config)
        rows[name] = {
            "trainable_parameters": parameters,
            "model_sha256": sha256_bytes(canonical_json_bytes(config)),
            "fusion": config["latent_mixture"]["value_fusion_mode"],
            "latent_aux_gain": config["loss"]["latent_aux_gain"],
        }
    report = {"status": "prepared_not_trained", "identity": identity(), "variants": rows,
              "aux_contract": contract["aux_contract"], "training_authorized": False}
    write_json(output / "preparation.json", report)
    return report


def resident_probe(config, batch, device):
    """Time a tiny resident batch; no optimizer exists and no parameter is updated."""
    torch.manual_seed(0)
    model = D1FoundationDetectionModel(config).to(device).train()
    stages = {}
    handles = []

    def before(name):
        def callback(*_):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            stages[name] = [event]
        return callback

    def after(name):
        def callback(*_):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            stages[name].append(event)
        return callback

    modules = {"adapter": model.adapter, **dict(model.mixtures.items()), "detect": model.detect}
    for name, module in modules.items():
        handles.extend((module.register_forward_pre_hook(before(name)), module.register_forward_hook(after(name))))
    elapsed, per_stage = [], {name: [] for name in modules}
    branch_gradients = {}
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    try:
        for step in range(7):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                loss, items = model(batch)
            if not torch.isfinite(loss).all() or not torch.isfinite(items).all():
                raise FloatingPointError("Non-finite diagnostic loss")
            loss.sum().backward()
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Non-finite diagnostic gradients")
            if step >= 2:
                elapsed.append(duration)
                for name, (start, end) in stages.items():
                    per_stage[name].append(start.elapsed_time(end))
        for level, branches in model.adapter.branches.items():
            for source, branch in branches.items():
                branch_gradients[f"{level}/{source}"] = sum(
                    float(p.grad.detach().float().norm()) for p in branch.parameters() if p.grad is not None
                )
        if any(not torch.equal(p.detach(), initial[name]) for name, p in model.named_parameters()):
            raise AssertionError("Read-only probe updated a parameter")
        return {"status": "passed", "optimizer_steps": 0, "batch": 2, "steps": 5, "warmup": 2,
                "resident_forward_backward_ms": 1000 * sum(elapsed) / len(elapsed),
                "forward_module_gpu_ms": {k: sum(v) / len(v) for k, v in per_stage.items()},
                "adapter_gradient_norms": branch_gradients,
                "loss_items": items.detach().float().cpu().tolist(),
                "limitation": "Single GPU, resident batch 2; not DDP, cold I/O, throughput, or full-training ETA"}
    finally:
        for handle in handles:
            handle.remove()


def check(args, output):
    from ultralytics.data.d1_cache import D1FeatureCacheDataset
    from ultralytics.nn.foundation import DINOv3Teacher
    from ultralytics.cfg import get_cfg

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Bounded real-input checks require CUDA")
    torch.cuda.set_device(device)
    expected = cache_contract(ROOT)
    weight_hash = sha256_file(args.weights_dir / "model.safetensors")
    if weight_hash != expected["teacher_weights_sha256"]:
        raise ValueError("Teacher weights differ from WP0")
    teacher = DINOv3Teacher(model_id=expected["model_id"], weights_path=args.weights_dir,
                           local_files_only=True, dtype="fp16", device=str(device), output_layers=(4, 8, 12))
    reports, train_paths = {}, []
    letterbox = make_letterbox()
    for split, root in (("train2017", args.train_cache), ("val2017", args.val_cache)):
        reader = FeatureCacheReader(root, max_open_shards=2)
        try:
            if reader.contract != expected:
                raise ValueError("Cache contract differs from WP0")
            selected = select_records(reader, split)
            errors = []
            samples = []
            for record in selected:
                path = args.data_root / record["image_path"]
                if sha256_file(path) != record["image_sha256"]:
                    raise ValueError("Source image hash differs from cache record")
                if split == "train2017":
                    train_paths.append(str(path.resolve()))
                cached = reader.get(record["sample_id"])
                online = teacher.encode(load_image(path, letterbox).unsqueeze(0))
                layers = {}
                for name in NAMES:
                    reference = cached[name]
                    if sha256_tensor(reference) != record["tensors"][name]["sha256"]:
                        raise ValueError("Selected cache tensor checksum mismatch")
                    actual = online.dense[name][0].detach().cpu().to(torch.float16)
                    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
                    matched = finite and bool(torch.allclose(actual, reference, rtol=1e-3, atol=1e-3))
                    delta = (actual.float() - reference.float()).abs()
                    layers[name] = {"passed": matched, "max_absolute_error": float(delta.max()),
                                    "mean_absolute_error": float(delta.mean())}
                    if not matched:
                        errors.append(f"{record['sample_id']}:{name}")
                samples.append({"sample_id": record["sample_id"], "layers": layers})
            reports[split] = {"status": "passed" if not errors else "failed", "failures": errors,
                              "cache_content_sha256": reader.index["content_sha256"],
                              "cache_contract_sha256": reader.index["contract_sha256"], "samples": samples}
        finally:
            reader.close()
    frozen = not teacher.training and not teacher.model.training and not any(p.requires_grad for p in teacher.parameters())
    del teacher
    torch.cuda.empty_cache()
    if not frozen or any(r["status"] != "passed" for r in reports.values()):
        write_json(output / "checks.json", {"status": "failed", "identity": identity(), "parity": reports})
        raise RuntimeError("Online/cache parity failed; do not authorize training")
    split_file = output / "selected-train.txt"
    split_file.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    coco = YAML.load(ROOT / "ultralytics/cfg/datasets/coco.yaml")
    dataset = D1FeatureCacheDataset(img_path=str(split_file), cache_dir=args.train_cache,
                                   data={"names": coco["names"], "nc": 80, "channels": 3},
                                   imgsz=640, batch_size=2, hyp=get_cfg(), max_open_shards=2)
    try:
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        batch["features"] = batch["features"].to(device, dtype=torch.float16)
        batch["img"] = batch["features"]
        for key in ("batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device)
        _, configs = variants()
        probes = {name: resident_probe(config, batch, device) for name, config in configs.items()}
    finally:
        dataset.feature_reader.close()
    report = {"status": "passed", "identity": identity(), "teacher_weights_sha256": weight_hash,
              "teacher_frozen": frozen, "parity": reports, "probes": probes, "training_authorized": False,
              "optimizer_steps": 0, "full_cache_reverified": False}
    write_json(output / "checks.json", report)
    return report


def scratch_train_eval(args, output):
    """Evaluate the existing scratch checkpoint on P0's fixed 5000 training images."""
    from scripts.d1 import run_wp8_p1_control as control
    from scripts.d1.launch_wp8_p1 import OfficialValidator, official_metrics
    from ultralytics.nn.tasks import DetectionModel

    relative, digest = split_paths(ROOT, "train2017", 5000)
    reference = json.loads(args.p0_report.read_text())
    if reference["selected_paths_sha256"] != digest or reference["selected_sample_count"] != 5000:
        raise ValueError("Training diagnostic subset differs from the frozen reference")
    paths = [str((args.data_root / p).resolve()) for p in relative]
    if not all(Path(path).is_file() for path in paths):
        raise FileNotFoundError("A selected COCO training image is missing")
    listing = output / "train5000.txt"
    listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
    coco = YAML.load(ROOT / "ultralytics/cfg/datasets/coco.yaml")
    data_yaml = output / "train5000.yaml"
    YAML.save(data_yaml, {"path": str(args.data_root.resolve()), "train": str(listing),
                         "val": str(listing), "names": coco["names"]})
    metadata = control.strict_checkpoint(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("ema")
    if saved is None:
        saved = checkpoint["model"]
    model = DetectionModel(YAML.load(ROOT / control.MODEL), nc=80, ch=3, verbose=False)
    model.load_state_dict(saved.float().state_dict(), strict=True)
    del checkpoint, saved
    validator = OfficialValidator(save_dir=output, args={
        "task": "detect", "data": str(data_yaml), "imgsz": 640, "batch": 64, "workers": 4,
        "device": args.device.removeprefix("cuda:"), "rect": False, "half": False,
        "quantize": None, "save_json": True, "plots": False, "verbose": False,
        "conf": 0.001, "iou": 0.7, "max_det": 300,
    })
    started = time.perf_counter()
    internal = validator(model=model.eval())
    seen = sorted(int(Path(p).stem) for p in validator.dataloader.dataset.im_files)
    if validator.seen != 5000 or seen != sorted(int(Path(p).stem) for p in relative):
        raise RuntimeError("Evaluation did not cover the exact 5000 diagnostic images")
    write_json(output / "predictions.json", validator.jdict)
    official = official_metrics(args.data_root / "annotations/instances_train2017.json",
                                output / "predictions.json", seen)
    internal = {key: float(value) for key, value in internal.items()}
    if not all(math.isfinite(value) for value in (*internal.values(), *official.values())):
        raise FloatingPointError("Non-finite diagnostic evaluation")
    report = {"status": "passed", "identity": identity(), "checkpoint": metadata,
              "selected_paths_sha256": digest, "seen": 5000, "internal": internal, "official": official,
              "frozen_reference_official": reference["official_coco_metrics"],
              "elapsed_seconds": time.perf_counter() - started, "optimizer_steps": 0,
              "scope": "Fixed train5000 diagnostic, not full train AP or held-out validation"}
    write_json(output / "report.json", report)
    return report



def probe(args, output):
    """Inspect cached downstream computation independently of online batch invariance."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.d1_cache import D1FeatureCacheDataset

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Real-input probe requires CUDA")
    torch.cuda.set_device(device)
    reader = FeatureCacheReader(args.train_cache)
    try:
        if reader.contract != cache_contract(ROOT):
            raise ValueError("Cache contract differs from WP0")
        selected = select_records(reader, "train2017")
        paths = [str((args.data_root / r["image_path"]).resolve()) for r in selected]
        for record, path in zip(selected[:2], paths[:2]):
            if sha256_file(path) != record["image_sha256"]:
                raise ValueError("Selected image checksum mismatch")
            for name, value in reader.get(record["sample_id"]).items():
                if sha256_tensor(value) != record["tensors"][name]["sha256"]:
                    raise ValueError("Selected tensor checksum mismatch")
        content = reader.index["content_sha256"]
    finally:
        reader.close()
    listing = output / "selected-train.txt"
    listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
    coco = YAML.load(ROOT / "ultralytics/cfg/datasets/coco.yaml")
    dataset = D1FeatureCacheDataset(img_path=str(listing), cache_dir=args.train_cache,
                                   data={"names": coco["names"], "nc": 80, "channels": 3},
                                   imgsz=640, batch_size=2, hyp=get_cfg(), max_open_shards=2)
    try:
        batch = dataset.collate_fn([dataset[0], dataset[1]])
        batch["features"] = batch["features"].to(device, dtype=torch.float16)
        batch["img"] = batch["features"]
        for key in ("batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device)
        _, configs = variants()
        probes = {name: resident_probe(config, batch, device) for name, config in configs.items()}
    finally:
        dataset.feature_reader.close()
    report = {"status": "passed", "identity": identity(), "probes": probes, "optimizer_steps": 0,
              "cache_content_sha256": content, "training_authorized": False,
              "scope": "Downstream-only component check; does not pass the online parity gate"}
    write_json(output / "probe.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "probe", "scratch-train-eval"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--val-cache", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--p0-report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    required = {
        "prepare": (), "check": ("data_root", "train_cache", "val_cache", "weights_dir"),
        "probe": ("data_root", "train_cache"),
        "scratch-train-eval": ("data_root", "checkpoint", "p0_report"),
    }
    if any(getattr(args, key) is None for key in required[args.command]):
        parser.error(f"{args.command} requires {required[args.command]}")
    output = fresh_output(args.output_dir)
    try:
        if args.command == "prepare":
            report = prepare(output)
        elif args.command == "check":
            report = check(args, output)
        elif args.command == "probe":
            report = probe(args, output)
        else:
            report = scratch_train_eval(args, output)
        print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "identity": identity(),
                                             "error": repr(exc), "training_authorized": False})
        raise


if __name__ == "__main__":
    main()

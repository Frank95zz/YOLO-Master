"""Prepare and explicitly launch P5-only ablations using the existing E1 runtime."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import torch

from scripts.d1 import run_p1p2 as e1
from scripts.d1.p1p2_runtime import E1FrozenTrainer, atomic_torch, load_initial_tensors
from scripts.d1.run_wp8_p1_control import LOCKED_TRAIN, write_json
from ultralytics.nn.foundation.cache import sha256_file
from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/experiments/d1/p1p2/p5-coco2017.yaml"
SCHEMA = "d1-p5-ablation-v1"
P5_PREFIX = "model.adapter.branches.p5."
VARIANTS = {
    "BASE": {"p5_mode": "conv", "p5_parameters": 2655744, "downstream_parameters": 3542567},
    "DW": {"p5_mode": "depthwise", "p5_parameters": 309120, "downstream_parameters": 1195943},
    "BN64": {
        "p5_mode": "bottleneck", "p5_bottleneck_channels": 64,
        "p5_parameters": 518016, "downstream_parameters": 1404839,
    },
}
MODEL_FILES = {
    "DW": ROOT / "ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-dw-n.yaml",
    "BN64": ROOT / "ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml",
}


def load_contract(path=CONFIG):
    """Reject drift instead of silently changing the registered screening recipe."""
    e1.load_contract()
    expected = {
        "schema_version": SCHEMA,
        "base_contract": str(e1.CONFIG.relative_to(ROOT)),
        "base_variant": "B", "schedule_epochs": 100, "screen_epochs": 50, "seed": 0, "maximum_runs": 3,
        "selection": {
            "max_ap_drop_points": 0.5, "max_ap75_drop_points": 1.0, "max_ap_large_drop_points": 1.0,
            "require_lower_measured_gpu_hours": True,
        },
        "variants": VARIANTS,
    }
    value = YAML.load(path)
    if value != expected or any(type(value.get(k)) is not int for k in ("seed", "screen_epochs", "maximum_runs")):
        raise ValueError("Unknown or modified P5 ablation contract")
    return value


def model_config(variant, p3_upsample_mode="bilinear"):
    if not isinstance(p3_upsample_mode, str) or p3_upsample_mode not in {"bilinear", "separable_bilinear2x"}:
        raise ValueError("Unknown registered P3 upsample implementation")
    if variant not in VARIANTS:
        raise ValueError("Unknown P5 variant")
    expected = e1.model_config("B")
    expected["adapter"]["p5_mode"] = VARIANTS[variant]["p5_mode"]
    if variant == "BN64":
        expected["adapter"]["p5_bottleneck_channels"] = 64
    if variant in MODEL_FILES:
        actual = YAML.load(MODEL_FILES[variant])
        if actual != expected:
            raise ValueError(f"{variant} changes fields outside its registered P5 architecture")
    if p3_upsample_mode != "bilinear":
        expected["adapter"]["p3_upsample_mode"] = p3_upsample_mode
    return expected


def construct_model(variant, p3_upsample_mode="bilinear"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = D1FoundationDetectionModel(model_config(variant, p3_upsample_mode))
        initialize_mixture_loss_ema_buffer(model)
    expected = VARIANTS[variant]
    if sum(p.numel() for p in model.parameters()) != expected["downstream_parameters"]:
        raise ValueError("Unexpected downstream parameter count")
    if sum(p.numel() for p in model.adapter.branches["p5"].parameters()) != expected["p5_parameters"]:
        raise ValueError("Unexpected P5 parameter count")
    return model


def paired_initial_state(model, source_tensors, variant):
    """Keep every unchanged tensor identical; only new P5 parameters are freshly initialized."""
    reference = construct_model("BASE")
    expected_keys = {k for k, v in reference.state_dict().items() if isinstance(v, torch.Tensor)}
    if set(source_tensors) != expected_keys:
        raise ValueError("Source initial tensor keys differ from the original architecture")
    for key, tensor in source_tensors.items():
        target = reference.state_dict()[key]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError(f"Source initial tensor metadata changed: {key}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Source initial tensor is non-finite: {key}")
    tensors = {}
    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        if variant == "BASE" or not key.startswith(P5_PREFIX):
            source = source_tensors[key]
            if source.shape != value.shape or source.dtype != value.dtype:
                raise ValueError(f"Shared initial tensor mismatch: {key}")
            tensors[key] = source.detach().clone()
        else:
            tensors[key] = value.detach().clone()
    load_initial_tensors(model, tensors)
    return tensors


def identity():
    source = e1.identity()
    source["contract_sha256"] = sha256_file(CONFIG)
    return source


def inspect(p3_upsample_mode="bilinear"):
    load_contract()
    result = {}
    for variant in VARIANTS:
        model = construct_model(variant, p3_upsample_mode)
        result[variant] = {
            **VARIANTS[variant], "adapter_parameters": sum(p.numel() for p in model.adapter.parameters()),
            "feature_shapes": {"p3": [64, 80, 80], "p4": [128, 40, 40], "p5": [256, 20, 20]},
            "candidates_per_scale": 3, "teacher_in_training_model": False,
        }
    return {"status": "configuration_ready", "training_started": False,
            "p3_upsample_mode": p3_upsample_mode, "variants": result}


def prepare(source_workspace, workspace, p3_upsample_mode="bilinear"):
    contract = load_contract()
    model_config("BASE", p3_upsample_mode)
    current = identity()
    source_workspace, workspace = source_workspace.resolve(), workspace.resolve()
    if workspace == source_workspace or ROOT in workspace.parents or workspace == ROOT:
        raise ValueError("P5 requires a new external workspace")
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError("P5 workspace must be empty; preparation cannot overwrite prior state")
    source = e1.load_matrix(source_workspace, check_code=False)
    tensors = torch.load(source_workspace / "inputs/initial-frozen.pt", map_location="cpu", weights_only=True)
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    models = {}
    for variant in VARIANTS:
        model = construct_model(variant, p3_upsample_mode)
        initial = paired_initial_state(model, tensors, variant)
        atomic_torch(inputs / f"initial-{variant}.pt", initial)
        path = inputs / f"model-{variant}.yaml"
        YAML.save(path, model_config(variant, p3_upsample_mode))
        models[variant] = str(path)
    matrix = {
        "schema_version": SCHEMA, "identity": current, "contract": contract, "workspace": str(workspace),
        "source_workspace": str(source_workspace),
        "source_matrix_sha256": sha256_file(source_workspace / "matrix.json"),
        "data_receipt_sha256": source["data_receipt_sha256"],
        "data_root": source["data_root"], "cache_root": source["cache_root"], "nvme_root": source["nvme_root"],
        "data_files": source["data_files"], "models": models,
        "parameters": {k: v["downstream_parameters"] for k, v in VARIANTS.items()},
        "input_hashes": {p.name: sha256_file(p) for p in inputs.iterdir() if p.is_file()},
        "training_started": False,
        "p3_upsample_mode": p3_upsample_mode,
    }
    write_json(workspace / "matrix.json", matrix)
    return matrix


def load_matrix(workspace):
    workspace = workspace.resolve()
    matrix = e1.read_json(workspace / "matrix.json")
    if matrix["schema_version"] != SCHEMA or matrix["contract"] != load_contract():
        raise ValueError("P5 contract changed")
    e1.verify_source_identity(matrix["identity"], identity())
    source_workspace = Path(matrix["source_workspace"])
    source = e1.load_matrix(source_workspace, check_code=False)
    if sha256_file(source_workspace / "matrix.json") != matrix["source_matrix_sha256"]:
        raise ValueError("Source E1 matrix changed")
    if matrix["workspace"] != str(workspace):
        raise ValueError("P5 workspace identity differs")
    for key in ("data_root", "cache_root", "nvme_root", "data_receipt_sha256", "data_files"):
        if matrix[key] != source[key]:
            raise ValueError(f"P5 data provenance changed: {key}")
    expected_files = {f"{kind}-{variant}.{suffix}" for variant in VARIANTS
                      for kind, suffix in (("initial", "pt"), ("model", "yaml"))}
    if set(matrix["input_hashes"]) != expected_files:
        raise ValueError("P5 prepared input set changed")
    for name, digest in matrix["input_hashes"].items():
        if sha256_file(workspace / "inputs" / name) != digest:
            raise ValueError(f"P5 prepared input changed: {name}")
    for variant in VARIANTS:
        path = workspace / "inputs" / f"model-{variant}.yaml"
        if matrix["models"][variant] != str(path) or YAML.load(path) != model_config(
            variant, matrix.get("p3_upsample_mode", "bilinear")
        ):
            raise ValueError(f"P5 model registration changed: {variant}")
        if matrix["parameters"][variant] != VARIANTS[variant]["downstream_parameters"]:
            raise ValueError("P5 parameter registration changed")
    return matrix


def spec_for(matrix, variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown P5 variant")
    workspace = Path(matrix["workspace"])
    run_id = f"P5-{variant}"
    return {
        "identity": {
            "source": matrix["identity"], "run_id": run_id, "variant": variant,
            "data_receipt_sha256": matrix["data_receipt_sha256"],
            "initial_sha256": matrix["input_hashes"][f"initial-{variant}.pt"],
            "model_sha256": matrix["input_hashes"][f"model-{variant}.yaml"],
        },
        "workspace": str(workspace), "run_id": run_id, "profile": "E1", "variant": variant,
        "window": 50, "seed": 0, "parameters": matrix["parameters"][variant],
        "output": str(workspace / "reports" / run_id),
        "initial_state": str(workspace / "inputs" / f"initial-{variant}.pt"),
    }


def overrides_for(matrix, spec):
    return {
        **deepcopy(LOCKED_TRAIN), "data": matrix["data_files"]["E1"], "model": matrix["models"][spec["variant"]],
        "workers": 4, "save_period": 5, "device": "0,1,2,3,4,5",
        "project": str(Path(spec["workspace"]) / "runs"), "name": spec["run_id"],
        "exist_ok": True, "verbose": False, "latent_aux_gain": 0.1, "mixture_aux_budget": 3.0,
    }


class P5FrozenTrainer(E1FrozenTrainer):
    """Use unchanged finite-update, AMP, scheduling, telemetry and exact-resume policies."""

    evaluation_module = "scripts.d1.run_p5_ablation"


def train(args):
    if not args.approved:
        raise ValueError("Explicit --approved is required before P5 training or resume")
    if int(os.environ.get("WORLD_SIZE", "0")) != 6 or int(os.environ.get("LOCAL_RANK", "-1")) not in range(6):
        raise ValueError("P5 training requires external six-rank torchrun")
    matrix = load_matrix(args.workspace)
    spec = spec_for(matrix, args.variant)
    output = Path(spec["output"])
    last = args.workspace / "runs" / spec["run_id"] / "weights/last.pt"
    overrides = overrides_for(matrix, spec)
    if args.resume:
        if not last.is_file() or not (output / "resume.pt").is_file():
            raise FileNotFoundError("P5 exact resume needs both last.pt and resume.pt")
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        if state["identity"] != spec["identity"] or state["epoch"] + 1 >= spec["window"]:
            raise ValueError("P5 resume identity differs or screening is already complete")
        overrides["resume"] = str(last)
    elif last.exists() or (output / "resume.pt").exists() or (output / "training-result.json").exists():
        raise FileExistsError("P5 run contains training state; use explicit validated resume")
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(matrix["cache_root"])
    trainer = P5FrozenTrainer(
        overrides=overrides, run_spec=spec,
        feature_caches={"train": cache / "train2017", "val": cache / "val2017"},
        trusted_feature_cache=True, max_open_feature_shards=4, feature_prefetch_factor=1,
        amp_init_scale=16, amp_growth_interval=1000000,
    )
    trainer.train()
    return {"status": "training_returned", "run_id": spec["run_id"]}


def evaluate(args):
    matrix = load_matrix(args.workspace)
    run_ids = {f"P5-{variant}": variant for variant in VARIANTS}
    if args.run_id not in run_ids:
        raise ValueError("Unknown registered P5 run-id")
    variant = run_ids[args.run_id]
    spec = spec_for(matrix, variant)
    model = construct_model(variant, matrix.get("p3_upsample_mode", "bilinear"))
    return e1.evaluate_registered(args, matrix, spec, model, overrides_for(matrix, spec))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "prepare", "train", "evaluate"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--source-workspace", type=Path)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--p3-upsample-mode", choices=("bilinear", "separable_bilinear2x"))
    args = parser.parse_args(argv)
    if args.p3_upsample_mode and args.command not in {"inspect", "prepare"}:
        parser.error("P3 implementation is fixed during prepare; train/evaluate read the registered matrix")
    if args.command == "train" and not args.approved:
        parser.error("Explicit --approved is required; inspection/preparation never authorizes training")
    required = {
        "inspect": (), "prepare": ("workspace", "source_workspace"),
        "train": ("workspace", "variant"), "evaluate": ("workspace", "run_id", "checkpoint", "output"),
    }
    for key in required[args.command]:
        if getattr(args, key) is None:
            parser.error(f"--{key.replace('_', '-')} is required")
    if args.workspace:
        args.workspace = args.workspace.resolve()
    if args.command == "inspect":
        result = inspect(args.p3_upsample_mode or "bilinear")
    elif args.command == "prepare":
        result = prepare(args.source_workspace, args.workspace, args.p3_upsample_mode or "bilinear")
    elif args.command == "train":
        result = train(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Prepare the D1 matched-parameter scratch control without implicit training or downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from copy import copy, deepcopy
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.data.augment import Compose, Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import one_cycle, unwrap_model

CONFIG = ROOT / "ultralytics/cfg/experiments/d1/wp8-p1-scratch-coco2017.yaml"
MODEL = "ultralytics/cfg/models/26/yolo26-d1-scratch-matched-n.yaml"
SCHEMA = "d1-wp8-p1-scratch-v1"
SPLITS = {"train2017": 118287, "val2017": 5000}
PARAMETERS = 3510624
AUGMENTATIONS = (
    "multi_scale", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix", "copy_paste", "erasing",
)
LOCKED_TRAIN = {
    "task": "detect", "epochs": 100, "imgsz": 640, "batch": 384, "nbs": 384, "seed": 0,
    "deterministic": True, "amp": True, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
    "momentum": 0.9, "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1, "cos_lr": True, "patience": 100, "save": True, "save_period": 10,
    "val": True, "plots": False, "pretrained": False, "resume": False, "freeze": None,
    "fraction": 1.0, "cache": False, "compile": False, "foundation_enabled": False,
    "rect": False, "close_mosaic": 0, "box": 7.5, "cls": 0.5, "dfl": 1.5,
    "cls_pw": 0.0, "max_det": 300,
    **dict.fromkeys(AUGMENTATIONS, 0.0),
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomically publish a finite JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_contract(path: Path = CONFIG) -> dict:
    """Fail closed if a registered experimental setting changes."""
    value = YAML.load(path)
    expected = {
        "schema_version": SCHEMA, "model": MODEL, "window_epochs": 30,
        "hardware": {"world_size": 6, "devices": "0,1,2,3,4,5", "per_gpu_batch": 64},
        "runtime": {
            "worker_candidates": [4, 8], "prefetch_factor": 1, "amp_init_scale": 16,
            "amp_growth_interval": 1000000, "gradient_accumulation": 1,
        },
        "train": LOCKED_TRAIN,
    }
    for key, locked in expected.items():
        if value.get(key) != locked:
            raise ValueError(f"Unregistered scratch setting: {key}")
    reference = value.get("reference", {})
    if (reference.get("trainable_parameters"), reference.get("scratch_parameters"),
            reference.get("maximum_parameter_delta")) != (3542567, PARAMETERS, 0.01):
        raise ValueError("The pre-registered parameter budget changed")
    return value


def audit_model(model: DetectionModel) -> dict:
    """Check the real forward architecture, not only the requested width multiplier."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = model.model[-1]
    forbidden = ("teacher", "dinov3", "latentmixture", "foundation")
    if any(any(word in (name + type(module).__name__).lower() for word in forbidden)
           for name, module in model.named_modules()):
        raise ValueError("Scratch must not contain a teacher, foundation adapter or mixture")
    if total != PARAMETERS or trainable != total or abs(total / 3542567 - 1) > 0.01:
        raise ValueError(f"Unexpected parameter budget: {total} total / {trainable} trainable")
    if (head.nc, head.reg_max, head.end2end) != (80, 1, True) or tuple(model.stride.tolist()) != (8, 16, 32):
        raise ValueError("Unexpected detection head")
    return {
        "total_parameters": total, "trainable_parameters": trainable,
        "reference_trainable_parameters": 3542567, "parameter_delta_fraction": total / 3542567 - 1,
        "input": [3, 640, 640], "head_channels": [72, 136, 272], "strides": [8, 16, 32],
        "nc": head.nc, "reg_max": head.reg_max, "end2end": head.end2end, "pretrained": False,
    }


class ScratchDataset(YOLODataset):
    """Use the exact WP0 geometry on original RGB images, with no preliminary resize."""

    prefetch_factor = 1

    def build_transforms(self, hyp=None):
        """Build deterministic transforms for both train and val."""
        return Compose([
            LetterBox((640, 640), auto=False, scale_fill=False, scaleup=True, center=True,
                      stride=32, padding_value=114, interpolation=cv2.INTER_LINEAR),
            Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
        ])

    def load_image(self, i, rect_mode=True, resize_short=False):
        """Decode the original image, bypassing the default pre-resize and image buffer."""
        image = cv2.imread(self.im_files[i], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.im_files[i])
        return image, image.shape[:2], image.shape[:2]

    def __getitem__(self, index):
        """Keep inverse-coordinate metadata aligned with the original-image LetterBox."""
        result = super().__getitem__(index)
        height, width = result["ori_shape"]
        gain = min(640 / height, 640 / width)
        left = round((640 - round(width * gain)) / 2 - 0.1)
        top = round((640 - round(height * gain)) / 2 - 0.1)
        result["ratio_pad"] = ((gain, gain), (left, top))
        return result


def build_dataset(args, data, img_path, batch):
    """Never use rectangular validation, image caching or random augmentation."""
    return ScratchDataset(
        img_path=img_path, data=data, imgsz=640, batch_size=batch, augment=False,
        hyp=args, rect=False, cache=False, stride=32, pad=0.0, task="detect",
    )


class ScratchValidator(DetectionValidator):
    """Use the same fixed geometry in training and independent validation."""

    def build_dataset(self, img_path, mode="val", batch=None):
        """Construct a fixed 640-square RGB validation dataset."""
        return build_dataset(self.args, self.data, img_path, batch or self.args.batch)


class ScratchTrainer(DetectionTrainer):
    """Keep scratch-specific policy outside the shared training engine."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Construct random weights, or strictly reload an explicitly resumed model."""
        if weights is not None and not self.resume:
            raise ValueError("Pretrained weights are forbidden for the scratch control")
        model = DetectionModel(cfg, nc=80, ch=3, verbose=verbose)
        audit_model(model)
        if weights is not None:
            model.load_state_dict(weights.float().state_dict(), strict=True)
        return self.set_model_names_for_load(model)

    def build_dataset(self, img_path, mode="train", batch=None):
        """Keep train and val pixel geometry identical."""
        return build_dataset(self.args, self.data, img_path, batch or self.args.batch)

    def get_validator(self):
        """Preserve standard detection metrics without adding auxiliary losses."""
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return ScratchValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                _callbacks=self.callbacks)

    def check_amp_compatibility(self):
        """Check this model locally instead of downloading a separate AMP-test model."""
        if self.device.type != "cuda":
            return False
        model = unwrap_model(self.model)
        was_training = model.training
        try:
            model.eval()
            sample = torch.linspace(0, 1, 3 * 640 * 640, device=self.device).reshape(1, 3, 640, 640)
            with torch.inference_mode():
                fp32 = model(sample)[0]
                with torch.autocast("cuda", dtype=torch.float16):
                    fp16 = model(sample)[0]
            if fp32.shape != fp16.shape or not torch.isfinite(fp32).all() or not torch.isfinite(fp16).all():
                raise FloatingPointError("Scratch AMP forward is not finite")
        finally:
            model.train(was_training)
        return True

    def _setup_train(self):
        """Match P0's initial loss scale without overwriting a resumed scaler."""
        super()._setup_train()
        if not self.amp:
            raise RuntimeError("AMP is required; do not silently fall back")
        if not self.resume:
            self.scaler = torch.amp.GradScaler("cuda", init_scale=16, growth_interval=1000000)

    def final_eval(self):
        """Keep best/last unstripped for recovery; official evaluation is a separate gate."""
        # Every epoch, including epoch 30, has already run full validation.
        if int(os.environ.get("RANK", "-1")) in (-1, 0):
            reports = {path.name: strict_checkpoint(path) for path in (self.best, self.last)}
            write_json(self.save_dir / "checkpoint-reload.json", reports)


def strict_checkpoint(path: Path) -> dict:
    """Strictly reconstruct a trusted checkpoint produced by this runner."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("ema")
    if saved is None:
        saved = checkpoint.get("model")
    if not isinstance(saved, DetectionModel):
        raise TypeError("Checkpoint must contain a standard DetectionModel")
    restored = DetectionModel(ROOT / MODEL, nc=80, ch=3, verbose=False)
    restored.load_state_dict(saved.float().state_dict(), strict=True)
    audit_model(restored)
    if not all(torch.isfinite(value).all() for value in restored.state_dict().values()):
        raise FloatingPointError("Checkpoint contains non-finite state")
    return {"sha256": sha256_file(path), "epoch": checkpoint["epoch"], "strict_reload": True,
            "parameters": PARAMETERS, "has_optimizer": checkpoint.get("optimizer") is not None}


def stop_at_window(trainer):
    """Stop after validation and checkpoint saving, leaving the 100-epoch schedules intact."""
    if trainer.epoch + 1 >= 30:
        trainer.stop = True


class TrainingTelemetry:
    """Report actual successful updates and finite metrics, separately for each rank."""

    def __init__(self, output: Path):
        self.output = output
        self.rank = max(int(os.environ.get("RANK", "-1")), 0)
        self.started = time.monotonic()

    def on_train_start(self, trainer):
        """Audit effective runtime after AMP setup."""
        model = unwrap_model(trainer.model)
        audit = audit_model(model)
        if trainer.args.epochs != 100 or trainer.batch_size != 384 or trainer.accumulate != 1:
            raise RuntimeError("Effective training settings differ from the registered contract")
        if type(trainer.optimizer).__name__ != "AdamW":
            raise RuntimeError("Effective optimizer is not AdamW")
        write_json(self.output / f"rank-{self.rank}-runtime.json", {
            **audit, "schedule_epochs": trainer.args.epochs, "window_epochs": 30,
            "optimizer": type(trainer.optimizer).__name__, "amp_scale": trainer.scaler.get_scale(),
            "optimizer_groups": [
                {"params": sum(p.numel() for p in group["params"]),
                 "lr": group["lr"], "weight_decay": group["weight_decay"]}
                for group in trainer.optimizer.param_groups
            ],
        })

    def on_train_batch_end(self, trainer):
        """Reject non-finite losses instead of accepting automatic recovery as a fair run."""
        if not torch.isfinite(trainer.loss).all() or not torch.isfinite(trainer.loss_items).all():
            raise FloatingPointError("Non-finite scratch loss")
        if getattr(trainer, "_gradient_nonfinite", False):
            raise FloatingPointError("Non-finite scratch gradients; investigate before changing AMP policy")

    def on_fit_epoch_end(self, trainer):
        """Record the schedule and measured metrics after validation."""
        values = {key: float(value) for key, value in (trainer.metrics or {}).items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite scratch validation metric")
        criterion = unwrap_model(trainer.model).criterion
        write_json(self.output / f"rank-{self.rank}-epoch-{trainer.epoch + 1:03d}.json", {
            "epoch": trainer.epoch + 1, "elapsed_seconds": time.monotonic() - self.started,
            "successful_optimizer_steps": int(trainer.optimizer_steps), "metrics": values,
            "lr": [group["lr"] for group in trainer.optimizer.param_groups],
            "amp_scale": trainer.scaler.get_scale(),
            "next_epoch_o2m": float(criterion.o2m), "next_epoch_o2o": float(criterion.o2o),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device),
        })
        stop_at_window(trainer)


def code_identity() -> dict:
    """Bind preparation to committed code and explicit file digests."""
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
    files = [ROOT / MODEL, CONFIG, Path(__file__).resolve()]
    return {"commit": commit, "dirty": bool(dirty),
            "files": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}}


def read_splits(manifest_dir: Path) -> tuple[dict, dict]:
    """Validate canonical lists and their WP0 checksums without touching COCO images."""
    manifest = json.loads((manifest_dir / "coco2017-splits.json").read_text())
    paths, evidence = {}, {}
    for split, count in SPLITS.items():
        entry = manifest["splits"][split]
        path = manifest_dir / entry["path"]
        names = path.read_text().splitlines()
        if sha256_file(path) != entry["sha256"] or entry["count"] != count or len(names) != count:
            raise ValueError(f"Invalid {split} manifest or checksum")
        if names != sorted(set(names)):
            raise ValueError(f"Invalid {split} ordering/duplicates")
        for name in names:
            if Path(name).parts != ("images", split, Path(name).name) or not name.endswith(".jpg"):
                raise ValueError(f"Unsafe split path: {name}")
        paths[split] = names
        evidence[split] = {"samples": count, "list_sha256": entry["sha256"],
                           "label_files": manifest["labels"][split]}
    if {Path(p).stem for p in paths["train2017"]} & {Path(p).stem for p in paths["val2017"]}:
        raise ValueError("COCO train and val image IDs overlap")
    return paths, evidence


def copy_ready(receipt: Path, data_root: Path) -> dict:
    """Require the atomic publication and complete copy-verification receipt."""
    state = json.loads(receipt.read_text())
    if (state.get("phase") != "COMPLETED" or state.get("mode") != "copy_only"
            or not state.get("preserve_source") or Path(state.get("destination", "")).resolve() != data_root.resolve()):
        raise ValueError("The requested verified COCO copy has not been published")
    if state.get("images") != SPLITS:
        raise ValueError("Copy receipt has incorrect split counts")
    checksum = Path(state["checksum_manifest"])
    if sha256_file(checksum) != state["checksum_manifest_sha256"]:
        raise ValueError("Copy checksum manifest changed")
    return {"checksum_manifest_sha256": state["checksum_manifest_sha256"], "phase": state["phase"]}


def training_overrides(contract: dict, data_yaml: Path, run_root: Path, workers: int) -> dict:
    """Keep runtime paths outside tracked configuration."""
    if type(workers) is not int or workers not in contract["runtime"]["worker_candidates"]:
        raise ValueError("workers must be a benchmark candidate: 4 or 8")
    return {**contract["train"], "model": str(ROOT / MODEL), "data": str(data_yaml),
            "device": contract["hardware"]["devices"], "workers": workers,
            "project": str(run_root.parent), "name": run_root.name, "exist_ok": True, "verbose": False}


def prepare(args) -> dict:
    """Inspect the model, optionally verify local data; never start training."""
    contract = load_contract()
    torch.manual_seed(0)
    model = DetectionModel(ROOT / MODEL, nc=80, ch=3, verbose=False)
    paths, evidence = read_splits(ROOT / "experiments/d1/manifests")
    report = {
        "schema_version": SCHEMA, "status": "model_ready_runtime_pending",
        "identity": code_identity(), "model": audit_model(model), "splits": evidence,
        "schedule_epochs": 100, "window_epochs": 30,
        "base_lr_at_epoch30": 0.001 * one_cycle(1, 0.01, 100)(29),
        "formal_training_approved": False,
        "pending": ["six_gpu_worker_benchmark", "real_batch_backward", "official_evaluation_gate", "user_approval"],
    }
    if not args.model_only:
        if args.data_root is None or args.copy_receipt is None:
            raise ValueError("Full prepare requires --data-root and --copy-receipt")
        data_root = args.data_root.resolve()
        report["copy"] = copy_ready(args.copy_receipt, data_root)
        for split, names in paths.items():
            actual = sorted(p.relative_to(data_root).as_posix() for p in (data_root / "images" / split).glob("*.jpg"))
            if actual != names:
                raise ValueError(f"Published {split} image set differs from WP0")
            labels = list((data_root / "labels" / split).glob("*.txt"))
            if len(labels) != evidence[split]["label_files"]:
                raise ValueError(f"Published {split} label count differs from WP0")
            valid_ids = {Path(name).stem for name in names}
            if any(path.stem not in valid_ids for path in labels):
                raise ValueError(f"Unexpected {split} labels")
        annotation = data_root / "annotations/instances_val2017.json"
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        inputs = args.output_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        for split, names in paths.items():
            (inputs / f"{split}.txt").write_text(
                "".join(str(data_root / name) + "\n" for name in names), encoding="utf-8")
        data_yaml = inputs / "coco2017.yaml"
        YAML.save(data_yaml, {"path": str(data_root), "train": str(inputs / "train2017.txt"),
                             "val": str(inputs / "val2017.txt"),
                             "names": YAML.load(ROOT / "ultralytics/cfg/datasets/coco.yaml")["names"]})
        report["runtime_inputs"] = {path.name: sha256_file(path) for path in inputs.iterdir() if path.is_file()}
        report["data_root"] = str(data_root)
        report["status"] = "data_ready_runtime_pending"
        report["overrides"] = training_overrides(contract, data_yaml, args.run_root, args.workers)
    write_json(args.output_dir / "preparation.json", report)
    return report


def require_launch_approval(approved: bool, preparation: dict, gate: dict, identity: dict) -> None:
    """Training requires separate, version-bound runtime evidence and user approval."""
    if not approved:
        raise RuntimeError("Formal training needs explicit user approval and --approved")
    if identity["dirty"] or preparation.get("identity") != identity or gate.get("identity") != identity:
        raise RuntimeError("Training requires a clean, unchanged code commit; repeat preparation after committing")
    required = ("six_gpu_worker_benchmark", "real_batch_backward", "strict_checkpoint_reload", "evaluation_ready")
    if preparation.get("status") != "data_ready_runtime_pending" or gate.get("status") != "passed":
        raise RuntimeError("Data/runtime gate has not passed")
    if any(gate.get(name) is not True for name in required):
        raise RuntimeError("Missing real runtime acceptance evidence")
    if gate.get("preparation_sha256") != preparation.get("_file_sha256"):
        raise RuntimeError("Runtime gate belongs to another preparation report")
    if gate.get("workers") != preparation["overrides"]["workers"]:
        raise RuntimeError("Runtime gate selected different workers")


def claim_run(run_root: Path, identity: dict, token: str, rank: int, timeout: float = 30) -> None:
    """Claim a fresh output directory before any rank can construct a Trainer."""
    if not token or token == "none":
        raise RuntimeError("Use torchrun --standalone to get a fresh rendezvous identity")
    marker = run_root / "scratch-launch.json"
    expected = {"identity": identity, "rendezvous_id": token}
    if rank == 0:
        run_root.parent.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(exist_ok=False)
        write_json(marker, expected)
        return
    deadline = time.monotonic() + timeout
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Rank zero did not publish the fresh run claim")
        time.sleep(0.1)
    if json.loads(marker.read_text()) != expected:
        raise RuntimeError("Output directory belongs to a different launch")


def train(args) -> dict:
    """Launch only the approved six-rank control; never auto-launch DDP or download models."""
    contract = load_contract()
    path = args.output_dir / "preparation.json"
    preparation = json.loads(path.read_text())
    preparation["_file_sha256"] = sha256_file(path)
    gate = json.loads(args.runtime_gate.read_text()) if args.runtime_gate else {}
    require_launch_approval(args.approved, preparation, gate, code_identity())
    if int(os.environ.get("WORLD_SIZE", "0")) != 6 or int(os.environ.get("LOCAL_RANK", "-1")) not in range(6):
        raise RuntimeError("Use torchrun with six ranks")
    if args.run_root.resolve() != Path(preparation["overrides"]["project"]) / preparation["overrides"]["name"]:
        raise RuntimeError("Run root differs from preparation")
    for name, digest in preparation["runtime_inputs"].items():
        if sha256_file(args.output_dir / "inputs" / name) != digest:
            raise RuntimeError("Runtime dataset inputs changed")
    overrides = deepcopy(preparation["overrides"])
    if overrides != training_overrides(contract, args.output_dir / "inputs/coco2017.yaml", args.run_root, gate["workers"]):
        raise RuntimeError("Prepared training arguments changed")
    claim_run(args.run_root, code_identity(), os.environ.get("TORCHELASTIC_RUN_ID", ""),
              int(os.environ.get("RANK", "-1")))
    trainer = ScratchTrainer(overrides=overrides)
    telemetry = TrainingTelemetry(args.output_dir / "epochs")
    for event in ("on_train_start", "on_train_batch_end", "on_fit_epoch_end"):
        trainer.add_callback(event, getattr(telemetry, event))
    trainer.train()
    if trainer.epoch != 29:
        raise RuntimeError(f"Unexpected stopping epoch: {trainer.epoch}")
    return {"status": "training_completed_evaluation_pending", "epochs": 30}


def main():
    """Expose preparation independently from the explicitly gated training command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--copy-receipt", type=Path)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=(4, 8), default=4)
    parser.add_argument("--runtime-gate", type=Path)
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.run_root = args.run_root.resolve()
    if ROOT == args.output_dir or ROOT in args.output_dir.parents or ROOT == args.run_root or ROOT in args.run_root.parents:
        raise ValueError("Runtime outputs must remain outside the Git repository")
    result = prepare(args) if args.command == "prepare" else train(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

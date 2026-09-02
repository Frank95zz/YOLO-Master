"""Training integration for the D1 cached-feature detector."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.d1_cache import D1FeatureCacheDataset, FeatureProvider, move_d1_batch_to_device
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.foundation_val import D1FoundationDetectionValidator
from ultralytics.nn.foundation_detection_model import (
    DEFAULT_D1_MODEL_CFG,
    D1FoundationDetectionModel,
)
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first


_DISABLED_AUGMENTATIONS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "erasing",
)


def _cache_paths(value: Mapping[str, str | Path]) -> dict[str, Path]:
    if not isinstance(value, Mapping):
        raise TypeError("feature_caches must map 'train' and 'val' to cache directories.")
    if set(value) != {"train", "val"}:
        raise ValueError("feature_caches must contain exactly the 'train' and 'val' keys.")
    paths = {name: Path(path).expanduser() for name, path in value.items()}
    missing = [f"{name}={path}" for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("D1 feature cache directory does not exist: " + ", ".join(missing))
    return paths


class D1FoundationDetectionTrainer(DetectionTrainer):
    """Train the WP4 downstream model from WP2 feature caches."""

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        *,
        feature_caches: Mapping[str, str | Path],
        validation_feature_mode: str = "cache",
        online_feature_provider: FeatureProvider | None = None,
    ) -> None:
        self.feature_caches = _cache_paths(feature_caches)
        if validation_feature_mode not in {"cache", "online"}:
            raise ValueError("validation_feature_mode must be 'cache' or 'online'.")
        if validation_feature_mode == "online" and not callable(online_feature_provider):
            raise ValueError("online validation requires an online_feature_provider callable.")
        self.validation_feature_mode = validation_feature_mode
        self.online_feature_provider = online_feature_provider
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._validate_d1_args()

    def _validate_d1_args(self) -> None:
        if self.args.imgsz != 640:
            raise ValueError(f"D1 cached training requires imgsz=640, got {self.args.imgsz}.")
        if self.args.rect:
            raise ValueError("D1 cached training requires rect=False.")
        if self.args.multi_scale != 0.0:
            raise ValueError("D1 cached training requires multi_scale=0.0.")
        if self.args.compile:
            raise ValueError("D1 WP5 does not support compile mode.")
        if self.args.batch == -1 or isinstance(self.args.batch, float):
            raise ValueError("D1 cached training requires an explicit integer batch size.")
        if self.args.cache:
            raise ValueError("D1 uses feature_caches; the RGB cache argument must remain disabled.")
        enabled = [name for name in _DISABLED_AUGMENTATIONS if float(getattr(self.args, name, 0.0)) != 0.0]
        if enabled:
            raise ValueError(f"D1 cached training requires disabled RGB/geometric augmentations: {enabled}.")
        if self.validation_feature_mode == "online" and self.args.workers != 0:
            raise ValueError("online parity validation requires workers=0 to keep the Teacher in the main process.")

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build a COCO-label dataset backed by the selected split cache."""
        if mode not in {"train", "val"}:
            raise ValueError(f"unsupported D1 dataset mode {mode!r}.")
        feature_mode = "cache" if mode == "train" else self.validation_feature_mode
        return D1FeatureCacheDataset(
            img_path=img_path,
            cache_dir=self.feature_caches[mode],
            data=self.data,
            feature_mode=feature_mode,
            online_feature_provider=self.online_feature_provider if mode == "val" else None,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            hyp=self.args,
            prefix=colorstr(f"{mode}: "),
            single_cls=self.args.single_cls or False,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
            stride=32,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move cached features and labels to the device without RGB ``/255`` normalization."""
        use_fp16 = self.device.type == "cuda" and bool(getattr(self, "amp", False))
        return move_d1_batch_to_device(
            batch,
            self.device,
            feature_dtype=torch.float16 if use_fp16 else torch.float32,
        )

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True):
        """Construct the WP4 downstream-only model and optionally restore a standard trainer checkpoint model."""
        model = D1FoundationDetectionModel(cfg or DEFAULT_D1_MODEL_CFG, verbose=verbose and RANK == -1)
        if model.detect.nc != self.data["nc"]:
            raise ValueError(f"D1 model nc={model.detect.nc} does not match dataset nc={self.data['nc']}.")
        model = self.set_model_names_for_load(model)
        if weights is not None:
            if not isinstance(weights, D1FoundationDetectionModel):
                raise TypeError("D1 trainer weights must contain a D1FoundationDetectionModel.")
            model.load_state_dict(weights.state_dict(), strict=True)
        return model

    def get_validator(self):
        """Return the cache-aware D1 detection validator."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return D1FoundationDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            feature_cache=self.feature_caches["val"],
            feature_mode=self.validation_feature_mode,
            online_feature_provider=self.online_feature_provider,
        )

    def auto_batch(self, max_num_obj: int = 0, dataset_size: int = 0):
        """Reject RGB-based automatic batch probing for cached-feature training."""
        raise RuntimeError("D1 cached training requires an explicit batch size; auto_batch uses RGB probes.")

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Feature batches intentionally have no RGB pixels to plot."""
        return None

    def checkpoint_smoke_inputs(self, model: D1FoundationDetectionModel) -> tuple[dict[str, torch.Tensor], ...]:
        """Build deterministic cached-feature inputs for recovery checkpoint health checks."""
        first = next(model.parameters(), None)
        device = first.device if first is not None else torch.device("cpu")
        zeros = {
            name: torch.zeros(1, 384, 40, 40, dtype=torch.float32, device=device)
            for name in model.source_names
        }
        ramp = torch.linspace(-1.0, 1.0, 384 * 40 * 40, dtype=torch.float32, device=device).reshape(1, 384, 40, 40)
        return (
            zeros,
            {name: ramp.clone() for name in model.source_names},
        )

    def final_eval(self) -> None:
        """Load the selected D1 checkpoint directly and evaluate it through cached features."""
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                checkpoint = strip_optimizer(self.last) if self.last.exists() else {}
                if self.best.exists():
                    strip_optimizer(self.best, updates={"train_results": checkpoint.get("train_results")})
        candidates, rejected = self._select_final_eval_checkpoints()
        if not candidates:
            raise RuntimeError(
                "No healthy checkpoint is available for final evaluation: " + ("; ".join(rejected) or "none exist")
            )
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.validator.args.half = False
        from ultralytics.utils.errors import MoERouterError

        router_failures = []
        original_model = self.model
        original_ema = self.ema.ema if self.ema else None
        try:
            for checkpoint_path in candidates:
                self._reset_non_checkpoint_moe_runtime_state()
                LOGGER.info(f"\nValidating {checkpoint_path} through D1 cached features...")
                try:
                    checkpoint_model, _checkpoint = load_checkpoint(checkpoint_path, device=self.device)
                    checkpoint_model.args = self.args
                    checkpoint_model.criterion = None
                    if not isinstance(checkpoint_model, D1FoundationDetectionModel):
                        raise TypeError("D1 final-eval checkpoint does not contain a D1FoundationDetectionModel.")
                    self.model = checkpoint_model
                    if self.ema:
                        self.ema.ema = None
                    self.metrics = self.validator(trainer=self)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
                    return
                except MoERouterError as exc:
                    router_failures.append(f"{checkpoint_path.name}: {exc}")
        finally:
            self.model = original_model
            if self.ema:
                self.ema.ema = original_ema
        raise RuntimeError(
            "No healthy checkpoint is available for final evaluation: " + "; ".join((*rejected, *router_failures))
        )


__all__ = ["D1FoundationDetectionTrainer"]

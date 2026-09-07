"""Shared, bounded E0/E1 trainer policy without modifying historical runners."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.d1.run_wp8_p1_control import ScratchTrainer, sha256_file, write_json
from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.utils.torch_utils import unwrap_model


def atomic_torch(path: Path, value) -> None:
    """Commit a checkpoint without replacing it with a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state(device) -> dict:
    """Preserve the main process RNG; deterministic workers are seeded by the loader."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state, device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def detached_state(module) -> dict:
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        for key, value in module.state_dict().items()
    }


def load_initial_tensors(model, tensors):
    """Share numerical initial state without overwriting the variant's routing configuration."""
    state = model.state_dict()
    expected = {key for key, value in state.items() if isinstance(value, torch.Tensor)}
    if set(tensors) != expected or not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise ValueError("Initial tensor key set changed")
    state.update(tensors)
    model.load_state_dict(state, strict=True)


def clean_child_env() -> dict:
    """A single-GPU evaluator must not inherit its parent's torchrun rank."""
    env = os.environ.copy()
    for key in list(env):
        if key in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        } or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def separated_gradients(native, auxiliary, parameters):
    """Verify additive diagnostic gradients without populating parameter.grad."""
    parameters = tuple(parameters)
    detection = torch.autograd.grad(native, parameters, retain_graph=True, allow_unused=True)
    aux = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
    total = torch.autograd.grad(native + auxiliary, parameters, allow_unused=True)
    rows = []
    for parameter, first, second, combined in zip(parameters, detection, aux, total):
        zero = torch.zeros_like(parameter)
        first, second, combined = (zero if value is None else value for value in (first, second, combined))
        if not all(torch.isfinite(value).all() for value in (first, second, combined)):
            raise FloatingPointError("Non-finite mechanism gradient")
        if not torch.allclose(first + second, combined, rtol=2e-4, atol=2e-5):
            raise ValueError("Diagnostic decomposition changed the total gradient")
        rows.append(
            {
                "detection": float(first.float().norm()),
                "aux": float(second.float().norm()),
                "total": float(combined.float().norm()),
            }
        )
    return rows


def mechanism_evidence(source, batch):
    """Probe an independent model; never update the trained model or its aux EMA."""
    from ultralytics.nn.mixture_loss import _collect_mixture_aux_loss

    model = D1FoundationDetectionModel(source.config_dict(), verbose=False)
    initialize_mixture_loss_ema_buffer(model)
    model.load_state_dict(source.state_dict(), strict=True)
    model.args = deepcopy(source.args)
    model.to(next(source.parameters()).device).train()
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(key in name for key in ("branches.", "router", "residual_gain"))
    ]
    amplitudes = {}

    def magnitude(value):
        return {
            "rms": float(value.detach().float().square().mean().sqrt()),
            "abs_max": float(value.detach().abs().max()),
        }

    def adapter_hook(module, inputs, outputs):
        amplitudes["candidates"] = {
            scale: [magnitude(tensor) for tensor in values] for scale, values in outputs.items()
        }

    hooks = [model.adapter.register_forward_hook(adapter_hook)]
    for scale, mixture in model.mixtures.items():

        def mixture_hook(module, inputs, output, scale=scale):
            amplitudes.setdefault("fused", {})[scale] = magnitude(output)

        hooks.append(mixture.register_forward_hook(mixture_hook))
    initial = detached_state(model)
    result = {}
    try:
        for case, balance, z_loss, gain in (
            ("active", 0.01, 0.001, float(source.args.latent_aux_gain)),
            ("aux_zero", 0.01, 0.001, 0.0),
            ("balance_only", 0.01, 0.0, 0.1),
            ("z_only", 0.0, 0.001, 0.1),
        ):
            model.load_state_dict(initial, strict=True)
            for mixture in model.mixtures.values():
                mixture.balance_loss_coeff, mixture.router_z_loss_coeff = balance, z_loss
            criterion = model.init_criterion().native_criterion
            source_criterion = getattr(source.criterion, "native_criterion", source.criterion)
            for name in ("updates", "o2m", "o2o"):
                if hasattr(source_criterion, name):
                    setattr(criterion, name, getattr(source_criterion, name))
            predictions = model.predict(batch["features"])
            native, _ = criterion(predictions, batch)
            auxiliary = _collect_mixture_aux_loss(
                model,
                native.device,
                moe_gain=0.0,
                mot_gain=0.0,
                moa_gain=0.0,
                latent_gain=gain,
                aux_budget=3.0,
            )
            gradients = separated_gradients(native.sum(), auxiliary, (parameter for _, parameter in selected))
            if case == "aux_zero" and any(row["aux"] != 0.0 for row in gradients):
                raise ValueError("Disabled aux has a nonzero gradient")
            result[case] = {
                "native_loss": float(native.detach().sum()),
                "applied_aux": float(auxiliary.detach()),
                "aux_requires_grad": auxiliary.requires_grad,
                "gradients": dict(zip((name for name, _ in selected), gradients)),
                "amplitudes": deepcopy(amplitudes),
            }
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Diagnostics unexpectedly populated .grad")
    finally:
        for hook in hooks:
            hook.remove()
    return result


class E1Policy:
    """Require finite updates, preserve exact resume state, and stop at a window boundary."""

    def __init__(self, *args, run_spec, **kwargs):
        self.e1 = run_spec
        self.e1_rank = int(os.environ.get("RANK", "0"))
        self.e1_output = Path(run_spec["output"])
        self.e1_stop_requested = False
        self.e1_epoch_batches = 0
        self.e1_previous_end = None
        self.e1_wait = 0.0
        self.e1_compute = 0.0
        self.e1_retries = 0
        super().__init__(*args, **kwargs)
        for event in (
            "on_train_start",
            "on_train_epoch_start",
            "on_train_batch_start",
            "on_train_batch_end",
            "on_train_epoch_end",
            "on_fit_epoch_end",
        ):
            self.add_callback(event, getattr(self, "e1_" + event))
        signal.signal(signal.SIGUSR1, lambda *_: setattr(self, "e1_stop_requested", True))

    def get_model(self, cfg=None, weights=None, verbose=True):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.e1["seed"])
            model = super().get_model(
                cfg=cfg,
                weights=None if isinstance(weights, D1FoundationDetectionModel) else weights,
                verbose=verbose,
            )
        if isinstance(model, D1FoundationDetectionModel):
            initialize_mixture_loss_ema_buffer(model)
            if weights is not None:
                model.load_state_dict(weights.state_dict(), strict=True)
        if not self.resume:
            state = torch.load(self.e1["initial_state"], map_location="cpu", weights_only=True)
            load_initial_tensors(model, state)
        return model

    def _setup_train(self):
        super()._setup_train()
        if not self.amp or self.accumulate != 1 or self.args.epochs != 100:
            raise RuntimeError("E1 requires AMP, one update per batch, and the original 100-epoch schedule")
        if self.resume:
            state = torch.load(self.e1_output / "resume.pt", map_location="cpu", weights_only=False)
            if state["identity"] != self.e1["identity"] or state["epoch"] + 1 != self.start_epoch:
                raise ValueError("Resume identity or epoch does not match")
            model = unwrap_model(self.model)
            model.load_state_dict(state["model"], strict=True)
            self.ema.ema.load_state_dict(state["ema"], strict=True)
            self.ema.updates = state["ema_updates"]
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.optimizer_steps = state["optimizer_steps"]
            model.criterion = model.init_criterion()
            native = getattr(model.criterion, "native_criterion", model.criterion)
            for name, value in state["criterion"].items():
                setattr(native, name, value)
            rank_state = state["ranks"][self.e1_rank]
            restore_rng(rank_state["rng"], self.device)
            generator = getattr(self.train_loader, "generator", None)
            if generator is not None and rank_state["loader_generator"] is not None:
                generator.set_state(rank_state["loader_generator"])

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        model = unwrap_model(self.model)
        self.e1_retry_batch = batch
        self.e1_retry_buffers = {k: v.clone() for k, v in model.named_buffers()}
        self.e1_retry_rng = rng_state(self.device)
        return batch

    def optimizer_step(self):
        for attempt in range(9):
            model = unwrap_model(self.model)
            bad = not bool(torch.isfinite(self.loss).all())
            bad |= any(getattr(model, "_mixture_aux_nonfinite", {}).values())
            if self._sync_nonfinite_flag(bad):
                raise FloatingPointError("Non-finite detection/aux loss; E1 cannot silently isolate it")
            scale = float(self.scaler.get_scale())
            self._gradient_nonfinite = False
            if self.e1_epoch_batches == 0:
                gradients = {
                    name: float(parameter.grad.detach().float().norm()) / scale
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and any(key in name for key in ("branches.", "router", "residual_gain"))
                }
                if all(math.isfinite(value) for value in gradients.values()):
                    write_json(
                        self.e1_output / "gradients" / f"rank-{self.e1_rank}-epoch-{self.epoch + 1:03d}.json",
                        {
                            "epoch": self.epoch + 1,
                            "total_loss_gradient_norms": gradients,
                            "scope": "Full detection-plus-aux gradient before clipping and optimizer update",
                        },
                    )
            # Skip ScratchTrainer's old retry loop: the same policy applies to both methods.
            from ultralytics.engine.trainer import BaseTrainer

            if BaseTrainer.optimizer_step(self):
                self.e1_retry_batch = self.e1_retry_buffers = self.e1_retry_rng = None
                return True
            if attempt == 8 or not math.isfinite(scale) or scale <= 0:
                raise FloatingPointError("AMP retry budget exhausted; no skipped optimizer updates are allowed")
            self.scaler.update(new_scale=scale / 2)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(self.e1_retry_buffers[name])
            restore_rng(self.e1_retry_rng, self.device)
            self.e1_retries += 1
            with torch.autocast("cuda", dtype=torch.float16):
                loss, self.loss_items = self.model(self.e1_retry_batch)
                self.loss = loss.sum() * self.world_size
            self.scaler.scale(self.loss).backward()
        raise AssertionError("Unreachable optimizer retry state")

    def _handle_nan_recovery(self, epoch):
        if self._sync_nonfinite_flag(not bool(torch.isfinite(self.tloss).all())):
            raise FloatingPointError("Non-finite epoch; refusing automatic replay as a fair comparison")
        return False

    def _recover_before_validation(self, epoch):
        if any(self._collect_prevalidation_nonfinite_flags().values()):
            raise FloatingPointError("Non-finite model/EMA before validation")
        return False

    def e1_on_train_start(self, trainer):
        model = unwrap_model(self.model)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != self.e1["parameters"]:
            raise ValueError(f"Parameter count changed: {count}")
        if self.batch_size != 384 or self.args.workers != 4 or type(self.optimizer).__name__ != "AdamW":
            raise ValueError("Resolved batch/workers/optimizer differs from E1")
        if any("teacher" in name.lower() or "dinov3" in name.lower() for name, _ in model.named_parameters()):
            raise ValueError("A Teacher unexpectedly entered the optimizer model")
        write_json(
            self.e1_output / f"runtime-rank-{self.e1_rank}.json",
            {
                "parameters": count,
                "schedule_epochs": self.args.epochs,
                "window_epochs": self.e1["window"],
                "global_batch": self.batch_size,
                "rank_batch": self.batch_size // self.world_size,
                "accumulate": self.accumulate,
                "amp_scale": self.scaler.get_scale(),
                "world_size": self.world_size,
                "start_epoch": self.start_epoch,
                "samples": len(self.train_loader.dataset),
                "batches_per_epoch": len(self.train_loader),
                "optimizer_groups": [
                    {"lr": g["lr"], "weight_decay": g["weight_decay"]} for g in self.optimizer.param_groups
                ],
            },
        )

    def e1_on_train_epoch_start(self, trainer):
        model = unwrap_model(self.model)
        if (
            isinstance(model, D1FoundationDetectionModel)
            and self.e1["profile"] != "benchmark"
            and (self.epoch == 0 or (self.e1["profile"] == "E1" and self.epoch in (24, 49)))
        ):
            saved_rng = rng_state(self.device)
            try:
                dataset = self.train_loader.dataset
                batch = dataset.collate_fn([dataset[(2 * self.e1_rank + i) % len(dataset)] for i in range(2)])
                batch = self.preprocess_batch(batch)
                report = mechanism_evidence(model, batch)
                write_json(
                    self.e1_output / "mechanism" / f"rank-{self.e1_rank}-epoch-{self.epoch + 1:03d}.json",
                    {
                        "epoch": self.epoch + 1,
                        "scope": "Independent FP32 copy on two real samples per rank",
                        "cases": report,
                    },
                )
            finally:
                restore_rng(saved_rng, self.device)
                self.e1_retry_batch = self.e1_retry_buffers = self.e1_retry_rng = None
        self.e1_epoch_started = time.monotonic()
        self.e1_epoch_batches = 0
        self.e1_previous_end = None
        self.e1_wait = self.e1_compute = 0.0
        self.e1_start_updates = self.optimizer_steps
        torch.cuda.reset_peak_memory_stats(self.device)

    def e1_on_train_batch_start(self, trainer):
        now = time.monotonic()
        self.e1_wait += now - (self.e1_previous_end or self.e1_epoch_started)
        self.e1_batch_started = now

    def e1_on_train_batch_end(self, trainer):
        self.e1_epoch_batches += 1
        if self.accumulate != 1 or self.optimizer_steps - self.e1_start_updates != self.e1_epoch_batches:
            raise RuntimeError("Missing/repeated optimizer update, OOM batch fallback, or gradient accumulation")
        if not torch.isfinite(self.loss_items).all():
            raise FloatingPointError("Non-finite training metrics")
        now = time.monotonic()
        self.e1_compute += now - self.e1_batch_started
        self.e1_previous_end = now
        if self.e1_epoch_batches == 1 or self.e1_epoch_batches % 10 == 0:
            model = unwrap_model(self.model)
            metrics = dict(getattr(model, "_last_latent_aux_metrics", {}))
            ema = getattr(model, "_mixture_loss_ema_buf", None)
            write_json(
                self.e1_output / f"progress-rank-{self.e1_rank}.json",
                {
                    "status": "running",
                    "epoch": self.epoch + 1,
                    "batch": self.e1_epoch_batches,
                    "optimizer_steps": self.optimizer_steps,
                    "loss": self.loss_items.float().cpu().tolist(),
                    "aux": metrics,
                    "amp_scale": self.scaler.get_scale(),
                    "amp_retries": self.e1_retries,
                    "aux_ema": ema.detach().cpu().tolist() if isinstance(ema, torch.Tensor) else None,
                    "updated_unix": time.time(),
                },
            )

    def e1_on_train_epoch_end(self, trainer):
        if self.e1_epoch_batches != len(self.train_loader):
            raise RuntimeError("Epoch did not consume the entire sampler")
        model = unwrap_model(self.model)
        routing = {key: value.routing_snapshot() for key, value in getattr(model, "mixtures", {}).items()}

        # Routing snapshots contain scalar/list telemetry, never graph-connected tensors.
        def plain(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(v) for v in value]
            return value

        write_json(
            self.e1_output / "epochs" / f"rank-{self.e1_rank}-epoch-{self.epoch + 1:03d}.json",
            {
                "epoch": self.epoch + 1,
                "rank": self.e1_rank,
                "batches": self.e1_epoch_batches,
                "data_wait_seconds": self.e1_wait,
                "step_seconds": self.e1_compute,
                "optimizer_steps": self.optimizer_steps,
                "amp_retries": self.e1_retries,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "routing": plain(routing),
            },
        )

    def e1_on_fit_epoch_end(self, trainer):
        epoch = self.epoch + 1
        generator = getattr(self.train_loader, "generator", None)
        local = {
            "rng": rng_state(self.device),
            "loader_generator": generator.get_state() if generator is not None else None,
        }
        ranks = [None] * self.world_size
        dist.all_gather_object(ranks, local)
        error = None
        if self.e1_rank == 0:
            try:
                model = unwrap_model(self.model)
                criterion = getattr(model.criterion, "native_criterion", model.criterion)
                state = {
                    "identity": self.e1["identity"],
                    "epoch": self.epoch,
                    "model": detached_state(model),
                    "ema": detached_state(self.ema.ema),
                    "optimizer": deepcopy(self.optimizer.state_dict()),
                    "scaler": self.scaler.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "ema_updates": self.ema.updates,
                    "optimizer_steps": self.optimizer_steps,
                    "ranks": ranks,
                    "criterion": {k: getattr(criterion, k) for k in ("updates", "o2m", "o2o") if hasattr(criterion, k)},
                }
                atomic_torch(self.e1_output / "resume.pt", state)
                values = {k: float(v) for k, v in self.metrics.items()}
                if not all(math.isfinite(v) for v in values.values()):
                    raise FloatingPointError("Non-finite validation metrics")
                write_json(
                    self.e1_output / "validation" / f"epoch-{epoch:03d}.json",
                    {
                        "epoch": epoch,
                        "metrics": values,
                        "seen": self.validator.seen,
                        "epoch_wall_seconds": time.monotonic() - self.e1_epoch_started,
                        "updated_unix": time.time(),
                    },
                )
                if epoch % 5 == 0 and self.e1["profile"] == "E1":
                    self.e1_standard_evaluation(epoch)
                if epoch >= self.e1["window"] or self.e1_stop_requested:
                    self.stop = True
            except Exception as exc:  # noqa: BLE001 - broadcast failures so other DDP ranks do not hang
                error = repr(exc)
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise RuntimeError(errors[0])

    def e1_standard_evaluation(self, epoch):
        output = self.e1_output / "official" / f"epoch-{epoch:03d}"
        output.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.d1.run_p1p2",
            "evaluate",
            "--workspace",
            self.e1["workspace"],
            "--run-id",
            self.e1["run_id"],
            "--checkpoint",
            str(self.last),
            "--output",
            str(output),
        ]
        started = time.monotonic()
        with (output / "evaluation.log").open("w") as log:
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=clean_child_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        report = json.loads((output / "report.json").read_text())
        best_file = self.e1_output / "standard-best.json"
        previous = json.loads(best_file.read_text()) if best_file.exists() else None
        if previous is None or report["official"]["AP_all"] > previous["AP"]:
            import shutil

            destination = self.save_dir / "weights" / "standard-best.pt"
            temporary = destination.with_suffix(".part")
            shutil.copyfile(self.last, temporary)
            os.replace(temporary, destination)
            write_json(
                best_file, {"epoch": epoch, "AP": report["official"]["AP_all"], "sha256": sha256_file(destination)}
            )
        write_json(output / "cost.json", {"seconds": time.monotonic() - started, "reserved_gpus": self.world_size})

    def final_eval(self):
        """Keep optimizer-bearing checkpoints; supervisor performs isolated final evaluation."""
        if self.e1_rank == 0:
            write_json(
                self.e1_output / "training-result.json",
                {
                    "status": "completed" if self.epoch + 1 == self.e1["window"] else "interrupted",
                    "epochs": self.epoch + 1,
                    "optimizer_steps": self.optimizer_steps,
                    "last_sha256": sha256_file(self.last),
                    "resume_sha256": sha256_file(self.e1_output / "resume.pt"),
                },
            )


class E1FrozenTrainer(E1Policy, D1FoundationDetectionTrainer):
    """Cached DINO features with the shared E1 contract."""


class E1ScratchTrainer(E1Policy, ScratchTrainer):
    """Original RGB images with the same E1 optimizer/stop/recovery policy."""

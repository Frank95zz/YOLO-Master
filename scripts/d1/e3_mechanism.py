"""Actual-coefficient, per-epoch auxiliary probes on an independent FP32 model."""

from copy import deepcopy

import torch

from scripts.d1.p1p2_runtime import diagnostic_precision, separated_gradients
from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.mixture_loss import _collect_mixture_aux_loss, initialize_mixture_loss_ema_buffer


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def probe(source, batch):
    """Caller preserves RNG; no mutation of the source model, optimizer, or aux EMA."""
    device = next(source.parameters()).device
    with diagnostic_precision(device.type):
        model = D1FoundationDetectionModel(source.config_dict(), verbose=False)
        initialize_mixture_loss_ema_buffer(model)
        model.load_state_dict(source.state_dict(), strict=True)
        model.args = deepcopy(source.args)
        model.to(device).train()
        native = model.init_criterion().native_criterion
        original = getattr(source.criterion, "native_criterion", source.criterion)
        for key in ("updates", "o2m", "o2o"):
            if hasattr(original, key):
                setattr(native, key, getattr(original, key))
        batch = {**batch, "features": {k: v.float() for k, v in batch["features"].items()}}
        predictions = model.predict(batch["features"])
        detection, _ = native(predictions, batch)
        before = model._mixture_loss_ema_buf.detach().clone()
        aux = _collect_mixture_aux_loss(
            model,
            device,
            moe_gain=0,
            moa_gain=0,
            mot_gain=0,
            latent_gain=float(source.args.latent_aux_gain),
            aux_budget=3.0,
        )
        if any(model._mixture_aux_nonfinite.values()):
            raise FloatingPointError("Nonfinite probe aux cannot be silently isolated")
        selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        gradients = separated_gradients(detection.sum(), aux, (p for _, p in selected), (name for name, _ in selected))
        scales = {}
        for name, mixture in model.mixtures.items():
            snapshot = plain(mixture.routing_snapshot())
            # All-zero coefficients short-circuit raw-loss calculation in the core module.
            # Recompute detached diagnostic values, not losses used by the training graph.
            probs, logits = mixture.routing_probs.detach().float(), mixture.routing_logits.detach().float()
            importance = probs.reshape(-1, probs.shape[-1]).mean(0)
            raw_balance = (mixture.num_experts * importance.square().sum() - 1).clamp_min(0)
            raw_z = logits.logsumexp(-1).square().mean()
            scales[name] = {
                "routing": snapshot,
                "raw_balance_local": float(raw_balance),
                "raw_z": float(raw_z),
                "weighted_balance": float(raw_balance) * mixture.balance_loss_coeff,
                "weighted_z": float(raw_z) * mixture.router_z_loss_coeff,
                "balance_coeff": mixture.balance_loss_coeff,
                "z_coeff": mixture.router_z_loss_coeff,
            }
        raw = sum(s["routing"]["aux_loss"] for s in scales.values())
        denominator = min(max(float(model._mixture_loss_ema_buf[3]), 1e-6), 1e6)
        normalized = float(source.args.latent_aux_gain) * raw / denominator
        budget_scale = min(1.0, 3.0 / max(abs(normalized), 1e-4))
        if abs(float(aux) - normalized * budget_scale) > 1e-5:
            raise ValueError("Aux telemetry differs from the actual collector")
        result = {
            "scope": "Independent FP32 copy; fixed two train samples per rank before this epoch; not epoch averages",
            "scales": scales,
            "aux_ema_before": plain(before),
            "aux_ema_after": plain(model._mixture_loss_ema_buf),
            "ema_denominator": denominator,
            "normalized_aux": normalized,
            "budget_scale": budget_scale,
            "effective_aux": float(aux.detach()),
            "native_loss": float(detection.detach().sum()),
            "aux_native_ratio": float(aux.detach() / detection.detach().sum().abs())
            if float(detection.detach().sum()) != 0
            else None,
            "gradients": dict(zip((name for name, _ in selected), gradients)),
        }
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Probe populated parameter gradients")
        return result

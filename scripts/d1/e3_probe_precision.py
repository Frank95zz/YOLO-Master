"""Higher-accuracy, first-order Softmax VJP for independent E3 diagnostics only."""

from contextlib import contextmanager

import torch
from torch.autograd.function import once_differentiable

VERSION = "softmax-vjp-fp64-v1"


class DiagnosticSoftmax(torch.autograd.Function):
    """Keep the exact FP32 forward, but accumulate the small routing VJP in FP64."""

    @staticmethod
    def forward(ctx, scaled_logits):
        if scaled_logits.dtype != torch.float32:
            raise TypeError("The independent E3 probe requires FP32 logits")
        probabilities = torch.softmax(scaled_logits, dim=-1)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        (probabilities,) = ctx.saved_tensors
        p, g = probabilities.double(), gradient.double()
        # Separate and joint VJPs otherwise amplify cancellation in this reduction.
        result = p * (g - (g * p).sum(dim=-1, keepdim=True))
        return result.to(gradient.dtype)


@contextmanager
def accurate_router_vjp(independent_model):
    """Install hooks only on the disposable probe copy and always remove them."""
    handles = []

    def replace_probabilities(router, inputs, output):
        logits, original = output
        scaled = logits / router.temperature.clamp_min(0.1)
        probabilities = DiagnosticSoftmax.apply(scaled)
        if not torch.equal(probabilities, original):
            raise ValueError("Diagnostic Softmax changed the FP32 forward")
        return logits, probabilities

    try:
        for mixture in independent_model.mixtures.values():
            handles.append(mixture.router.register_forward_hook(replace_probabilities))
        yield
    finally:
        for handle in handles:
            handle.remove()

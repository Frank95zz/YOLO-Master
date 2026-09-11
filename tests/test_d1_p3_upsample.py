"""Regression gates for the opt-in deterministic P3 implementation and registered suite."""

import pytest
import torch
import torch.nn.functional as F

from scripts.d1 import train as p5
from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.modules import DINOFeaturePyramidAdapter, SeparableBilinear2x

MODE = "separable_bilinear2x"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((1, 1, 1, 1), (2, 3, 1, 7), (1, 3, 5, 1), (2, 3, 5, 7)))
def test_cpu_bilinear_output_and_gradient(dtype, shape):
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    expected = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    actual = SeparableBilinear2x()(z)
    grad = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z, grad)[0], torch.autograd.grad(expected, x, grad)[0], atol=2e-6, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32, torch.float16))
@pytest.mark.parametrize("shape", ((1, 1, 1, 1), (2, 3, 1, 7), (1, 2, 5, 1), (2, 3, 5, 7), (2, 64, 40, 40)))
@pytest.mark.parametrize("layout", ("contiguous", "channels_last", "transposed"))
def test_cuda_deterministic_output_gradient_and_repeat(dtype, shape, layout):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        x = torch.randn(shape, dtype=dtype, device="cuda")
        if layout == "channels_last":
            x = x.contiguous(memory_format=torch.channels_last)
        elif layout == "transposed":
            x = x.transpose(-2, -1)
        x = x.detach().requires_grad_(True)
        z = x.detach().clone().requires_grad_(True)
        expected = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        actual = SeparableBilinear2x()(z)
        assert torch.equal(expected, actual)
        grad = torch.randn_like(actual)
        a = torch.autograd.grad(expected, x, grad)[0]
        b = torch.autograd.grad(actual, z, grad)[0]
        atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-6, 1e-5)
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        repeated_input = x.detach().clone().requires_grad_(True)
        repeated = SeparableBilinear2x()(repeated_input)
        repeated_grad = torch.autograd.grad(repeated, repeated_input, grad)[0]
        assert torch.equal(actual, repeated) and torch.equal(b, repeated_grad)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_double_backward_and_empty_state():
    module = SeparableBilinear2x()
    x = torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(module, (x,))
    assert torch.autograd.gradgradcheck(module, (x,))
    assert not module.state_dict() and not list(module.parameters())


@pytest.mark.parametrize("mode", (None, True, [], "nearest", "typo"))
def test_invalid_p3_modes_fail_closed(mode):
    with pytest.raises(ValueError):
        DINOFeaturePyramidAdapter(p3_upsample_mode=mode)
    with pytest.raises(ValueError):
        p5.model_config("BASE", mode)


@pytest.mark.parametrize("variant", tuple(p5.VARIANTS))
def test_registered_model_preserves_initial_state_and_checkpoint(variant, tmp_path):
    legacy = p5.construct_model(variant)
    fast = p5.construct_model(variant, MODE)
    assert all(isinstance(fast.adapter.branches["p3"][n][-1], SeparableBilinear2x) for n in fast.source_names)
    assert all(isinstance(legacy.adapter.branches["p3"][n][-1], torch.nn.Upsample) for n in legacy.source_names)
    assert set(legacy.state_dict()) == set(fast.state_dict())
    fast.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(p.numel() for p in fast.parameters()) == p5.VARIANTS[variant]["downstream_parameters"]
    features = {n: torch.randn(1, 384, 4, 4) for n in fast.source_names}
    batch = {
        "features": features,
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, items = fast(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.adapter.parameters())
    path = tmp_path / "checkpoint.pt"
    torch.save(fast.checkpoint_payload(), path)
    restored = D1FoundationDetectionModel.from_checkpoint_payload(torch.load(path, weights_only=False)).eval()
    assert restored.config_dict()["adapter"]["p3_upsample_mode"] == MODE
    with torch.no_grad():
        torch.testing.assert_close(fast.eval()(features)[0], restored(features)[0], rtol=0, atol=0)

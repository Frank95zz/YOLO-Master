"""D1 opt-in EMA arithmetic, state lifecycle and registration boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts.d1 import train as p5
from scripts.d1.ema import configure_d1_ema, validate_ema_implementation
from ultralytics.utils.torch_utils import ModelEMA


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("variant", p5.VARIANTS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.float16, torch.bfloat16))
def test_exact_update_and_persistent_buffer_rules(variant, dtype):
    model = p5.construct_model(variant).to(dtype=dtype)
    model.register_buffer("test_float", torch.randn(7, dtype=dtype))
    model.register_buffer("test_int", torch.tensor(4))
    model.register_buffer("test_bool", torch.tensor(False))
    model.register_buffer("test_temporary", torch.tensor(1.0), persistent=False)
    # Include heterogeneous source dtypes and empty tensors in the same update.
    model.register_buffer("test_double", torch.ones(3, dtype=torch.float64))
    model.register_buffer("test_empty", torch.empty(0))
    original = ModelEMA(model, updates=100, decay=0.99, tau=30)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    for _ in range(3):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.002)
            model.test_float.mul_(1.5)
            model.test_int.add_(1)
            model.test_bool.logical_not_()
            model.test_temporary.add_(1)
        original.update(model)
        fast.update(model)
        assert fast.updates == original.updates
        assert_same(original.ema.state_dict(), fast.ema.state_dict())
        assert fast.ema.test_temporary.item() == 1
        assert not any(p.requires_grad for p in fast.ema.parameters())


def test_conversion_preserves_restored_state_and_is_local():
    model = p5.construct_model("BASE")
    existing = ModelEMA(model, updates=87)
    existing.enabled = False
    fast = configure_d1_ema(existing, model, "foreach-v1")
    assert fast.ema is existing.ema and fast.decay is existing.decay
    assert fast.updates == 87 and not fast.enabled
    fast.update(torch.nn.Linear(2, 2))
    assert fast.updates == 87
    assert type(existing) is ModelEMA
    assert configure_d1_ema(existing, model, "scalar-v1") is existing
    assert configure_d1_ema(None, model, "foreach-v1") is None
    with pytest.raises(TypeError):
        configure_d1_ema(fast, model, "foreach-v1")
    with pytest.raises(TypeError):
        configure_d1_ema(ModelEMA(torch.nn.Linear(2, 2)), model, "foreach-v1")
    with pytest.raises(TypeError):
        configure_d1_ema(existing, torch.nn.Linear(2, 2), "foreach-v1")


def test_dynamic_buffers_and_no_per_step_state_serialization(monkeypatch):
    model = p5.construct_model("BASE")
    original = ModelEMA(model)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    for obj in (model, original.ema, fast.ema):
        obj.register_buffer("late_buffer", torch.tensor(3.0))
    model.late_buffer = torch.tensor(5.0)
    original.update(model)
    monkeypatch.setattr(model, "state_dict", lambda *a, **k: pytest.fail("Serialized source during fast update"))
    fast.update(model)
    assert_same(original.ema.state_dict(), fast.ema.state_dict())


def test_hook_and_shape_changes_fail_before_partial_update():
    model = p5.construct_model("BASE")
    fast = configure_d1_ema(ModelEMA(model), model, "foreach-v1")
    hook = model.register_state_dict_pre_hook(lambda *a: None)
    with pytest.raises(ValueError, match="hooks"):
        fast.update(model)
    hook.remove()
    model._mixture_loss_ema_buf = torch.ones(100)
    before = deepcopy(fast.ema.state_dict())
    with pytest.raises(ValueError, match="shape/device"):
        fast.update(model)
    assert fast.updates == 0
    assert_same(before, fast.ema.state_dict())


@pytest.mark.parametrize("value", [None, True, "foreach", "typo", [], 1])
def test_invalid_registration(value):
    with pytest.raises(ValueError):
        validate_ema_implementation(value)


@pytest.mark.parametrize("variant", p5.VARIANTS)
def test_real_loss_optimizer_and_ema_exact(variant):
    torch.manual_seed(0)
    model = p5.construct_model(variant).train()
    original = ModelEMA(model)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = {
        "features": {key: torch.randn(1, 384, 4, 4) for key in model.adapter.source_names},
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss.sum().backward()
        optimizer.step()
        original.update(model)
        fast.update(model)
        assert_same(original.ema.state_dict(), fast.ema.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA; no download")
@pytest.mark.parametrize("variant", p5.VARIANTS)
def test_cuda_amp_overflow_and_resume(variant):
    from ultralytics.engine.extensions.recovery import TrainingRecoveryController
    from ultralytics.engine.trainer import BaseTrainer

    model = p5.construct_model(variant).cuda().train()
    ema = configure_d1_ema(ModelEMA(model), model, "foreach-v1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=16)
    trainer = SimpleNamespace(
        model=model, ema=ema, optimizer=optimizer, scaler=scaler, optimizer_steps=0, adapter_controller=None
    )
    trainer._recovery_controller = lambda: TrainingRecoveryController(trainer)
    trainer._sync_nonfinite_flag = lambda bad: bool(bad)
    before = deepcopy(ema.ema.state_dict())
    scaler.scale(next(model.parameters()).square().mean()).backward()
    next(model.parameters()).grad.reshape(-1)[0] = float("inf")
    assert not BaseTrainer.optimizer_step(trainer)
    assert trainer.optimizer_steps == ema.updates == 0 and scaler.get_scale() == 8
    assert_same(before, ema.ema.state_dict())
    reference = ModelEMA(model)
    reference.ema.load_state_dict(ema.ema.state_dict(), strict=True)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(next(model.parameters()).square().mean()).backward()
        assert BaseTrainer.optimizer_step(trainer)
        reference.update(model)
        assert_same(reference.ema.state_dict(), ema.ema.state_dict())
    restored = ModelEMA(model, updates=ema.updates)
    restored.ema.load_state_dict(ema.ema.state_dict(), strict=True)
    restored = configure_d1_ema(restored, model, "foreach-v1")
    restored.update(model)
    ema.update(model)
    assert_same(restored.ema.state_dict(), ema.ema.state_dict())

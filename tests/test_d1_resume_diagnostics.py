"""Tests for the first-difference probe; no training or network models required."""

import pytest
import torch

from scripts.d1.diagnose_resume import STAGES, compare, cpu_tree, fingerprint
from scripts.d1.p1p2_runtime import rank_buffer_policy, rank_buffer_state, restore_rank_buffers


def test_cpu_tree_is_independent_and_weights_only_safe(tmp_path):
    source = {"tensor": torch.tensor([1.0], requires_grad=True), "metadata": (1, None, False)}
    snapshot = cpu_tree(source)
    with torch.no_grad():
        source["tensor"].add_(1)
    assert snapshot["tensor"].item() == 1
    assert not snapshot["tensor"].requires_grad
    path = tmp_path / "probe.pt"
    torch.save(snapshot, path)
    assert compare(snapshot, torch.load(path, weights_only=True))["exact"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int64, torch.bool])
def test_fingerprint_content_and_layout(dtype):
    value = torch.ones(2, 3, dtype=dtype)
    assert fingerprint(value) == fingerprint(value.T.contiguous().T)
    modified = value.clone()
    modified[0, 0] = 0
    assert fingerprint(value) != fingerprint(modified)


def test_compare_detects_single_ulp_without_tolerance():
    left = torch.tensor([1.0, 2.0])
    right = torch.nextafter(left, torch.full_like(left, float("inf")))
    report = compare({"gradient": left}, {"gradient": right})
    assert not report["exact"]
    assert report["differences"][0]["different_elements"] == 2
    assert report["differences"][0]["max_abs"] == 2**-22


@pytest.mark.parametrize("left,right", [({"a": 1}, {}), ([1], [1, 2]), (1, 1.0), (False, True)])
def test_compare_rejects_structure_drift(left, right):
    assert not compare(left, right)["exact"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_not_an_exact_pass(value):
    assert not compare(torch.tensor(value), torch.tensor(value))["exact"]


def test_unsupported_snapshot_rejected():
    with pytest.raises(TypeError):
        cpu_tree(object())
    with pytest.raises(TypeError):
        fingerprint(object())


def test_stage_order_distinguishes_ddp_reduction():
    assert STAGES.index("local_gradients") < STAGES.index("reduced_gradients") < STAGES.index("after")


def test_rank_buffers_restore_and_do_not_alias():
    module = torch.nn.BatchNorm1d(3)
    module.register_buffer("transient", torch.tensor(9.0), persistent=False)
    state = rank_buffer_state(module)
    module.running_mean.fill_(7)
    module.transient.fill_(4)
    restore_rank_buffers(module, state)
    assert torch.equal(module.running_mean, torch.zeros(3))
    assert module.transient.item() == 9
    module.running_mean.add_(1)
    assert state["running_mean"].sum() == 0


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "dtype", "nonfinite", "type"])
def test_rank_buffers_fail_before_any_copy(failure):
    module = torch.nn.BatchNorm1d(3)
    state = rank_buffer_state(module)
    state["running_mean"].fill_(7)
    if failure == "missing":
        del state["running_var"]
    elif failure == "extra":
        state["extra"] = torch.tensor(1)
    elif failure == "shape":
        state["running_var"] = torch.ones(4)
    elif failure == "dtype":
        state["running_var"] = state["running_var"].double()
    elif failure == "nonfinite":
        state["running_var"].fill_(float("nan"))
    else:
        state = None
    with pytest.raises(ValueError):
        restore_rank_buffers(module, state)
    assert torch.equal(module.running_mean, torch.zeros(3))


def test_old_resume_identity_is_unchanged():
    legacy = {"identity": {"variant": "BASE"}}
    assert not rank_buffer_policy(legacy)
    assert legacy == {"identity": {"variant": "BASE"}}
    assert rank_buffer_policy({"resume_rank_buffers": True, "identity": {"resume_rank_buffers": True}})


@pytest.mark.parametrize("enabled,recorded", [(True, False), (False, True), (1, True), (True, 1), ("true", True)])
def test_rank_buffer_policy_is_strict(enabled, recorded):
    with pytest.raises(ValueError):
        rank_buffer_policy({"resume_rank_buffers": enabled, "identity": {"resume_rank_buffers": recorded}})

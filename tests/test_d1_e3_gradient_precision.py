"""Diagnostic-only VJP accuracy and fail-closed E3 code provenance."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.d1 import run_e3 as e3
from scripts.d1.e3_probe_precision import DiagnosticSoftmax, accurate_router_vjp
from scripts.d1.p1p2_runtime import separated_gradients
from ultralytics.nn.modules.latent_mixture import LatentRouter


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
def test_exact_forward_and_fp64_vjp(shape):
    torch.manual_seed(12)
    x = torch.randn(shape, requires_grad=True)
    g = torch.randn(shape) * 10000
    actual = DiagnosticSoftmax.apply(x)
    original = torch.softmax(x, -1)
    assert torch.equal(actual, original)
    p = original.detach().double()
    expected = (p * (g.double() - (g.double() * p).sum(-1, keepdim=True))).float()
    grad = torch.autograd.grad(actual, x, g)[0]
    assert torch.equal(grad, expected)
    assert grad.dtype == torch.float32


def test_hook_scope_and_exception_cleanup():
    router = LatentRouter(8, 4)
    model = SimpleNamespace(mixtures={"p3": SimpleNamespace(router=router)})
    inputs = torch.randn(2, 8)
    before = tuple(x.detach().clone() for x in router(inputs))
    state = {k: v.clone() for k, v in router.state_dict().items()}
    with pytest.raises(RuntimeError, match="test exit"):
        with accurate_router_vjp(model):
            assert len(router._forward_hooks) == 1
            assert all(torch.equal(a, b) for a, b in zip(before, router(inputs)))
            raise RuntimeError("test exit")
    assert not router._forward_hooks
    assert all(torch.equal(v, router.state_dict()[k]) for k, v in state.items())
    assert all(p.grad is None for p in router.parameters())


def test_forward_drift_is_rejected():
    router = LatentRouter(8, 4)
    model = SimpleNamespace(mixtures={"p3": SimpleNamespace(router=router)})
    with accurate_router_vjp(model), patch.object(DiagnosticSoftmax, "apply", return_value=torch.zeros(2, 4)):
        with pytest.raises(ValueError, match="changed the FP32 forward"):
            router(torch.randn(2, 8))


def test_non_fp32_input_rejected():
    with pytest.raises(TypeError, match="FP32"):
        DiagnosticSoftmax.apply(torch.zeros(2, 4, dtype=torch.float64))


def test_strict_gradient_gate_still_rejects_corruption_and_nonfinite():
    x = torch.ones(1, requires_grad=True)
    with patch("torch.autograd.grad", side_effect=[(torch.ones(1),), (torch.ones(1),), (torch.ones(1),)]):
        with pytest.raises(ValueError, match="decomposition changed"):
            separated_gradients(x.sum(), x.sum(), (x,))
    with patch("torch.autograd.grad", side_effect=[(torch.tensor([float("nan")]),), (x,), (x,)]):
        with pytest.raises(FloatingPointError):
            separated_gradients(x.sum(), x.sum(), (x,))


def revision_files(tmp_path):
    old = {"commit": "a" * 40, "contract_sha256": "c" * 64}
    new = {**old, "commit": "b" * 40}
    evidence = tmp_path / "diagnostic-repair-validation.json"
    evidence.write_text(json.dumps({"status": "PASSED", "execution_identity": new}))
    revision = {
        "schema_version": "d1-e3-diagnostic-revision-v1",
        "approved": True,
        "original_identity": old,
        "execution_identity": new,
        "validation_sha256": e3.file_sha(evidence),
    }
    (tmp_path / "execution-revision.json").write_text(json.dumps(revision))
    return old, new, revision


def test_revision_only_permits_scoped_diagnostic_change(tmp_path):
    old, new, _ = revision_files(tmp_path)
    with patch.object(e3.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
        with patch.object(e3.subprocess, "check_output", return_value="scripts/d1/e3_probe_precision.py\n"):
            e3.verify_diagnostic_revision(tmp_path, old, new)
        with patch.object(e3.subprocess, "check_output", return_value="ultralytics/nn/modules/latent_mixture.py\n"):
            with pytest.raises(ValueError, match="unapproved"):
                e3.verify_diagnostic_revision(tmp_path, old, new)


@pytest.mark.parametrize("failure", ["no_approval", "contract", "commit", "hash", "validation"])
def test_revision_rejects_unverified_execution(tmp_path, failure):
    old, new, revision = revision_files(tmp_path)
    if failure == "no_approval":
        revision["approved"] = False
    elif failure == "contract":
        new = {**new, "contract_sha256": "d" * 64}
        revision["execution_identity"] = new
    elif failure == "commit":
        new = {**new, "commit": "d" * 40}
    elif failure == "hash":
        revision["validation_sha256"] = "0" * 64
    else:
        path = tmp_path / "diagnostic-repair-validation.json"
        path.write_text(json.dumps({"status": "FAILED", "execution_identity": new}))
        revision["validation_sha256"] = e3.file_sha(path)
    (tmp_path / "execution-revision.json").write_text(json.dumps(revision))
    with patch.object(e3.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
        with patch.object(e3.subprocess, "check_output", return_value="scripts/d1/e3_probe_precision.py\n"):
            with pytest.raises(ValueError):
                e3.verify_diagnostic_revision(tmp_path, old, new)

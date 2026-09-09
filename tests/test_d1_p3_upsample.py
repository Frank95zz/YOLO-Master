"""Regression gates for the opt-in deterministic P3 implementation and registered suite."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.d1 import run_p5_ablation as p5
from scripts.d1 import run_p5_suite as suite
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


@pytest.mark.parametrize("variant", suite.ORDER)
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


def test_prepare_records_mode_and_rejects_matrix_drift(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "inputs").mkdir(parents=True)
    tensors = {k: v for k, v in p5.construct_model("BASE").state_dict().items() if isinstance(v, torch.Tensor)}
    torch.save(tensors, source / "inputs/initial-frozen.pt")
    (source / "matrix.json").write_text("{}")
    identity = {"commit": "fixed", "source_sha256": "source", "contract_sha256": "contract"}
    provenance = {
        "data_root": "/nvme/rgb",
        "cache_root": "/nvme/cache",
        "nvme_root": "/nvme",
        "data_receipt_sha256": "receipt",
        "data_files": {"E1": "/nvme/coco.yaml"},
    }
    monkeypatch.setattr(p5, "identity", lambda: identity)
    monkeypatch.setattr(p5.e1, "load_matrix", lambda *a, **k: provenance)
    work = tmp_path / "suite"
    matrix = p5.prepare(source, work, MODE)
    assert matrix["p3_upsample_mode"] == MODE
    assert p5.load_matrix(work) == matrix
    matrix["p3_upsample_mode"] = "bilinear"
    p5.write_json(work / "matrix.json", matrix)
    with pytest.raises(ValueError, match="model registration"):
        p5.load_matrix(work)


def make_results():
    return {
        v: {
            "active_job_GPUh": 1.0,
            "final": {"last": {"official": {"AP_all": 0.30, "AP_75": 0.31, "AP_large": 0.40, "AP_small": 0.13}}},
        }
        for v in suite.ORDER
    }


def test_selection_requires_precision_and_measured_cost():
    rows = make_results()
    rows["DW"]["active_job_GPUh"] = 0.9
    rows["BN64"]["active_job_GPUh"] = 0.8
    rows["BN64"]["final"]["last"]["official"]["AP_75"] = 0.29
    result = suite.comparison(rows, p5.load_contract()["selection"])
    assert result["DW"]["retained"] and not result["BN64"]["retained"]
    rows["DW"]["active_job_GPUh"] = 1.0
    assert not suite.comparison(rows, p5.load_contract()["selection"])["DW"]["retained"]
    rows["DW"]["active_job_GPUh"] = float("nan")
    with pytest.raises(ValueError):
        suite.comparison(rows, p5.load_contract()["selection"])


def test_suite_order_and_no_overwrite(tmp_path, monkeypatch):
    matrix = {
        "identity": {"commit": "fixed"},
        "p3_upsample_mode": MODE,
        "ema_implementation": "foreach-v1",
        "resume_temperature_policy": "epoch-boundary-v1",
        "resume_rank_buffers": True,
        "parameters": {v: p5.VARIANTS[v]["downstream_parameters"] for v in suite.ORDER},
        "contract": p5.load_contract(),
    }
    monkeypatch.setattr(p5, "load_matrix", lambda work: deepcopy(matrix))
    monkeypatch.setattr(suite.signal, "signal", lambda *a: None)
    monkeypatch.setattr(suite.e1, "validate_evaluation", lambda *a: None)
    labels = []

    def fake_child(self, label, command):
        labels.append(label)
        if "--variant" in command:
            variant = command[command.index("--variant") + 1]
            output = self.workspace / "reports" / f"P5-{variant}"
            weights = self.workspace / "runs" / f"P5-{variant}" / "weights"
            output.mkdir(parents=True)
            weights.mkdir(parents=True)
            (weights / "last.pt").write_bytes(b"checkpoint")
            (weights / "standard-best.pt").write_bytes(b"checkpoint")
            (output / "resume.pt").write_bytes(b"resume")
            suite.write_json(
                output / "training-result.json",
                {
                    "status": "completed",
                    "epochs": 50,
                    "optimizer_steps": 15450,
                    "last_sha256": suite.sha256_file(weights / "last.pt"),
                    "resume_sha256": suite.sha256_file(output / "resume.pt"),
                },
            )
        else:
            destination = command[command.index("--output") + 1]
            suite.write_json(
                suite.Path(destination) / "report.json",
                {
                    "checkpoint_epoch": 50,
                    "official": make_results()["BASE"]["final"]["last"]["official"],
                },
            )
        return 2.0

    monkeypatch.setattr(suite.P5Pipeline, "child_run", fake_child)
    pipeline = suite.P5Pipeline(SimpleNamespace(workspace=tmp_path, approved=True))
    pipeline.execute()
    launch = suite.e1.read_json(tmp_path / "suite-launch.json")
    assert launch["ema_implementation"] == "foreach-v1"
    assert launch["resume_temperature_policy"] == "epoch-boundary-v1"
    assert launch["resume_rank_buffers"] is True
    assert labels == [
        name for v in suite.ORDER for name in (f"P5-{v}", f"P5-{v}-final-last", f"P5-{v}-final-standard-best")
    ]
    with pytest.raises(FileExistsError):
        pipeline.execute()


def test_suite_cli_requires_approval(tmp_path):
    with pytest.raises(SystemExit):
        suite.main(["--workspace", str(tmp_path)])
    assert not (tmp_path / "suite-launch.json").exists()

"""Offline gates for the independent E0/E1 experiment contract."""

import pytest
import torch

from scripts.d1 import run_p1p2 as experiment
from scripts.d1.p1p2_runtime import clean_child_env, detached_state, load_initial_tensors, load_resume_tensors
from ultralytics.utils import YAML


def test_registered_contract():
    contract = experiment.load_contract()
    assert contract["screen_epochs"] == contract["schedule_epochs"] * contract["screen_fraction"] == 50
    assert contract["global_batch"] == contract["world_size"] * contract["per_gpu_batch"] == 384


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("screen_epochs", 30),
        ("seed", True),
        ("nc", 10),
        ("world_size", 2),
        ("workers", 8),
        ("dataset", "coco8"),
        ("unknown", "accepted"),
    ],
)
def test_modified_contract_rejected(tmp_path, key, value):
    contract = experiment.load_contract()
    contract[key] = value
    path = tmp_path / "contract.yaml"
    YAML.save(path, contract)
    with pytest.raises(ValueError):
        experiment.load_contract(path)


def test_common_initial_state_and_parameters():
    states = {}
    for variant in "ABCS":
        model = experiment.construct_model(variant)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count == (3510624 if variant == "S" else 3542567)
        assert not any("teacher" in name.lower() for name, _ in model.named_parameters())
        states[variant] = model.state_dict()
    for variant in "BC":
        assert states[variant].keys() == states["A"].keys()
        assert all(
            torch.equal(value, states[variant][key])
            for key, value in states["A"].items()
            if isinstance(value, torch.Tensor)
        )
    assert abs(3510624 / 3542567 - 1) < 0.01


def test_initial_loading_keeps_variant_metadata():
    a = experiment.construct_model("A")
    b = experiment.construct_model("B")
    state = {key: value for key, value in a.state_dict().items() if isinstance(value, torch.Tensor)}
    load_initial_tensors(b, state)
    assert all(module.value_fusion_mode == "weighted_sum" for module in b.mixtures.values())
    metadata = [value for key, value in detached_state(b).items() if key.endswith("_extra_state")]
    assert len(metadata) == 3
    assert all(value["value_fusion_mode"] == "weighted_sum" for value in metadata)
    state.pop(next(iter(state)))
    with pytest.raises(ValueError):
        load_initial_tensors(b, state)


@pytest.mark.parametrize("variant", list("ABC"))
def test_resume_registers_aux_buffer_before_strict_load(monkeypatch, variant):
    from scripts.d1.p1p2_runtime import E1FrozenTrainer
    from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer

    checkpoint = experiment.construct_model(variant)
    checkpoint._mixture_loss_ema_buf.fill_(2.5)

    def parent_get_model(self, cfg=None, weights=None, verbose=True):
        assert weights is None
        model = experiment.construct_model(variant)
        del model._mixture_loss_ema_buf
        return model

    monkeypatch.setattr(D1FoundationDetectionTrainer, "get_model", parent_get_model)
    trainer = object.__new__(E1FrozenTrainer)
    trainer.e1 = {"seed": 0}
    trainer.resume = True
    restored = trainer.get_model(weights=checkpoint, verbose=False)
    assert torch.equal(restored._mixture_loss_ema_buf, checkpoint._mixture_loss_ema_buf)
    assert all(
        module.value_fusion_mode == ("router_only" if variant == "A" else "weighted_sum")
        for module in restored.mixtures.values()
    )


@pytest.mark.parametrize("saved_aux", [False, True])
def test_scratch_resume_matches_saved_buffer_schema(saved_aux):
    from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer

    source = experiment.construct_model("S")
    if saved_aux:
        initialize_mixture_loss_ema_buffer(source).fill_(2.5)
    saved = detached_state(source)
    for _ in ("online", "ema"):
        restored = experiment.construct_model("S")
        initialize_mixture_loss_ema_buffer(restored)
        load_resume_tensors(restored, saved, variant="S")
        actual = restored.state_dict()
        assert actual.keys() == saved.keys()
        assert all(torch.equal(value, actual[key]) for key, value in saved.items())
        assert hasattr(restored, "_mixture_loss_ema_buf") == saved_aux


@pytest.mark.parametrize("variant", list("ABC"))
def test_frozen_resume_preserves_aux_and_rejects_missing_buffer(variant):
    source = experiment.construct_model(variant)
    source._mixture_loss_ema_buf.fill_(2.5)
    saved = detached_state(source)
    restored = experiment.construct_model(variant)
    load_resume_tensors(restored, saved, variant=variant)
    assert torch.equal(restored._mixture_loss_ema_buf, source._mixture_loss_ema_buf)
    saved.pop("_mixture_loss_ema_buf")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_resume_tensors(restored, saved, variant=variant)
    with pytest.raises(ValueError, match="Scratch resume"):
        load_resume_tensors(restored, saved, variant="S")


@pytest.mark.parametrize("variant", list("ABCS"))
@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_resume_keeps_strict_parameter_validation(variant, corruption):
    from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer

    source = experiment.construct_model(variant)
    saved = detached_state(source)
    restored = experiment.construct_model(variant)
    initialize_mixture_loss_ema_buffer(restored)
    if corruption == "missing":
        saved.pop(next(iter(dict(source.named_parameters()))))
    else:
        saved["unexpected_parameter"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        load_resume_tensors(restored, saved, variant=variant)


def matrix(tmp_path):
    return {
        "workspace": str(tmp_path),
        "identity": {"commit": "abc"},
        "data_receipt_sha256": "receipt",
        "parameters": dict.fromkeys("ABC", 3542567) | {"S": 3510624},
        "data_files": {key: f"/nvme/{key}.yaml" for key in ("E0", "E1", "benchmark")},
        "models": {key: str(tmp_path / f"{key}.yaml") for key in "ABCS"},
    }


def test_profile_and_resume_identity(tmp_path):
    value = matrix(tmp_path)
    first = experiment.spec_for(value, "E0", "A")
    resumed = experiment.spec_for(value, "E0-resume", "A")
    assert first["identity"] == resumed["identity"]
    assert (first["window"], resumed["window"]) == (1, 2)
    assert first["identity"] != experiment.spec_for(value, "E0", "B")["identity"]
    assert first["identity"] != experiment.spec_for(value, "E1", "A")["identity"]
    with pytest.raises(ValueError):
        experiment.spec_for(value, "E1", "A", window=30)
    with pytest.raises(ValueError):
        experiment.spec_for(value, "E2", "A")


def test_fair_overrides(tmp_path):
    value = matrix(tmp_path)
    prepared = []
    for variant in "ABCS":
        spec = experiment.spec_for(value, "E1", variant)
        overrides = experiment.overrides_for(value, spec)
        assert overrides["epochs"] == 100 and spec["window"] == 50
        assert overrides["batch"] == overrides["nbs"] == 384
        assert overrides["workers"] == 4 and overrides["save_period"] == 5
        assert all(overrides[key] == 0 for key in experiment.scratch.AUGMENTATIONS)
        for key in ("name", "model", "latent_aux_gain"):
            overrides.pop(key)
        prepared.append(overrides)
    assert all(row == prepared[0] for row in prepared)


def test_evaluator_does_not_inherit_ddp(monkeypatch):
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        monkeypatch.setenv(key, "5")
    env = clean_child_env()
    assert not any(key in env for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "TORCHELASTIC_RUN_ID"))


def test_wrong_variant_rejected():
    with pytest.raises(ValueError):
        experiment.model_config("D")


@pytest.mark.parametrize("gain", [0.0, 0.1])
def test_gradient_diagnostics_preserve_additive_gradient(gain):
    from scripts.d1.p1p2_runtime import separated_gradients

    first = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    second = torch.nn.Parameter(torch.tensor([5.0]))
    rows = separated_gradients(first.square().sum(), gain * (first.sum() + second.square().sum()), (first, second))
    assert all(parameter.grad is None for parameter in (first, second))
    assert rows[0]["detection"] > 0 and rows[1]["detection"] == 0
    assert (rows[1]["aux"] > 0) == (gain > 0)


def test_mechanism_probe_does_not_modify_training_model():
    from scripts.d1.p1p2_runtime import mechanism_evidence

    model = experiment.construct_model("B").train()
    before = detached_state(model)
    batch = {
        "features": {name: torch.randn(2, 384, 4, 4).half() for name in model.source_names},
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]]),
    }
    result = mechanism_evidence(model, batch)
    assert set(result) == {"active", "aux_zero", "balance_only", "z_only"}
    assert all(row["aux_requires_grad"] for row in result.values())
    assert set(result["active"]["amplitudes"]["fused"]) == {"p3", "p4", "p5"}
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value) if isinstance(value, torch.Tensor) else before[key] == value


def test_source_identity_accepts_only_unchanged_descendants(monkeypatch):
    from types import SimpleNamespace

    recorded = {"commit": "a", "source_sha256": "s", "contract_sha256": "c"}
    experiment.verify_source_identity(recorded, recorded)
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    experiment.verify_source_identity(recorded, {**recorded, "commit": "doc"})
    with pytest.raises(ValueError):
        experiment.verify_source_identity(recorded, {**recorded, "source_sha256": "changed"})
    with pytest.raises(ValueError):
        experiment.verify_source_identity(recorded, {**recorded, "contract_sha256": "changed"})
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError):
        experiment.verify_source_identity(recorded, {**recorded, "commit": "unrelated"})


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_diagnostic_precision_restores_backend_and_autocast(fail, precision):
    from scripts.d1.p1p2_runtime import diagnostic_precision

    original = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = True
        x = torch.ones(2, 2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            try:
                with diagnostic_precision("cpu"):
                    assert not torch.backends.cuda.matmul.allow_tf32
                    assert not torch.backends.cudnn.allow_tf32
                    assert (x @ x).dtype == torch.float32
                    if fail:
                        raise RuntimeError("probe failed")
            except RuntimeError as error:
                assert fail and str(error) == "probe failed"
            assert (x @ x).dtype == torch.bfloat16
        assert torch.get_float32_matmul_precision() == precision
        assert torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(original[0])
        torch.backends.cudnn.allow_tf32 = original[1]


def test_mechanism_enters_precision_context(monkeypatch):
    from scripts.d1 import p1p2_runtime as runtime

    source = torch.nn.Linear(2, 2)
    previous = torch.backends.cudnn.allow_tf32

    def probe(model, batch):
        assert model is source
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        return {"verified": True}

    monkeypatch.setattr(runtime, "_mechanism_evidence", probe)
    assert runtime.mechanism_evidence(source, {}) == {"verified": True}
    assert torch.backends.cudnn.allow_tf32 == previous


@pytest.mark.parametrize("nonfinite", [False, True])
def test_diagnostic_failure_identifies_parameter(monkeypatch, nonfinite):
    from scripts.d1.p1p2_runtime import separated_gradients

    parameter = torch.nn.Parameter(torch.ones(2))
    first = torch.tensor([float("nan"), 1.0]) if nonfinite else torch.ones(2)
    gradients = iter([(first,), (torch.ones(2),), (torch.zeros(2),)])
    monkeypatch.setattr(torch.autograd, "grad", lambda *args, **kwargs: next(gradients))
    with pytest.raises(FloatingPointError if nonfinite else ValueError, match="adapter.weight") as caught:
        separated_gradients(parameter.sum(), parameter.sum(), (parameter,), ("adapter.weight",))
    if not nonfinite:
        assert "failed_elements=2/2" in str(caught.value)
        assert "max_abs=2" in str(caught.value)


def test_source_revision_is_exact_and_narrow(monkeypatch):
    from types import SimpleNamespace

    original = {"commit": "a", "source_sha256": "s", "contract_sha256": "c"}
    repaired = {**original, "commit": "b", "source_sha256": "fixed"}
    revision = {
        "schema_version": "d1-e1-diagnostic-revision-v1",
        "approved": True,
        "original_identity": original,
        "execution_identity": repaired,
    }
    monkeypatch.setattr(experiment.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **k: "scripts/d1/p1p2_runtime.py\n")
    experiment.verify_source_identity(original, repaired, revision)
    experiment.verify_source_identity(original, {**repaired, "commit": "docs"}, revision)
    for changed in [None, {**revision, "approved": False}, {**revision, "original_identity": repaired}]:
        with pytest.raises(ValueError):
            experiment.verify_source_identity(original, repaired, changed)
    with pytest.raises(ValueError):
        experiment.verify_source_identity(original, {**repaired, "source_sha256": "other"}, revision)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **k: "ultralytics/nn/tasks.py\n")
    with pytest.raises(ValueError):
        experiment.verify_source_identity(original, repaired, revision)


def test_pipeline_resumes_once_then_reuses_completed_outputs(tmp_path):
    from types import SimpleNamespace

    value = matrix(tmp_path)
    pipeline = object.__new__(experiment.Pipeline)
    pipeline.workspace = tmp_path
    pipeline.args = SimpleNamespace(resume=True)
    output = tmp_path / "reports/E1-A"
    weights = tmp_path / "runs/E1-A/weights"
    output.mkdir(parents=True)
    weights.mkdir(parents=True)
    torch.save({"epoch": 23}, weights / "last.pt")
    torch.save({"epoch": 23, "identity": experiment.spec_for(value, "E1", "A")["identity"]}, output / "resume.pt")
    write = experiment.scratch.write_json
    write(tmp_path / "jobs/E1-A-before.json", {"label": "E1-A", "seconds": 5.0})
    calls = []

    def child(label, command):
        calls.append((label, command))
        if label == "E1-A":
            assert command[-1] == "--resume"
            torch.save({"epoch": 49}, weights / "last.pt")
            torch.save({"epoch": 49}, weights / "standard-best.pt")
            torch.save({"epoch": 49}, output / "resume.pt")
            write(
                output / "training-result.json",
                {
                    "status": "completed",
                    "epochs": 50,
                    "optimizer_steps": 15450,
                    "last_sha256": experiment.sha256_file(weights / "last.pt"),
                    "resume_sha256": experiment.sha256_file(output / "resume.pt"),
                },
            )
            write(tmp_path / "jobs/E1-A-after.json", {"label": "E1-A", "seconds": 7.0})
        else:
            name = "standard-best" if label.endswith("standard-best") else "last"
            write(
                output / "final" / name / "report.json",
                {
                    "status": "passed",
                    "seen": 5000,
                    "identity": value["identity"],
                    "run_id": "E1-A",
                    "strict_reload": True,
                    "checkpoint_sha256": experiment.sha256_file(weights / f"{name}.pt"),
                },
            )
        return 7.0

    pipeline.child_run = child
    pipeline.finish_variant(value, "A")
    assert len(calls) == 3
    assert experiment.read_json(output / "cost.json")["seconds"] == 12.0
    pipeline.finish_variant(value, "A")
    assert len(calls) == 3
    assert experiment.read_json(output / "cost.json")["seconds"] == 12.0
    pipeline.args.resume = False
    with pytest.raises(FileExistsError):
        pipeline.finish_variant(value, "A")
    pipeline.args.resume = True
    path = output / "final/last/report.json"
    write(path, {**experiment.read_json(path), "checkpoint_sha256": "changed"})
    with pytest.raises(ValueError, match="checkpoint or protocol"):
        pipeline.finish_variant(value, "A")


def test_pipeline_rejects_missing_paired_resume(tmp_path):
    from types import SimpleNamespace

    pipeline = object.__new__(experiment.Pipeline)
    pipeline.workspace = tmp_path
    pipeline.args = SimpleNamespace(resume=True)
    path = tmp_path / "reports/E1-A"
    path.mkdir(parents=True)
    torch.save({"epoch": 23}, path / "resume.pt")
    with pytest.raises(FileNotFoundError, match="Both last.pt and resume.pt"):
        pipeline.finish_variant(matrix(tmp_path), "A")

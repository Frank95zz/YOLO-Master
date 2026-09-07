"""Offline gates for the independent E0/E1 experiment contract."""

import pytest
import torch

from scripts.d1 import run_p1p2 as experiment
from scripts.d1.p1p2_runtime import clean_child_env, detached_state, load_initial_tensors
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

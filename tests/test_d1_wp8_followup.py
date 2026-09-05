"""D1 follow-up preparation and direct spatial-fusion regression tests."""

from copy import deepcopy

import pytest
import torch

from scripts.d1 import inspect_wp8_followup as followup
from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.modules import LatentMixture


def test_variants_are_isolated_equal_budget_and_not_authorized():
    contract, configs = followup.variants()
    assert contract["training_authorized"] is False
    assert contract["checks"]["optimizer_steps"] == 0
    assert tuple(configs) == ("A", "B", "C")
    assert configs["A"]["latent_mixture"]["value_fusion_mode"] == "router_only"
    assert configs["B"]["latent_mixture"]["value_fusion_weights"] == [1, 1, 1]
    assert configs["C"]["loss"]["latent_aux_gain"] == 0
    for config in configs.values():
        model = D1FoundationDetectionModel(config)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3542567
    b = deepcopy(configs["B"])
    b["latent_mixture"].pop("value_fusion_weights")
    b["latent_mixture"]["value_fusion_mode"] = "router_only"
    assert b == configs["A"]
    c = deepcopy(configs["C"])
    c["loss"]["latent_aux_gain"] = 0.1
    assert c == configs["B"]
    assert followup.YAML.load(followup.ROOT / contract["base_model"]) == configs["A"]
    assert contract["aux_contract"]["multiply_aux_by_local_batch"] is False


@pytest.mark.parametrize("mode", ["router_only", "weighted_sum"])
def test_spatial_causality_without_router_or_residual_changes(mode):
    module = LatentMixture([8, 8, 8], 8, value_fusion_mode=mode, residual_init=0).eval()
    xs = [torch.randn(2, 8, 4, 4, requires_grad=True) for _ in range(3)]
    output = module(xs)
    expected = xs[0] if mode == "router_only" else sum(xs) / 3
    torch.testing.assert_close(output, expected)
    gradients = torch.autograd.grad(output.sum(), xs, allow_unused=True)
    for index, grad in enumerate(gradients):
        expected_grad = 1.0 if mode == "router_only" and index == 0 else 1 / 3 if mode == "weighted_sum" else 0
        assert grad is not None
        torch.testing.assert_close(grad, torch.full_like(grad, expected_grad))
    changed = [x.detach().clone() for x in xs]
    # Zero-mean spatial perturbation leaves the pooled Router tokens unchanged.
    changed[2][:, :, 0, 0] += 2
    changed[2][:, :, 0, 1] -= 2
    altered = module(changed)
    if mode == "router_only":
        torch.testing.assert_close(output.detach(), altered)
    else:
        assert not torch.allclose(output.detach(), altered)
    restored = LatentMixture([8, 8, 8], 8, value_fusion_mode=mode).eval()
    restored.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(restored(changed), altered)


def test_weighted_detection_loss_reaches_all_nine_adapter_branches():
    _, configs = followup.variants()
    torch.manual_seed(0)
    model = D1FoundationDetectionModel(configs["C"]).train()
    batch = {
        "features": {name: torch.randn(2, 384, 4, 4) for name in followup.NAMES},
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]]),
    }
    loss, items = model(batch)
    assert loss.shape == (3,)
    assert items[-1] == 0
    loss.sum().backward()
    for branches in model.adapter.branches.values():
        for branch in branches.values():
            gradients = [p.grad for p in branch.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_preparation_is_deterministic_and_refuses_existing_output(tmp_path):
    output = followup.fresh_output(tmp_path / "first")
    first = followup.prepare(output)
    second = followup.prepare(followup.fresh_output(tmp_path / "second"))
    assert first["variants"] == second["variants"]
    assert first["status"] == "prepared_not_trained"
    with pytest.raises(FileExistsError):
        followup.fresh_output(output)
    with pytest.raises(ValueError, match="outside"):
        followup.fresh_output(followup.ROOT / "forbidden-runtime")


def test_selection_spans_entire_split_and_rejects_misaligned_records(monkeypatch):
    from types import SimpleNamespace

    paths = [f"images/val2017/{index:012d}.jpg" for index in range(5000)]
    records = {
        f"val2017/{index:012d}": {"sample_id": f"val2017/{index:012d}", "image_path": path}
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(followup, "split_paths", lambda *args: (paths, "unused"))
    reader = SimpleNamespace(records=records)
    selected = followup.select_records(reader, "val2017")
    assert len(selected) == 8
    assert selected[0]["image_path"] == paths[0]
    assert selected[-1]["image_path"] == paths[-1]
    assert len({r["sample_id"] for r in selected}) == 8
    assert selected == followup.select_records(reader, "val2017")
    records[selected[0]["sample_id"]]["image_path"] = paths[-1]
    with pytest.raises(ValueError, match="identity"):
        followup.select_records(reader, "val2017")

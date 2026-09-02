"""Tests for the D1 WP6 latent auxiliary-loss closure."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.modules.routing_protocol import get_aux_record, iter_aux_records


SOURCE_NAMES = ("block4", "block8", "block12")
PYRAMID_NAMES = ("p3", "p4", "p5")


def make_features(*, batch: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        name: torch.randn(batch, 384, 4, 4, generator=generator)
        for name in SOURCE_NAMES
    }


def empty_batch(features: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "features": features,
        "batch_idx": torch.empty(0),
        "cls": torch.empty(0, 1),
        "bboxes": torch.empty(0, 4),
    }


def test_three_latent_publications_are_collected_once_and_reported() -> None:
    model = D1FoundationDetectionModel().train()
    for mixture in model.mixtures.values():
        with torch.no_grad():
            mixture.router.expert_head.bias.copy_(torch.tensor([0.3, -0.2, 0.1, -0.1]))

    loss, items = model(empty_batch(make_features()))

    assert tuple(loss.shape) == (3,)
    assert tuple(items.shape) == (7,)
    diagnostics = model._mixture_aux_diagnostics
    assert diagnostics["counts_by_kind"]["latent"] == 3
    assert diagnostics["stale_skipped"] == 0
    assert diagnostics["eval_skipped"] == 0
    assert diagnostics["duplicate_skipped"] == 0
    assert diagnostics["modules"] == ["LatentMixture"] * 3
    assert len(diagnostics["values_by_kind"]["latent"]) == 3

    records = iter_aux_records(model)
    assert len(records) == 3
    assert {record.kind for _module, record in records} == {"latent"}
    assert {record.step for _module, record in records} == {diagnostics["step"]}
    assert all(record.training and record.value.requires_grad for _module, record in records)

    metrics = model.last_latent_aux_metrics
    assert metrics["latent_publications"] == 3
    assert metrics["aux_step"] == diagnostics["step"]
    assert items[3].item() == pytest.approx(metrics["latent_balance_loss"])
    assert items[4].item() == pytest.approx(metrics["latent_z_loss"])
    assert items[5].item() == pytest.approx(metrics["latent_aux_loss"])
    assert items[6].item() == pytest.approx(metrics["mixture_aux_loss"])
    assert metrics["latent_aux_loss"] == pytest.approx(sum(diagnostics["values_by_kind"]["latent"]))
    assert metrics["latent_balance_loss"] > 0.0

    for name in PYRAMID_NAMES:
        snapshot = model.mixtures[name].routing_snapshot()
        expected = (
            model.mixtures[name].balance_loss_coeff * snapshot["balance_loss"]
            + model.mixtures[name].router_z_loss_coeff * snapshot["z_loss"]
        )
        assert snapshot["aux_loss"] == pytest.approx(expected)
        assert metrics[f"{name}_aux_loss"] == pytest.approx(expected)


def test_new_forward_step_replaces_publications_without_accumulation() -> None:
    model = D1FoundationDetectionModel().train()
    model(empty_batch(make_features()))
    first = deepcopy(model._mixture_aux_diagnostics)

    model(empty_batch(make_features()))
    second = model._mixture_aux_diagnostics

    assert second["step"] > first["step"]
    assert first["counts_by_kind"]["latent"] == second["counts_by_kind"]["latent"] == 3
    assert len(iter_aux_records(model)) == 3
    assert all(get_aux_record(model.mixtures[name]).step == second["step"] for name in PYRAMID_NAMES)


def test_enabled_latent_aux_reaches_every_router() -> None:
    model = D1FoundationDetectionModel().train()

    loss, _items = model(empty_batch(make_features(batch=2)))
    loss.sum().backward()

    assert model.last_latent_aux_metrics["latent_aux_loss"] > 0.0
    assert model.last_latent_aux_metrics["mixture_aux_loss"] > 0.0
    for name in PYRAMID_NAMES:
        gradient = model.mixtures[name].router.expert_head.bias.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_zero_global_latent_gain_disables_only_the_applied_aux() -> None:
    config = D1FoundationDetectionModel().config_dict()
    config["loss"]["latent_aux_gain"] = 0.0
    model = D1FoundationDetectionModel(config).train()

    loss, items = model(empty_batch(make_features(batch=2)))

    assert items[4] > 0.0
    assert items[5] > 0.0
    assert items[6] == 0.0
    assert model.last_latent_aux_metrics["mixture_aux_loss"] == 0.0
    loss.sum().backward()
    for name in PYRAMID_NAMES:
        gradient = model.mixtures[name].router.expert_head.bias.grad
        assert gradient is not None
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_disabled_latent_aux_is_exact_graph_connected_zero() -> None:
    config = D1FoundationDetectionModel().config_dict()
    config["latent_mixture"]["balance_loss_coeff"] = 0.0
    config["latent_mixture"]["router_z_loss_coeff"] = 0.0
    config["loss"]["latent_aux_gain"] = 0.0
    model = D1FoundationDetectionModel(config).train()

    loss, items = model(empty_batch(make_features(batch=2)))

    assert torch.equal(items[3:], torch.zeros_like(items[3:]))
    assert model._mixture_aux_diagnostics["counts_by_kind"]["latent"] == 3
    assert all(
        value == 0.0
        for key, value in model.last_latent_aux_metrics.items()
        if key not in {"aux_step", "latent_publications"}
    )
    loss.sum().backward()
    for name in PYRAMID_NAMES:
        gradient = model.mixtures[name].router.expert_head.bias.grad
        assert gradient is not None
        assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("field,value", [("counts_by_kind", {"latent": 2}), ("duplicate_skipped", 1)])
def test_incomplete_or_duplicate_collection_fails_fast(field: str, value: object) -> None:
    model = D1FoundationDetectionModel().train()
    model(empty_batch(make_features()))
    model._mixture_aux_diagnostics[field] = value

    with pytest.raises(RuntimeError, match="exactly 3|duplicate_skipped"):
        model.mixture_aux_report_items()

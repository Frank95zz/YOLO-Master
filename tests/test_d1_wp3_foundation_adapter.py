"""Tests for the D1 WP3 DINO feature-pyramid adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.foundation.cache import FeatureCacheReader
from ultralytics.nn.modules import DINOFeaturePyramidAdapter, LatentMixture


SOURCE_NAMES = ("block4", "block8", "block12")
EXPECTED_SHAPES = {
    "p3": (2, 64, 80, 80),
    "p4": (2, 128, 40, 40),
    "p5": (2, 256, 20, 20),
}


def make_features(
    *,
    batch: int = 2,
    channels: int = 384,
    height: int = 40,
    width: int = 40,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(
            batch,
            channels,
            height,
            width,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        for name in SOURCE_NAMES
    }


def test_formal_shapes_keys_and_candidate_order() -> None:
    adapter = DINOFeaturePyramidAdapter().eval()
    features = make_features()

    with torch.no_grad():
        outputs = adapter(features)

    assert tuple(outputs) == ("p3", "p4", "p5")
    assert adapter.source_names == SOURCE_NAMES
    assert adapter.pyramid_names == ("p3", "p4", "p5")
    assert adapter.out_channels == (64, 128, 256)
    assert adapter.strides == (8, 16, 32)
    for level, expected_shape in EXPECTED_SHAPES.items():
        assert len(outputs[level]) == 3
        assert all(tuple(candidate.shape) == expected_shape for candidate in outputs[level])
        for index, name in enumerate(SOURCE_NAMES):
            assert torch.equal(outputs[level][index], adapter.branches[level][name](features[name]))


def test_nine_branches_have_independent_parameters_and_group_norm_only() -> None:
    adapter = DINOFeaturePyramidAdapter()
    parameter_sets = []
    for level in adapter.pyramid_names:
        for name in adapter.source_names:
            parameters = {id(parameter) for parameter in adapter.branches[level][name].parameters()}
            assert parameters
            parameter_sets.append(parameters)

    assert len(parameter_sets) == 9
    assert all(first.isdisjoint(second) for i, first in enumerate(parameter_sets) for second in parameter_sets[i + 1 :])
    assert any(isinstance(module, nn.GroupNorm) for module in adapter.modules())
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in adapter.modules())


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"in_channels": 0}, ValueError),
        ({"in_channels": True}, ValueError),
        ({"source_names": "block4"}, TypeError),
        ({"source_names": ("block4", "block8")}, ValueError),
        ({"source_names": ("block4", "block4", "block12")}, ValueError),
        ({"source_names": ("block.4", "block8", "block12")}, ValueError),
        ({"pyramid_channels": (64, 128)}, ValueError),
        ({"pyramid_channels": (64, 0, 256)}, ValueError),
        ({"norm_groups": 0}, ValueError),
    ],
)
def test_invalid_constructor_arguments_fail_fast(kwargs, exception) -> None:
    with pytest.raises(exception):
        DINOFeaturePyramidAdapter(**kwargs)


def test_feature_keys_must_match_exactly() -> None:
    adapter = DINOFeaturePyramidAdapter()
    missing = make_features()
    missing.pop("block8")
    with pytest.raises(ValueError, match="missing"):
        adapter(missing)

    extra = make_features()
    extra["other"] = extra["block4"]
    with pytest.raises(ValueError, match="unexpected"):
        adapter(extra)


@pytest.mark.parametrize(
    ("mutate", "exception", "message"),
    [
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 40, dtype=torch.float32)), ValueError, "BCHW"),
        (lambda xs: xs.update(block8=torch.zeros(2, 383, 40, 40)), ValueError, "channels"),
        (lambda xs: xs.update(block8=torch.zeros(1, 384, 40, 40)), ValueError, "batch"),
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 38, 40)), ValueError, "spatial"),
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 40, 40, dtype=torch.float64)), ValueError, "dtype"),
        (lambda xs: xs.update(block8=torch.zeros(2, 384, 40, 40, dtype=torch.int64)), TypeError, "floating"),
    ],
)
def test_invalid_feature_tensors_fail_fast(mutate, exception, message) -> None:
    features = make_features()
    mutate(features)
    with pytest.raises(exception, match=message):
        DINOFeaturePyramidAdapter()(features)


def test_odd_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        DINOFeaturePyramidAdapter()(make_features(height=39))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for a device-mismatch tensor")
def test_device_mismatch_is_rejected() -> None:
    features = make_features()
    features["block12"] = features["block12"].cuda()
    with pytest.raises(ValueError, match="device"):
        DINOFeaturePyramidAdapter()(features)


def test_forward_is_deterministic_and_state_dict_round_trips() -> None:
    torch.manual_seed(0)
    source = DINOFeaturePyramidAdapter().eval()
    restored = DINOFeaturePyramidAdapter().eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    features = make_features(batch=1)

    with torch.no_grad():
        first = source(features)
        repeated = source(features)
        loaded = restored(features)

    for level in source.pyramid_names:
        for a, b, c in zip(first[level], repeated[level], loaded[level]):
            assert torch.equal(a, b)
            assert torch.equal(a, c)
            assert torch.isfinite(a).all()


def test_all_branches_and_inputs_receive_gradients() -> None:
    adapter = DINOFeaturePyramidAdapter().train()
    features = make_features(batch=1, requires_grad=True)
    outputs = adapter(features)
    loss = sum(candidate.square().mean() for candidates in outputs.values() for candidate in candidates)
    loss.backward()

    for feature in features.values():
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0
    for level in adapter.pyramid_names:
        for name in adapter.source_names:
            convolution = next(
                module for module in adapter.branches[level][name].modules() if isinstance(module, nn.Conv2d)
            )
            assert convolution.weight.grad is not None
            assert torch.isfinite(convolution.weight.grad).all()
            assert convolution.weight.grad.abs().sum() > 0


def test_candidates_feed_three_single_scale_latent_mixtures() -> None:
    adapter = DINOFeaturePyramidAdapter().train()
    features = make_features(batch=1)
    candidates = adapter(features)
    mixtures = nn.ModuleDict(
        {
            level: LatentMixture([channels] * 3, channels, residual_init=0.01)
            for level, channels in zip(adapter.pyramid_names, adapter.out_channels)
        }
    ).train()

    outputs = {level: mixtures[level](candidates[level]) for level in adapter.pyramid_names}

    assert tuple(outputs["p3"].shape) == (1, 64, 80, 80)
    assert tuple(outputs["p4"].shape) == (1, 128, 40, 40)
    assert tuple(outputs["p5"].shape) == (1, 256, 20, 20)
    sum(output.square().mean() for output in outputs.values()).backward()
    assert all(any(parameter.grad is not None for parameter in mixtures[level].parameters()) for level in outputs)


def test_real_wp2_cache_cuda_fp16() -> None:
    cache_value = os.environ.get("D1_WP2_CACHE")
    if not cache_value:
        pytest.skip("D1_WP2_CACHE is not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    reader = FeatureCacheReader(Path(cache_value))
    sample_id = sorted(reader.records)[0]
    cached = reader.get(sample_id)
    assert tuple(cached) == SOURCE_NAMES
    assert all(value.dtype == torch.float16 and tuple(value.shape) == (384, 40, 40) for value in cached.values())
    features = {name: value.unsqueeze(0).cuda(non_blocking=True) for name, value in cached.items()}
    adapter = DINOFeaturePyramidAdapter().cuda().train()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = adapter(features)
        loss = sum(candidate.float().square().mean() for values in outputs.values() for candidate in values)
    loss.backward()

    output_dtypes = {candidate.dtype for values in outputs.values() for candidate in values}
    assert len(output_dtypes) == 1
    assert all(candidate.is_floating_point() for values in outputs.values() for candidate in values)
    assert tuple(outputs["p3"][0].shape) == (1, 64, 80, 80)
    assert tuple(outputs["p4"][0].shape) == (1, 128, 40, 40)
    assert tuple(outputs["p5"][0].shape) == (1, 256, 20, 20)
    assert all(torch.isfinite(candidate).all() for values in outputs.values() for candidate in values)

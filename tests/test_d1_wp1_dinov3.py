"""Optional real-weight integration test for D1 WP1."""

import os
from pathlib import Path

import pytest
import torch

from ultralytics.nn.foundation import DINOv3Teacher


def test_real_dinov3_vits16_multilayer_640():
    weights_value = os.getenv("D1_DINOV3_WEIGHTS")
    if not weights_value:
        pytest.skip("set D1_DINOV3_WEIGHTS to a local Transformers model directory")
    if not torch.cuda.is_available():
        pytest.skip("D1 WP1 real-weight integration test requires CUDA")

    weights = Path(weights_value)
    assert weights.is_dir(), f"D1_DINOV3_WEIGHTS is not a directory: {weights}"
    teacher = DINOv3Teacher(
        weights_path=weights,
        local_files_only=True,
        dtype="fp16",
        device="cuda:0",
        output_layers=(4, 8, 12),
    )
    torch.manual_seed(0)
    images = torch.rand(1, 3, 640, 640)

    first = teacher.encode(images)
    second = teacher.encode(images)

    assert tuple(first.dense) == ("block4", "block8", "block12")
    assert all(feature.shape == (1, 384, 40, 40) for feature in first.dense.values())
    assert first.metadata["output_layers"] == (4, 8, 12)
    assert first.metadata["output_layer_indices"] == (3, 7, 11)
    assert first.metadata["backbone_stages"] == ("stage4", "stage8", "stage12")
    assert first.metadata["feature_names"] == ("block4", "block8", "block12")
    assert first.metadata["prefix_tokens"] == 5
    assert first.metadata["grid_size"] == (40, 40)
    assert first.metadata["hidden_dim"] == 384
    assert teacher.training is False
    assert teacher.model.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(not feature.requires_grad for feature in first.dense.values())
    for name in first.dense:
        torch.testing.assert_close(first.dense[name], second.dense[name], rtol=0, atol=0)
    torch.testing.assert_close(first.pooled, second.pooled, rtol=0, atol=0)

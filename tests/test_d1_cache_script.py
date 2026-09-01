"""Contracts for the D1 WP2 extraction entrypoint."""

from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.cache_d1_features import cache_contract, load_image, make_letterbox, split_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cache_contract_is_derived_from_tracked_wp0_manifests():
    value = cache_contract(REPO_ROOT)

    assert value["schema_version"] == "d1-cache-v1"
    assert value["model_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert value["teacher_weights_sha256"] == "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
    assert value["output_layers"] == [4, 8, 12]
    assert value["feature_names"] == ["block4", "block8", "block12"]
    assert value["dtype"] == "float16"
    assert value["expected_shape"] == [384, 40, 40]


def test_fixed_100_paths_are_sorted_and_stable():
    first, first_sha256 = split_paths(REPO_ROOT, "train2017", 100)
    second, second_sha256 = split_paths(REPO_ROOT, "train2017", 100)

    assert len(first) == 100
    assert first == sorted(first)
    assert first == second
    assert first_sha256 == second_sha256


def test_wp0_letterbox_is_deterministic_rgb_chw(tmp_path):
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    letterbox = make_letterbox()

    first = load_image(path, letterbox)
    second = load_image(path, letterbox)

    assert first.shape == (3, 640, 640)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.allclose(first[:, 320, 320], torch.tensor([30, 20, 10]) / 255)
    assert first.min() >= 0 and first.max() <= 1

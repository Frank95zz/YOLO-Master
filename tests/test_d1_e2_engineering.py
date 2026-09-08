"""Bounded VisDrone trainer, cache identity, export, and budget regressions."""

from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.d1.p1p2_runtime import E1Policy
from scripts.d1.prepare_visdrone import NAMES
from scripts.d1.run_e2 import (
    EXPECTED,
    E2Trainer,
    VisDroneValidator,
    benchmark_summary,
    construct_model,
    load_contract,
    model_config,
    overrides,
    run_spec,
)
from ultralytics.data.d1_cache import D1FeatureCacheDataset, _sample_id
from ultralytics.nn.foundation.cache import FeatureCacheWriter, sha256_file
from ultralytics.utils import DEFAULT_CFG_DICT, YAML


def test_registered_budget_and_e1_compatibility(tmp_path):
    assert load_contract() == EXPECTED
    assert (E1Policy.required_schedule_epochs, E1Policy.required_global_batch, E1Policy.required_workers) == (
        100,
        384,
        4,
    )
    assert (E2Trainer.required_schedule_epochs, E2Trainer.required_global_batch, E2Trainer.required_workers) == (
        300,
        96,
        4,
    )
    config = deepcopy(EXPECTED)
    config["profiles"]["benchmark"]["window_epochs"] = 300
    path = tmp_path / "bad.yaml"
    YAML.save(path, config)
    with pytest.raises(ValueError, match="Unregistered"):
        load_contract(path)


@pytest.mark.parametrize("profile,window", [("smoke", 1), ("smoke-resume", 2), ("benchmark", 3)])
def test_bounded_profile_and_resume_identity(tmp_path, profile, window):
    mat = {"workspace": str(tmp_path), "identity": {"commit": "a" * 40}, "parameters": 1}
    spec = run_spec(mat, profile)
    config = overrides(mat, spec)
    assert spec["window"] == window and spec["profile"] != "E1"
    assert config["epochs"] == 300 and config["batch"] == config["nbs"] == 96
    assert config["max_det"] == 500 and config["workers"] == 4 and config["imgsz"] == 640
    assert config["amp"] and config["deterministic"] and config["optimizer"] == "AdamW"
    assert not any(config[k] for k in ("mosaic", "mixup", "copy_paste", "fliplr", "flipud", "multi_scale"))
    assert run_spec(mat, "smoke")["identity"] == run_spec(mat, "smoke-resume")["identity"]
    with pytest.raises(ValueError, match="registered"):
        run_spec(mat, "formal")


def test_ten_class_model_strict_reload_and_backward():
    torch.set_num_threads(1)
    model = construct_model().train()
    cfg = model_config()
    assert cfg["detect"]["nc"] == model.detect.nc == 10 and model.detect.max_det == 500
    assert cfg["latent_mixture"]["value_fusion_mode"] == "weighted_sum"
    assert not any("teacher" in name.lower() for name, _ in model.named_parameters())
    inputs = {name: torch.randn(1, 384, 40, 40) for name in ("block4", "block8", "block12")}
    loss, metrics = model.loss(
        {
            "img": inputs,
            "features": inputs,
            "batch_idx": torch.zeros(1),
            "cls": torch.tensor([[9.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
    )
    assert torch.isfinite(loss).all() and torch.isfinite(metrics).all()
    loss.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    restored = construct_model()
    restored.load_state_dict(model.state_dict(), strict=True)


@pytest.fixture
def visdrone_cache(tmp_path):
    split, sid = "visdrone-train", "000001_00001_d_0000001"
    data, cache = tmp_path / "data", tmp_path / "cache"
    image = data / "images" / split / f"{sid}.jpg"
    image.parent.mkdir(parents=True)
    label = data / "labels" / split / f"{sid}.txt"
    label.parent.mkdir(parents=True)
    cv2.imwrite(str(image), np.zeros((80, 120, 3), dtype=np.uint8))
    label.write_text("9 0.5 0.5 0.2 0.3\n")
    contract = {
        "schema_version": "d1-cache-v1",
        "model_id": "test/teacher",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": ["block4", "block8", "block12"],
        "expected_shape": [384, 40, 40],
        "dtype": "float16",
    }
    with FeatureCacheWriter(cache, split=split, contract=contract) as writer:
        writer.add(
            sample_id=f"{split}/{sid}",
            split=split,
            image_path=f"images/{split}/{sid}.jpg",
            image_sha256=sha256_file(image),
            features={
                name: torch.full((384, 40, 40), i, dtype=torch.float16)
                for i, name in enumerate(contract["feature_names"])
            },
        )
    return data, cache, image


def test_visdrone_cached_dataset_and_geometry(visdrone_cache):
    data, cache, image = visdrone_cache
    dataset = D1FeatureCacheDataset(
        img_path=str(image.parent),
        cache_dir=cache,
        data={"path": str(data), "nc": 10, "names": dict(enumerate(NAMES))},
        hyp=SimpleNamespace(**DEFAULT_CFG_DICT),
        batch_size=1,
    )
    assert dataset.sample_ids == (f"visdrone-train/{image.stem}",)
    sample = dataset[0]
    assert sample["cls"].item() == 9 and sample["features"]["block12"].dtype == torch.float16
    # The 427px resized height leaves asymmetric integer padding: top=106, bottom=107.
    expected_y = (40 * (640 / 120) + 106) / 640
    assert torch.allclose(sample["bboxes"], torch.tensor([[0.5, expected_y, 0.2, 0.2]]), atol=1e-6)


def test_visdrone_rejects_coco_class_config(visdrone_cache):
    data, cache, image = visdrone_cache
    with pytest.raises(ValueError, match="10"):
        D1FeatureCacheDataset(
            img_path=str(image.parent),
            cache_dir=cache,
            data={"path": str(data), "nc": 80, "names": dict(enumerate(map(str, range(80))))},
            hyp=SimpleNamespace(**DEFAULT_CFG_DICT),
            batch_size=1,
        )


@pytest.mark.parametrize("split", ["visdrone-test-dev", "test2017", "arbitrary"])
def test_unregistered_split_rejected(split):
    with pytest.raises(ValueError):
        _sample_id(f"/images/{split}/000001.jpg")


def test_export_preserves_original_coordinates_and_classes():
    validator = SimpleNamespace(jdict=[], degenerate_predictions=0)
    pred = {
        "bboxes": torch.tensor([[12.25, 20.5, 32.75, 30.75], [1, 1, 1, 2]]),
        "conf": torch.tensor([0.5, 0.4]),
        "cls": torch.tensor([9, 0]),
    }
    VisDroneValidator.pred_to_json(validator, pred, {"im_file": "/val/00001_d_0001.jpg"})
    assert validator.jdict == [
        {"image_id": "00001_d_0001", "bbox": [12.25, 20.5, 20.5, 10.25], "score": 0.5, "category_id": 9}
    ]
    assert validator.degenerate_predictions == 1
    stats = {"metrics/mAP50(B)": 0.1}
    assert VisDroneValidator.eval_json(validator, stats) is stats


@pytest.mark.parametrize("category", [float("nan"), 10, -1, 0.5])
def test_invalid_prediction_category_fails(category):
    validator = SimpleNamespace(jdict=[], degenerate_predictions=0)
    pred = {"bboxes": torch.tensor([[1, 2, 3, 4.0]]), "conf": torch.tensor([0.5]), "cls": torch.tensor([category])}
    with pytest.raises((ValueError, FloatingPointError)):
        VisDroneValidator.pred_to_json(validator, pred, {"im_file": "id.jpg"})


def test_benchmark_summary_update_coverage_and_retry_deltas(tmp_path, monkeypatch):
    from scripts.d1 import run_e2
    from scripts.d1.run_wp8_p1_control import write_json

    monkeypatch.setattr(run_e2, "matrix", lambda _: {"identity": {"commit": "a" * 40}})
    root = tmp_path / "reports/E2-benchmark"
    for epoch in (1, 2, 3):
        write_json(root / "validation" / f"epoch-{epoch:03d}.json", {"seen": 548, "epoch_wall_seconds": 20 + epoch})
        for rank in range(6):
            write_json(
                root / "epochs" / f"rank-{rank}-epoch-{epoch:03d}.json",
                {
                    "batches": 68,
                    "optimizer_steps": epoch * 68,
                    "data_wait_seconds": 1,
                    "peak_allocated_bytes": 1024,
                    "amp_retries": 1,
                },
            )
    report = benchmark_summary(tmp_path)
    assert report["median_epoch_seconds"] == 22.5
    assert [row["amp_retries"] for row in report["epochs"]] == [6, 0, 0]
    assert report["estimated_300_epoch_gpu_hours"] == 11.25
    write_json(root / "validation/epoch-003.json", {"seen": 547, "epoch_wall_seconds": 1})
    with pytest.raises(ValueError, match="coverage"):
        benchmark_summary(tmp_path)

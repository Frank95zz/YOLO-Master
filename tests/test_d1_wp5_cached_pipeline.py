"""Tests for the D1 WP5 cache Dataset, Trainer, and Validator pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os

import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.d1_cache import (
    D1_FEATURE_NAMES,
    D1_FEATURE_SHAPE,
    D1FeatureBatch,
    D1FeatureCacheDataset,
)
from ultralytics.models.yolo.detect import (
    D1FoundationDetectionTrainer,
    D1FoundationDetectionValidator,
)
from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.foundation.cache import FeatureCacheReader, FeatureCacheWriter, sha256_bytes
from ultralytics.utils import DEFAULT_CFG_DICT, YAML


def cache_contract(**updates):
    contract = {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-dinov3-vits16",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": list(D1_FEATURE_NAMES),
        "dtype": "float16",
        "expected_shape": list(D1_FEATURE_SHAPE),
    }
    contract.update(updates)
    return contract


def sample_features(value: float) -> dict[str, torch.Tensor]:
    return {
        name: torch.full(D1_FEATURE_SHAPE, value + index, dtype=torch.float32)
        for index, name in enumerate(D1_FEATURE_NAMES)
    }


def data_config() -> dict:
    return {"names": {index: str(index) for index in range(80)}, "nc": 80, "channels": 3}


def hyp_config() -> SimpleNamespace:
    return SimpleNamespace(**DEFAULT_CFG_DICT)


def build_fixture(tmp_path: Path, count: int = 2) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "coco"
    image_dir = data_root / "images" / "train2017"
    label_dir = data_root / "labels" / "train2017"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    paths = []
    with FeatureCacheWriter(
        cache_dir,
        split="train2017",
        contract=cache_contract(),
        target_shard_bytes=16 * 1024**2,
    ) as writer:
        for index in range(1, count + 1):
            image_id = f"{index:012d}"
            image_path = image_dir / f"{image_id}.jpg"
            height, width = ((100, 200) if index % 2 else (200, 100))
            assert cv2.imwrite(str(image_path), np.full((height, width, 3), index, dtype=np.uint8))
            (label_dir / f"{image_id}.txt").write_text("0 0.5 0.5 0.5 0.4\n", encoding="utf-8")
            paths.append(str(image_path))
            writer.add(
                sample_id=f"train2017/{image_id}",
                split="train2017",
                image_path=f"images/train2017/{image_id}.jpg",
                image_sha256=sha256_bytes(f"image-{index}".encode()),
                features=sample_features(float(index)),
            )
    split_file = tmp_path / "train100.txt"
    split_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return data_root, split_file, cache_dir


def make_dataset(tmp_path: Path, **kwargs) -> D1FeatureCacheDataset:
    _data_root, split_file, cache_dir = build_fixture(tmp_path)
    return D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
        **kwargs,
    )


def test_cache_dataset_maps_sample_id_without_loading_rgb(tmp_path, monkeypatch) -> None:
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(dataset, "load_image", lambda *_args, **_kwargs: pytest.fail("RGB image was loaded"))

    sample = dataset[0]

    assert sample["sample_id"] == "train2017/000000000001"
    assert tuple(sample["features"]) == D1_FEATURE_NAMES
    assert all(value.dtype == torch.float16 for value in sample["features"].values())
    assert all(tuple(value.shape) == D1_FEATURE_SHAPE for value in sample["features"].values())
    assert sample["ori_shape"] == (100, 200)
    assert sample["resized_shape"] == (640, 640)
    assert sample["ratio_pad"] == ((3.2, 3.2), (0, 160))
    assert torch.allclose(sample["bboxes"], torch.tensor([[0.5, 0.5, 0.5, 0.2]]))
    assert torch.equal(sample["cls"], torch.tensor([[0.0]]))



def test_trusted_cache_checks_each_shard_only_once_per_worker(tmp_path, monkeypatch) -> None:
    dataset = make_dataset(tmp_path, trusted_cache=True, max_open_shards=2, prefetch_factor=1)
    original = torch.isfinite
    calls = 0

    def counted_isfinite(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(torch, "isfinite", counted_isfinite)
    first = dataset[0]["features"]
    repeated = dataset[0]["features"]

    assert calls == len(D1_FEATURE_NAMES)
    assert all(torch.equal(first[name], repeated[name]) for name in D1_FEATURE_NAMES)
    assert dataset.feature_reader.max_open_shards == 2
    assert dataset.prefetch_factor == 1
def test_collate_exposes_virtual_640_shape_without_rgb_tensor(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert isinstance(batch["features"], D1FeatureBatch)
    assert batch["img"] is batch["features"]
    assert batch["img"].shape == torch.Size((2, 3, 640, 640))
    assert all(tuple(value.shape) == (2, *D1_FEATURE_SHAPE) for value in batch["features"].values())
    assert torch.equal(batch["batch_idx"], torch.tensor([0.0, 1.0]))
    assert batch["sample_id"] == ("train2017/000000000001", "train2017/000000000002")


def test_trainer_preprocess_moves_features_without_rgb_division(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    trainer = object.__new__(D1FoundationDetectionTrainer)
    trainer.device = torch.device("cpu")
    trainer.amp = False

    result = trainer.preprocess_batch(batch)

    assert result["features"] is result["img"]
    assert result["features"]["block4"].dtype == torch.float32
    assert torch.all(result["features"]["block4"][0] == 1.0)
    assert torch.all(result["features"]["block12"][0] == 3.0)
    assert result["cls"].device.type == "cpu"


def test_validator_uses_virtual_size_and_restores_original_coordinates(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batch = dataset.collate_fn([dataset[0]])
    validator = object.__new__(D1FoundationDetectionValidator)
    validator.device = torch.device("cpu")
    validator.args = SimpleNamespace(quantize=None)
    prepared = validator.preprocess(batch)

    validator.device = torch.device("cpu")
    target = validator._prepare_batch(0, prepared)
    prediction = {
        "bboxes": torch.tensor([[160.0, 256.0, 480.0, 384.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
    }
    scaled = validator.scale_preds(prediction, target)

    assert target["imgsz"] == torch.Size((640, 640))
    assert torch.allclose(target["bboxes"], prediction["bboxes"])
    assert torch.allclose(scaled["bboxes"], torch.tensor([[50.0, 30.0, 150.0, 70.0]]), atol=1e-4)


def test_online_mode_is_explicit_and_can_compare_with_cache(tmp_path) -> None:
    _data_root, split_file, cache_dir = build_fixture(tmp_path)
    reader = FeatureCacheReader(cache_dir)

    def provider(im_file: str):
        return reader.get(f"train2017/{Path(im_file).stem}")

    dataset = D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        feature_mode="online",
        online_feature_provider=provider,
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
    )

    assert dataset.feature_mode == "online"
    assert dataset.compare_online_with_cache(0) == {name: 0.0 for name in D1_FEATURE_NAMES}
    assert torch.equal(dataset[0]["features"]["block8"], reader.get(dataset.sample_ids[0])["block8"])


def test_online_mismatch_and_invalid_contract_fail_fast(tmp_path) -> None:
    _data_root, split_file, cache_dir = build_fixture(tmp_path)
    reader = FeatureCacheReader(cache_dir)

    def changed_provider(im_file: str):
        values = reader.get(f"train2017/{Path(im_file).stem}")
        values["block8"] = values["block8"] + 1
        return values

    dataset = D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        feature_mode="online",
        online_feature_provider=changed_provider,
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
    )
    with pytest.raises(ValueError, match="block8 differ"):
        dataset.compare_online_with_cache(0)

    bad_cache = tmp_path / "bad-cache"
    with FeatureCacheWriter(
        bad_cache,
        split="train2017",
        contract=cache_contract(expected_shape=[384, 20, 20]),
    ):
        pass
    with pytest.raises(ValueError, match="expected_shape"):
        D1FeatureCacheDataset(
            img_path=str(split_file),
            cache_dir=bad_cache,
            data=data_config(),
            imgsz=640,
            batch_size=2,
            hyp=hyp_config(),
        )


def test_missing_cache_sample_and_non_coco_parent_fail_fast(tmp_path) -> None:
    _data_root, split_file, cache_dir = build_fixture(tmp_path, count=1)
    extra = tmp_path / "coco" / "images" / "train2017" / "000000000002.jpg"
    extra.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(extra), np.zeros((32, 32, 3), dtype=np.uint8))
    label = tmp_path / "coco" / "labels" / "train2017" / "000000000002.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("", encoding="utf-8")
    split_file.write_text(split_file.read_text(encoding="utf-8") + f"{extra}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing 1 dataset samples"):
        D1FeatureCacheDataset(
            img_path=str(split_file),
            cache_dir=cache_dir,
            data=data_config(),
            imgsz=640,
            batch_size=2,
            hyp=hyp_config(),
        )


def test_feature_batch_rejects_bad_shape_dtype_or_keys() -> None:
    good = {name: torch.zeros(1, *D1_FEATURE_SHAPE) for name in D1_FEATURE_NAMES}
    with pytest.raises(ValueError, match="keys"):
        D1FeatureBatch({"block4": good["block4"]})
    bad = dict(good)
    bad["block8"] = torch.zeros(1, 384, 20, 20)
    with pytest.raises(ValueError, match="shape"):
        D1FeatureBatch(bad)
    bad = dict(good)
    bad["block12"] = torch.zeros(1, *D1_FEATURE_SHAPE, dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        D1FeatureBatch(bad)


def test_real_wp2_cache_dataset_trainer_and_model(tmp_path) -> None:
    cache_value = os.environ.get("D1_WP2_CACHE")
    data_value = os.environ.get("D1_COCO_ROOT")
    if not cache_value or not data_value:
        pytest.skip("D1_WP2_CACHE and D1_COCO_ROOT are not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    cache_dir, data_root = Path(cache_value), Path(data_value)
    reader = FeatureCacheReader(cache_dir)
    records = [reader.records[sample_id] for sample_id in sorted(reader.records)[:2]]
    paths = [str(data_root / record["image_path"]) for record in records]
    split_file = tmp_path / "real-cache.txt"
    split_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
    dataset = D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
    )
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    trainer = object.__new__(D1FoundationDetectionTrainer)
    trainer.device = torch.device("cuda")
    trainer.amp = True
    batch = trainer.preprocess_batch(batch)
    model = D1FoundationDetectionModel().cuda().eval()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        predictions, _raw = model(batch["img"])

    assert batch["features"]["block4"].dtype == torch.float16
    assert tuple(predictions.shape) == (2, 300, 6)
    assert torch.isfinite(predictions).all()


def test_real_wp2_cache_one_batch_train_and_validate(tmp_path) -> None:
    cache_value = os.environ.get("D1_WP2_CACHE")
    data_value = os.environ.get("D1_COCO_ROOT")
    if not cache_value or not data_value:
        pytest.skip("D1_WP2_CACHE and D1_COCO_ROOT are not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    cache_dir, data_root = Path(cache_value), Path(data_value)
    reader = FeatureCacheReader(cache_dir)
    records = [reader.records[sample_id] for sample_id in sorted(reader.records)[:2]]
    split_file = tmp_path / "real-train-val.txt"
    split_file.write_text(
        "\n".join(str(data_root / record["image_path"]) for record in records) + "\n",
        encoding="utf-8",
    )
    data_yaml = tmp_path / "coco-two.yaml"
    YAML.save(
        data_yaml,
        {
            "path": str(data_root),
            "train": str(split_file),
            "val": str(split_file),
            "names": data_config()["names"],
            "channels": 3,
        },
    )
    repo_root = Path(__file__).resolve().parents[1]
    experiment = YAML.load(
        repo_root / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml"
    )
    trainer = D1FoundationDetectionTrainer(
        overrides={
            **experiment,
            "data": str(data_yaml),
            "epochs": 1,
            "batch": 2,
            "workers": 0,
            "device": 0,
            "amp": False,
            "project": str(tmp_path / "runs"),
            "name": "wp5-one-batch",
            "exist_ok": True,
            "plots": False,
            "save": False,
            "val": True,
            "pretrained": False,
            "verbose": False,
            "close_mosaic": 0,
        },
        feature_caches={"train": cache_dir, "val": cache_dir},
    )

    trainer.train()

    assert trainer.optimizer_steps == 1
    assert trainer.loss_items.shape == torch.Size((7,))
    assert trainer.loss_names == (
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "latent_balance_loss",
        "latent_z_loss",
        "latent_aux_loss",
        "mixture_aux_loss",
    )
    assert trainer.validator.seen == 2
    csv_header = trainer.csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {
        "train/latent_balance_loss",
        "train/latent_z_loss",
        "train/latent_aux_loss",
        "train/mixture_aux_loss",
    }.issubset(csv_header)

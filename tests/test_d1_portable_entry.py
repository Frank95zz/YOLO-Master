"""Portable-entry regression gates; synthetic data only, no network downloads."""

import cv2
import numpy as np
import pytest
import torch

from scripts.d1 import prepare_wp0, train
from scripts.d1.cache_features import cache_contract
from scripts.d1.ema import D1ModelEMA
from ultralytics.models.yolo.detect.foundation_train import D1FoundationDetectionTrainer
from ultralytics.nn.foundation.cache import FeatureCacheWriter
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "data"
    caches = {}
    for split, value in (("train2017", 1), ("val2017", 2)):
        images, labels = data / "images" / split, data / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        name = f"{value:012d}"
        image = images / f"{name}.jpg"
        assert cv2.imwrite(str(image), np.full((40, 60, 3), value, dtype=np.uint8))
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        cache = tmp_path / split
        with FeatureCacheWriter(cache, split=split, contract=cache_contract(train.ROOT)) as writer:
            writer.add(
                sample_id=f"{split}/{name}",
                split=split,
                image_path=f"images/{split}/{name}.jpg",
                image_sha256=train.sha256_file(image),
                features={
                    n: torch.full((384, 40, 40), value / 10, dtype=torch.float16)
                    for n in ("block4", "block8", "block12")
                },
            )
        caches[split] = cache
    yaml = tmp_path / "data.yaml"
    YAML.save(
        yaml,
        {
            "path": str(data),
            "train": "images/train2017",
            "val": "images/val2017",
            "names": {i: str(i) for i in range(80)},
        },
    )
    args = train.parser().parse_args(
        [
            "train",
            "--data",
            str(yaml),
            "--train-cache",
            str(caches["train2017"]),
            "--val-cache",
            str(caches["val2017"]),
            "--output",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--batch",
            "1",
            "--epochs",
            "1",
            "--workers",
            "0",
            "--fp32",
            "--approved",
        ]
    )
    return args


def test_training_needs_approval_before_reading_data(inputs):
    inputs.approved = False
    inputs.data = None
    with pytest.raises(ValueError, match="approved"):
        train.input_contract(inputs)


def test_rejects_internal_ddp_launch(inputs, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    inputs.device = "0,1"
    with pytest.raises(ValueError, match="torchrun"):
        train.input_contract(inputs)


def test_output_claim_fails_closed(tmp_path):
    output = tmp_path / "run"
    train.claim_output(output, {"run": "one"})
    train.claim_output(output, {"run": "one"}, rank=1)
    with pytest.raises(FileExistsError):
        train.claim_output(output, {"run": "two"})
    with pytest.raises(RuntimeError):
        train.claim_output(output, {"run": "two"}, rank=1, timeout=0)
    with pytest.raises(RuntimeError):
        train.claim_output(tmp_path / "missing", {}, rank=1, timeout=0)


def test_input_contract_and_fresh_run_only(inputs):
    identity, overrides, _, caches = train.input_contract(inputs)
    assert identity["dataset"] == "coco"
    assert overrides["batch"] == overrides["nbs"] == 1
    assert overrides["resume"] is False
    assert not overrides["amp"]
    assert overrides["model"]["adapter"]["p5_mode"] == "bottleneck"
    assert set(caches) == {"train", "val"}
    inputs.dataset = "visdrone"
    with pytest.raises(ValueError, match="names"):
        train.input_contract(inputs)


@pytest.mark.parametrize("implementation", ("scalar-v1", "foreach-v1"))
def test_install_ema_after_setup(implementation, monkeypatch):
    model = train.construct_model("BN64")
    existing = ModelEMA(model, updates=27)
    trainer = object.__new__(train.CachedTrainer)
    trainer.ema_implementation = implementation

    def setup(self):
        self.model, self.ema = model, existing

    monkeypatch.setattr(D1FoundationDetectionTrainer, "_setup_train", setup)
    trainer._setup_train()
    assert trainer.ema.ema is existing.ema
    assert trainer.ema.updates == 27
    assert isinstance(trainer.ema, D1ModelEMA) == (implementation == "foreach-v1")


def test_checkpoint_roundtrip_and_rejection(tmp_path):
    model = train.construct_model("BN64")
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model, "epoch": 2}, checkpoint)
    restored, epoch = train.strict_checkpoint(checkpoint)
    assert epoch == 2
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    torch.save({"model": torch.nn.Linear(1, 1)}, checkpoint)
    with pytest.raises(TypeError, match="D1"):
        train.strict_checkpoint(checkpoint)


@pytest.mark.parametrize("dataset", ("coco", "visdrone"))
def test_independent_evaluation_uses_all_images(inputs, dataset):
    inputs.dataset = dataset
    nc = 80 if dataset == "coco" else 10
    data = YAML.load(inputs.data)
    data["names"] = {i: str(i) for i in range(nc)}
    YAML.save(inputs.data, data)
    model = train.construct_model("BN64", nc=nc)
    checkpoint = inputs.output.parent / "checkpoint.pt"
    torch.save({"model": model, "epoch": 3}, checkpoint)
    inputs.command = "evaluate"
    inputs.checkpoint = checkpoint
    train.run(inputs)
    import json

    report = json.loads((inputs.output / "evaluation.json").read_text())
    assert report["strict_reload"] and report["images"] == 1
    assert report["checkpoint_epoch_zero_based"] == 3
    assert report["official"] is None
    assert (inputs.output / "predictions.json").is_file()
    if dataset == "visdrone":
        assert report["visdrone_export"]["image_count"] == 1


def test_synthetic_one_epoch_completes_and_saves(inputs):
    train.run(inputs)
    assert (inputs.output / "completed.json").is_file()
    assert (inputs.output / "weights/last.pt").is_file()
    train.strict_checkpoint(inputs.output / "weights/last.pt")


def test_visdrone_export_preserves_ids_and_filters_only_zero_area():
    validator = object.__new__(train.ExportValidator)
    validator.dataset_kind = "visdrone"
    validator.degenerate_boxes_removed = 0
    validator.jdict = []
    validator.class_map = list(range(10))
    prediction = {
        "bboxes": torch.tensor([[1.0, 2.0, 4.0, 6.0], [3.0, 4.0, 3.0, 5.0]]),
        "conf": torch.tensor([0.8, 0.9]),
        "cls": torch.tensor([1.0, 2.0]),
    }
    validator.pred_to_json(prediction, {"im_file": "00012.jpg"})
    assert len(validator.jdict) == 1
    assert validator.jdict[0]["image_id"] == "00012"
    assert validator.jdict[0]["category_id"] == 1
    assert validator.degenerate_boxes_removed == 1
    prediction["conf"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        validator.pred_to_json(prediction, {"im_file": "00012.jpg"})


def test_coco_npy_conversion_preserves_source_and_is_repeatable(inputs):
    from scripts.d1.npy import convert_preserving_source
    from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader

    source = inputs.train_cache
    before = {p.name: train.sha256_file(p) for p in source.iterdir() if p.is_file()}
    output = inputs.output.parent / "npy"
    first = convert_preserving_source(source, output)
    assert first == convert_preserving_source(source, output)
    assert before == {p.name: train.sha256_file(p) for p in source.iterdir() if p.is_file()}
    reader = NpyFeatureCacheReader(output / "train2017")
    for sample in reader.records:
        reader.verify_sample(sample)


def test_ddp_batch_must_be_divisible(inputs, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    inputs.device = "0,1"
    inputs.batch = 3
    with pytest.raises(ValueError, match="divisible"):
        train.input_contract(inputs)


def test_preparation_keeps_published_manifest_unchanged(tmp_path, monkeypatch):
    repo, workspace = tmp_path / "repo", tmp_path / "work"
    manifest = repo / "experiments/d1/manifests"
    manifest.mkdir(parents=True)
    published = {"splits": {}, "labels": {"train2017": 1, "val2017": 1}}
    original = {"coco2017-splits.json": train.canonical_json_bytes(published)}
    for name, data in original.items():
        (manifest / name).write_bytes(data)
    monkeypatch.setattr(prepare_wp0, "verify_contract", lambda *a: None)
    monkeypatch.setattr(
        prepare_wp0,
        "validated_splits",
        lambda *a: {split: [f"images/{split}/{split}.jpg"] for split in ("train2017", "val2017")},
    )
    monkeypatch.setattr(prepare_wp0, "verify_labels", lambda *a: published["labels"])
    monkeypatch.setattr(prepare_wp0, "verify_model", lambda *a: {"files": {}})
    report = prepare_wp0.verify_inputs(workspace / "data", workspace / "weights", workspace, repo=repo)
    assert {p.name: p.read_bytes() for p in manifest.iterdir()} == original
    assert (workspace / "verification.json").is_file()
    assert report["source_archives_verified"] is False


def test_official_coco_handles_empty_predictions(tmp_path):
    pytest.importorskip("faster_coco_eval")
    import json

    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "height": 40, "width": 60}],
                "categories": [{"id": 1, "name": "test"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0}
                ],
            }
        )
    )
    report = train.official_coco(annotations, [], [1])
    assert report["metrics"]["AP"] == 0
    with pytest.raises(ValueError, match="absent"):
        train.official_coco(annotations, [], [99])

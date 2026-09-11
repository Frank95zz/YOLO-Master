"""Portable training/evaluation entry points and exact optional EMA updates."""

import cv2
import numpy as np
import pytest
import torch
from scripts.d1 import prepare_coco, train
from scripts.d1.cache_features import cache_contract
from scripts.d1.ema import D1ModelEMA
from ultralytics.models.yolo.detect.foundation_train import D1FoundationDetectionTrainer
from ultralytics.nn.foundation.cache import FeatureCacheWriter
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA
from copy import deepcopy
from types import SimpleNamespace
from scripts.d1 import train as p5
from scripts.d1.ema import configure_d1_ema, validate_ema_implementation


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
    from scripts.d1.convert_npy import convert_preserving_source
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
    monkeypatch.setattr(prepare_coco, "verify_contract", lambda *a: None)
    monkeypatch.setattr(
        prepare_coco,
        "validated_splits",
        lambda *a: {split: [f"images/{split}/{split}.jpg"] for split in ("train2017", "val2017")},
    )
    monkeypatch.setattr(prepare_coco, "verify_labels", lambda *a: published["labels"])
    monkeypatch.setattr(prepare_coco, "verify_model", lambda *a: {"files": {}})
    report = prepare_coco.verify_inputs(workspace / "data", workspace / "weights", workspace, repo=repo)
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


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("variant", p5.VARIANTS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.float16, torch.bfloat16))
def test_exact_update_and_persistent_buffer_rules(variant, dtype):
    model = p5.construct_model(variant).to(dtype=dtype)
    model.register_buffer("test_float", torch.randn(7, dtype=dtype))
    model.register_buffer("test_int", torch.tensor(4))
    model.register_buffer("test_bool", torch.tensor(False))
    model.register_buffer("test_temporary", torch.tensor(1.0), persistent=False)
    # Include heterogeneous source dtypes and empty tensors in the same update.
    model.register_buffer("test_double", torch.ones(3, dtype=torch.float64))
    model.register_buffer("test_empty", torch.empty(0))
    original = ModelEMA(model, updates=100, decay=0.99, tau=30)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    for _ in range(3):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.002)
            model.test_float.mul_(1.5)
            model.test_int.add_(1)
            model.test_bool.logical_not_()
            model.test_temporary.add_(1)
        original.update(model)
        fast.update(model)
        assert fast.updates == original.updates
        assert_same(original.ema.state_dict(), fast.ema.state_dict())
        assert fast.ema.test_temporary.item() == 1
        assert not any(p.requires_grad for p in fast.ema.parameters())


def test_conversion_preserves_restored_state_and_is_local():
    model = p5.construct_model("BASE")
    existing = ModelEMA(model, updates=87)
    existing.enabled = False
    fast = configure_d1_ema(existing, model, "foreach-v1")
    assert fast.ema is existing.ema and fast.decay is existing.decay
    assert fast.updates == 87 and not fast.enabled
    fast.update(torch.nn.Linear(2, 2))
    assert fast.updates == 87
    assert type(existing) is ModelEMA
    assert configure_d1_ema(existing, model, "scalar-v1") is existing
    assert configure_d1_ema(None, model, "foreach-v1") is None
    with pytest.raises(TypeError):
        configure_d1_ema(fast, model, "foreach-v1")
    with pytest.raises(TypeError):
        configure_d1_ema(ModelEMA(torch.nn.Linear(2, 2)), model, "foreach-v1")
    with pytest.raises(TypeError):
        configure_d1_ema(existing, torch.nn.Linear(2, 2), "foreach-v1")


def test_dynamic_buffers_and_no_per_step_state_serialization(monkeypatch):
    model = p5.construct_model("BASE")
    original = ModelEMA(model)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    for obj in (model, original.ema, fast.ema):
        obj.register_buffer("late_buffer", torch.tensor(3.0))
    model.late_buffer = torch.tensor(5.0)
    original.update(model)
    monkeypatch.setattr(model, "state_dict", lambda *a, **k: pytest.fail("Serialized source during fast update"))
    fast.update(model)
    assert_same(original.ema.state_dict(), fast.ema.state_dict())


def test_hook_and_shape_changes_fail_before_partial_update():
    model = p5.construct_model("BASE")
    fast = configure_d1_ema(ModelEMA(model), model, "foreach-v1")
    hook = model.register_state_dict_pre_hook(lambda *a: None)
    with pytest.raises(ValueError, match="hooks"):
        fast.update(model)
    hook.remove()
    model._mixture_loss_ema_buf = torch.ones(100)
    before = deepcopy(fast.ema.state_dict())
    with pytest.raises(ValueError, match="shape/device"):
        fast.update(model)
    assert fast.updates == 0
    assert_same(before, fast.ema.state_dict())


@pytest.mark.parametrize("value", [None, True, "foreach", "typo", [], 1])
def test_invalid_registration(value):
    with pytest.raises(ValueError):
        validate_ema_implementation(value)


@pytest.mark.parametrize("variant", p5.VARIANTS)
def test_real_loss_optimizer_and_ema_exact(variant):
    torch.manual_seed(0)
    model = p5.construct_model(variant).train()
    original = ModelEMA(model)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = {
        "features": {key: torch.randn(1, 384, 4, 4) for key in model.adapter.source_names},
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss.sum().backward()
        optimizer.step()
        original.update(model)
        fast.update(model)
        assert_same(original.ema.state_dict(), fast.ema.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA; no download")
@pytest.mark.parametrize("variant", p5.VARIANTS)
def test_cuda_amp_overflow_and_resume(variant):
    from ultralytics.engine.extensions.recovery import TrainingRecoveryController
    from ultralytics.engine.trainer import BaseTrainer

    model = p5.construct_model(variant).cuda().train()
    ema = configure_d1_ema(ModelEMA(model), model, "foreach-v1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=16)
    trainer = SimpleNamespace(
        model=model, ema=ema, optimizer=optimizer, scaler=scaler, optimizer_steps=0, adapter_controller=None
    )
    trainer._recovery_controller = lambda: TrainingRecoveryController(trainer)
    trainer._sync_nonfinite_flag = lambda bad: bool(bad)
    before = deepcopy(ema.ema.state_dict())
    scaler.scale(next(model.parameters()).square().mean()).backward()
    next(model.parameters()).grad.reshape(-1)[0] = float("inf")
    assert not BaseTrainer.optimizer_step(trainer)
    assert trainer.optimizer_steps == ema.updates == 0 and scaler.get_scale() == 8
    assert_same(before, ema.ema.state_dict())
    reference = ModelEMA(model)
    reference.ema.load_state_dict(ema.ema.state_dict(), strict=True)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(next(model.parameters()).square().mean()).backward()
        assert BaseTrainer.optimizer_step(trainer)
        reference.update(model)
        assert_same(reference.ema.state_dict(), ema.ema.state_dict())
    restored = ModelEMA(model, updates=ema.updates)
    restored.ema.load_state_dict(ema.ema.state_dict(), strict=True)
    restored = configure_d1_ema(restored, model, "foreach-v1")
    restored.update(model)
    ema.update(model)
    assert_same(restored.ema.state_dict(), ema.ema.state_dict())

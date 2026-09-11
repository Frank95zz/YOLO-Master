"""Offline gates for the D1 P1 scratch preparation and runtime policy."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.d1 import run_wp8_p1_control as p1
from ultralytics.cfg import get_cfg
from ultralytics.data.d1_cache import _letterbox_boxes
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import one_cycle


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DetectionModel(p1.ROOT / p1.MODEL, nc=80, ch=3, verbose=False)


def test_registered_contract():
    contract = p1.load_contract()
    assert contract["window_epochs"] == 30
    assert contract["train"]["epochs"] == 100
    assert contract["train"]["batch"] == contract["train"]["nbs"] == 384
    assert "/data/" not in p1.CONFIG.read_text()
    assert "/root/" not in p1.CONFIG.read_text()
    args = get_cfg(overrides=contract["train"])
    assert args.optimizer == "AdamW" and args.pretrained is False
    assert all(getattr(args, key) == 0 for key in p1.AUGMENTATIONS)


@pytest.mark.parametrize("key,value", [
    ("epochs", 30), ("pretrained", "weights.pt"), ("freeze", [0]), ("optimizer", "auto"),
    ("batch", 16), ("nbs", 64), ("rect", True), ("scale", 0.5), ("cache", "ram"),
    ("warmup_bias_lr", 0.01), ("amp", False), ("foundation_enabled", True),
])
def test_reject_contract_drift(tmp_path, key, value):
    contract = YAML.load(p1.CONFIG)
    contract["train"][key] = value
    path = tmp_path / "contract.yaml"
    YAML.save(path, contract)
    with pytest.raises(ValueError, match="Unregistered"):
        p1.load_contract(path)


def test_exact_budget_and_real_pyramid(model):
    audit = p1.audit_model(model)
    assert audit["trainable_parameters"] == 3510624
    assert abs(audit["parameter_delta_fraction"]) < 0.01
    outputs = {}
    handles = []
    for index in (16, 19, 22):
        handles.append(model.model[index].register_forward_hook(
            lambda module, args, output, index=index: outputs.update({index: tuple(output.shape)})))
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, 640, 640))
    finally:
        for handle in handles:
            handle.remove()
    assert outputs == {16: (1, 72, 80, 80), 19: (1, 136, 40, 40), 22: (1, 272, 20, 20)}


def test_parameter_freezing_is_not_allowed(model):
    parameter = next(model.parameters())
    parameter.requires_grad_(False)
    try:
        with pytest.raises(ValueError, match="budget"):
            p1.audit_model(model)
    finally:
        parameter.requires_grad_(True)


@pytest.mark.parametrize("shape", [(333, 500), (901, 317), (127, 129), (640, 640)])
def test_pixels_boxes_and_inverse_geometry_match_wp0(tmp_path, shape):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[..., :] = (20, 60, 180)
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), image)
    dataset = p1.ScratchDataset.__new__(p1.ScratchDataset)
    dataset.im_files = [str(image_path)]
    dataset.rect = False
    dataset.use_obb = False
    original_boxes = np.array([[0.5, 0.5, 0.25, 0.4]], dtype=np.float32)
    dataset.labels = [{
        "im_file": str(image_path), "shape": shape, "cls": np.array([[0]], dtype=np.float32),
        "bboxes": original_boxes, "segments": [], "normalized": True, "bbox_format": "xywh",
    }]
    dataset.transforms = dataset.build_transforms()
    sample = dataset[0]
    boxes, ratio_pad = _letterbox_boxes(original_boxes, shape, 640)
    assert sample["img"].shape == (3, 640, 640)
    torch.testing.assert_close(sample["bboxes"], boxes, rtol=0, atol=1e-6)
    assert sample["ratio_pad"] == ratio_pad
    expected = p1.LetterBox((640, 640), auto=False, scaleup=True, stride=32)(image=image)
    rgb = torch.from_numpy(np.ascontiguousarray(expected[..., ::-1].transpose(2, 0, 1)))
    assert torch.equal(sample["img"], rgb)
    assert torch.equal(dataset[0]["img"], sample["img"])
    assert "features" not in sample


def test_rgb_div255_not_dino_normalization():
    trainer = p1.ScratchTrainer.__new__(p1.ScratchTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0)
    batch = trainer.preprocess_batch({"img": torch.tensor([0, 114, 255], dtype=torch.uint8)})
    torch.testing.assert_close(batch["img"], torch.tensor([0, 114 / 255, 1]))


def test_100_epoch_lr_and_loss_schedule(model):
    model.args = get_cfg(overrides=p1.load_contract()["train"])
    criterion = E2ELoss(model)
    for _ in range(29):
        criterion.update()
    assert criterion.o2m == pytest.approx(0.8 - 0.7 * 29 / 99)
    assert criterion.one2one.hyp.epochs == 100
    lr = 0.001 * one_cycle(1, 0.01, 100)(29)
    assert lr == pytest.approx(0.000808389, abs=1e-9)
    assert lr > 10 * 0.001 * one_cycle(1, 0.01, 30)(29)


@pytest.mark.parametrize("epoch,expected", [(0, False), (28, False), (29, True), (30, True)])
def test_stop_keeps_schedule_horizon(epoch, expected):
    trainer = SimpleNamespace(epoch=epoch, stop=False, epochs=100, args=SimpleNamespace(epochs=100))
    p1.stop_at_window(trainer)
    assert trainer.stop is expected
    assert trainer.epochs == trainer.args.epochs == 100


def test_strict_checkpoint_roundtrip(tmp_path, model):
    path = tmp_path / "last.pt"
    torch.save({"model": deepcopy(model).half(), "ema": None, "epoch": 29, "optimizer": {}}, path)
    report = p1.strict_checkpoint(path)
    assert report["strict_reload"] and report["has_optimizer"]
    assert report["epoch"] == 29


def test_scratch_rejects_pretrained(model):
    trainer = p1.ScratchTrainer.__new__(p1.ScratchTrainer)
    trainer.resume = False
    with pytest.raises(ValueError, match="Pretrained"):
        trainer.get_model(p1.ROOT / p1.MODEL, weights=model)


def test_published_split_contract_is_complete():
    manifest = json.loads((p1.ROOT / "experiments/d1/manifests/coco2017-splits.json").read_text())
    assert {name: entry["count"] for name, entry in manifest["splits"].items()} == p1.SPLITS
    assert manifest["labels"]["train2017"] == 117266


@pytest.mark.parametrize("failure", [None, "checksum", "overlap"])
def test_generated_split_lists_are_validated(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(p1, "SPLITS", {"train2017": 2, "val2017": 1})
    splits = {"train2017": ["000001", "000002"], "val2017": ["000003"]}
    if failure == "overlap":
        splits["val2017"] = ["000001"]
    manifest = {"splits": {}, "labels": {"train2017": 2, "val2017": 1}}
    for split, ids in splits.items():
        path = tmp_path / f"coco2017-{split}.txt"
        path.write_text("".join(f"images/{split}/{image_id}.jpg\n" for image_id in ids))
        manifest["splits"][split] = {"path": path.name, "count": len(ids), "sha256": p1.sha256_file(path)}
    if failure == "checksum":
        manifest["splits"]["val2017"]["sha256"] = "0" * 64
    (tmp_path / "coco2017-splits.json").write_text(json.dumps(manifest))
    if failure:
        with pytest.raises(ValueError, match="checksum|overlap"):
            p1.read_splits(tmp_path)
    else:
        paths, evidence = p1.read_splits(tmp_path)
        assert {name: len(values) for name, values in paths.items()} == p1.SPLITS
        assert evidence["train2017"]["label_files"] == 2


def test_copy_not_published_is_rejected(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"phase": "COPYING", "destination": str(tmp_path)}))
    with pytest.raises(ValueError, match="not been published"):
        p1.copy_ready(path, tmp_path)


@pytest.mark.parametrize("workers", [0, 2, 16, True])
def test_invalid_workers(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        p1.training_overrides(p1.load_contract(), tmp_path / "coco.yaml", tmp_path / "run", workers)


def test_valid_overrides(tmp_path):
    overrides = p1.training_overrides(p1.load_contract(), tmp_path / "coco.yaml", tmp_path / "run", 8)
    args = get_cfg(overrides=overrides)
    assert args.workers == 8
    assert args.epochs == 100
    assert args.device == "0,1,2,3,4,5"
    assert args.pretrained is False


def test_no_implicit_training_approval():
    with pytest.raises(RuntimeError, match="explicit user approval"):
        p1.require_launch_approval(False, {}, {}, {})


@pytest.mark.parametrize("change", ["dirty", "missing_backward", "unmatched_report", "workers"])
def test_training_gate_fails_closed(change):
    identity = {"commit": "a" * 40, "dirty": False}
    preparation = {"identity": deepcopy(identity), "status": "data_ready_runtime_pending",
                   "overrides": {"workers": 4}, "_file_sha256": "b" * 64}
    gate = {"identity": deepcopy(identity), "status": "passed", "workers": 4,
            "preparation_sha256": "b" * 64, "six_gpu_worker_benchmark": True,
            "real_batch_backward": True, "strict_checkpoint_reload": True, "evaluation_ready": True}
    if change == "dirty":
        identity["dirty"] = True
    elif change == "missing_backward":
        gate.pop("real_batch_backward")
    elif change == "unmatched_report":
        gate["preparation_sha256"] = "c" * 64
    else:
        gate["workers"] = 8
    with pytest.raises(RuntimeError):
        p1.require_launch_approval(True, preparation, gate, identity)


def test_atomic_report_rejects_nonfinite(tmp_path):
    path = tmp_path / "report.json"
    p1.write_json(path, {"valid": 1})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        p1.write_json(path, {"value": float("nan")})
    assert path.read_bytes() == before

def test_scratch_detection_loss_backward_and_update(model):
    student = deepcopy(model).train()
    student.args = get_cfg(overrides=p1.load_contract()["train"])
    student.criterion = None
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.001)
    batch = {
        "img": torch.rand(2, 3, 96, 96), "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4]]),
    }
    loss, items = student(batch)
    assert isinstance(student.criterion, E2ELoss)
    assert items.shape == (3,) and torch.isfinite(items).all()
    loss.sum().backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad) > 0 for grad in grads)
    parameter = next(student.parameters())
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter)


def test_prepare_model_only_never_reads_dataset_or_starts_training(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model-only preparation accessed data or training")
    monkeypatch.setattr(p1, "copy_ready", forbidden)
    monkeypatch.setattr(p1, "read_splits", forbidden)
    monkeypatch.setattr(p1.ScratchTrainer, "train", forbidden)
    args = SimpleNamespace(model_only=True, output_dir=tmp_path, run_root=tmp_path / "unused",
                           data_root=None, copy_receipt=None, workers=4)
    report = p1.prepare(args)
    assert report["status"] == "model_ready_runtime_pending"
    assert report["formal_training_approved"] is False
    assert report["split_lists_verified"] is False
    assert report["splits"] == {}
    assert report["model"]["trainable_parameters"] == 3510624
    assert not args.run_root.exists()
    assert (tmp_path / "preparation.json").is_file()

def test_launch_claim_does_not_overwrite_an_existing_run(tmp_path):
    root = tmp_path / "run"
    p1.claim_run(root, {"commit": "test"}, "launch-a", 0)
    result = root / "results.csv"
    result.write_text("important results")
    with pytest.raises(FileExistsError):
        p1.claim_run(root, {"commit": "test"}, "launch-b", 0)
    assert result.read_text() == "important results"


def test_peer_must_match_current_launch(tmp_path):
    root = tmp_path / "run"
    p1.claim_run(root, {"commit": "test"}, "launch-a", 0)
    p1.claim_run(root, {"commit": "test"}, "launch-a", 1)
    with pytest.raises(RuntimeError, match="different launch"):
        p1.claim_run(root, {"commit": "test"}, "launch-b", 1)


def test_launch_requires_fresh_rendezvous(tmp_path):
    with pytest.raises(RuntimeError, match="standalone"):
        p1.claim_run(tmp_path / "run", {}, "none", 0)
    assert not (tmp_path / "run").exists()

class FakeScale:
    def __init__(self):
        self.value = 16.0

    def get_scale(self):
        return self.value

    def update(self, new_scale=None):
        self.value = self.value / 2 if new_scale is None else new_scale

    def scale(self, loss):
        return loss


class RetryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, batch):
        loss = (self.bn(batch["img"]) * self.weight).square().mean() * torch.rand(())
        return loss.unsqueeze(0), loss.detach().unsqueeze(0)


def retry_trainer(monkeypatch, always_fail=False):
    monkeypatch.setenv("RANK", "-1")
    trainer = p1.ScratchTrainer.__new__(p1.ScratchTrainer)
    trainer.device = torch.device("cpu")
    trainer.amp = True
    trainer.args = SimpleNamespace(multi_scale=0)
    trainer.model = RetryModel().train()
    trainer.scaler = FakeScale()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.optimizer_steps = 0
    trainer.attempts = 0

    def parent_step(self):
        self.attempts += 1
        if self.attempts == 1 or always_fail:
            self.model.zero_grad()
            self.scaler.update()
            return False
        assert all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.optimizer_steps += 1
        return True

    monkeypatch.setattr(p1.DetectionTrainer, "optimizer_step", parent_step)
    batch = trainer.preprocess_batch({"img": torch.randint(0, 256, (2, 1, 8, 8), dtype=torch.uint8)})
    loss, trainer.loss_items = trainer.model(batch)
    trainer.loss = loss.sum()
    trainer.loss.backward()
    return trainer


def test_amp_retry_preserves_batchnorm_rng_and_exactly_one_update(monkeypatch):
    trainer = retry_trainer(monkeypatch)
    buffers = {name: value.clone() for name, value in trainer.model.named_buffers()}
    rng = torch.get_rng_state()
    assert trainer.optimizer_step()
    assert trainer.optimizer_steps == 1
    assert trainer.amp_retry_count == 1
    assert trainer.scaler.get_scale() == 8
    assert trainer.model.bn.num_batches_tracked.item() == 1
    for name, value in trainer.model.named_buffers():
        torch.testing.assert_close(value, buffers[name])
    assert torch.equal(torch.get_rng_state(), rng)
    assert trainer._amp_retry_state is None
    assert trainer._amp_retry_batch is None


def test_amp_retry_is_bounded_and_never_updates_on_failure(monkeypatch):
    trainer = retry_trainer(monkeypatch, always_fail=True)
    with pytest.raises(FloatingPointError, match="bounded AMP"):
        trainer.optimizer_step()
    assert trainer.optimizer_steps == 0
    assert trainer.attempts == 9
    assert trainer.amp_retry_count == 8

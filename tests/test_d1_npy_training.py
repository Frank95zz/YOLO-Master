"""Production NPY reader, provenance, pinned-memory alias, and resume regressions."""

from __future__ import annotations

import copy
import json
import os
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data._utils.pin_memory import pin_memory

from ultralytics.data.d1_cache import D1FeatureCacheDataset, D1TrainingBatch
from ultralytics.models.yolo.detect.foundation_train import D1FoundationDetectionTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.foundation.cache import FeatureCacheReader, FeatureCacheWriter, sha256_file
from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader, open_feature_cache, validate_npy_evidence
from ultralytics.utils import DEFAULT_CFG_DICT

NAMES = ["block4", "block8", "block12"]


@pytest.fixture
def converted(tmp_path):
    split = "train2017"
    source = tmp_path / "source"
    root = tmp_path / "npy"
    target = root / split
    target.mkdir(parents=True)
    contract = {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-teacher",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "feature_names": NAMES,
        "output_layers": [4, 8, 12],
        "expected_shape": [384, 40, 40],
        "dtype": "float16",
    }
    data = tmp_path / "coco"
    (data / "images" / split).mkdir(parents=True)
    (data / "labels" / split).mkdir(parents=True)
    with FeatureCacheWriter(source, split=split, contract=contract) as writer:
        for i in (1, 2):
            sid = f"{split}/{i:012d}"
            image_path = data / "images" / f"{sid}.jpg"
            assert cv2.imwrite(str(image_path), np.full((80, 120, 3), i, dtype=np.uint8))
            (data / "labels" / f"{sid}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
            writer.add(
                sample_id=sid,
                split=split,
                image_path=f"images/{sid}.jpg",
                image_sha256=sha256_file(image_path),
                features={name: torch.full((384, 40, 40), i + j, dtype=torch.float16) for j, name in enumerate(NAMES)},
            )
    reader = FeatureCacheReader(source)
    provenance = root / "provenance" / split
    provenance.mkdir(parents=True)
    for name in ("index.json", "samples.jsonl"):
        shutil.copyfile(source / name, provenance / name)
    records = []
    for sid, original in reader.records.items():
        array = np.stack([reader.get(sid)[name].numpy() for name in NAMES])
        path = root / f"{sid}.npy"
        np.save(path, array, allow_pickle=False)
        record = copy.deepcopy(original)
        record["source_shard"] = record.pop("shard")
        record.update(npy_path=f"{sid}.npy", npy_bytes=path.stat().st_size, npy_sha256=sha256_file(path))
        records.append(record)
    manifest = root / f"{split}-samples.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    index = {
        "schema_version": "d1-npy-cache-v1",
        "split": split,
        "sample_count": len(records),
        "shape": [3, 384, 40, 40],
        "dtype": "float16",
        "feature_names": NAMES,
        "contract": reader.contract,
        "contract_sha256": reader.index["contract_sha256"],
        "source_index_sha256": sha256_file(source / "index.json"),
        "source_content_sha256": reader.index["content_sha256"],
        "samples_manifest": manifest.name,
        "samples_manifest_sha256": sha256_file(manifest),
        "npy_bytes": sum(r["npy_bytes"] for r in records),
    }
    (root / f"{split}-index.json").write_text(json.dumps(index))
    (root / "receipts").mkdir()
    for shard in reader.index["shards"]:
        selected = [r for r in records if r["source_shard"] == shard["filename"]]
        receipt = {
            "schema_version": "d1-npy-cache-v1",
            "state": "VERIFIED",
            "source_index_sha256": index["source_index_sha256"],
            "source_shard": shard,
            "sample_count": len(selected),
            "verified_tensor_count": len(selected) * 3,
            "files": [
                {"sample_id": r["sample_id"], "path": r["npy_path"], "bytes": r["npy_bytes"], "sha256": r["npy_sha256"]}
                for r in selected
            ],
        }
        (root / "receipts" / f"{shard['filename']}.json").write_text(json.dumps(receipt))
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "d1-npy-cache-v1",
                "status": "COMPLETED",
                "source_indices_sha256": {split: index["source_index_sha256"]},
            }
        )
    )
    return source, target, data, summary


def test_reader_is_bitwise_equal_to_safetensors_and_picklable(converted):
    source, target, _, summary = converted
    original = FeatureCacheReader(source)
    reader = pickle.loads(pickle.dumps(open_feature_cache(target)))
    assert isinstance(reader, NpyFeatureCacheReader)
    assert isinstance(open_feature_cache(source), FeatureCacheReader)
    for sid in reader.records:
        assert tuple(reader.get(sid)) == tuple(NAMES)
        reader.verify_sample(sid)
        for name in NAMES:
            assert torch.equal(reader.get(sid)[name], original.get(sid)[name])
            assert not reader.get(sid)[name].requires_grad
    evidence = validate_npy_evidence(target, summary, "train2017", 2)
    assert evidence["sample_count"] == 2
    assert len(evidence["preflight_hashed_samples"]) == 2


@pytest.mark.parametrize("kind", ["missing", "part", "receipt", "source", "checksum", "summary"])
def test_preflight_fails_closed(converted, kind):
    _, target, _, summary = converted
    path = next(target.glob("*.npy"))
    if kind == "missing":
        path.unlink()
    elif kind == "part":
        (target / "unexpected.part").touch()
    elif kind == "receipt":
        p = next((target.parent / "receipts").glob("*.json"))
        doc = json.loads(p.read_text())
        doc["files"].pop()
        p.write_text(json.dumps(doc))
    elif kind == "source":
        p = target.parent / "provenance/train2017/index.json"
        p.write_text(p.read_text() + " ")
    elif kind == "checksum":
        with path.open("r+b") as stream:
            stream.seek(-2, 2)
            stream.write(b"\x00\x00")
    else:
        summary.write_text('{"status":"FAILED"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_npy_evidence(target, summary, "train2017", 2)


@pytest.mark.parametrize("kind", ["path", "duplicate", "order", "split", "hash", "shape"])
def test_reader_rejects_invalid_manifest(converted, kind):
    _, target, _, _ = converted
    index_path = target.parent / "train2017-index.json"
    index = json.loads(index_path.read_text())
    manifest = target.parent / index["samples_manifest"]
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    if kind == "path":
        records[0]["npy_path"] = "../outside.npy"
    elif kind == "duplicate":
        records[1] = records[0]
    elif kind == "order":
        records.reverse()
    elif kind == "split":
        records[0]["split"] = "val2017"
    elif kind == "hash":
        index["contract_sha256"] = "f" * 64
    else:
        index["shape"] = [3, 384, 20, 20]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    index["samples_manifest_sha256"] = sha256_file(manifest)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        open_feature_cache(target)


def test_reader_rejects_wrong_dtype_and_nan(converted):
    _, target, _, _ = converted
    reader = NpyFeatureCacheReader(target)
    sid = next(iter(reader.records))
    path = target.parent / reader.records[sid]["npy_path"]
    np.save(path, np.zeros((3, 384, 40, 40), dtype=np.float32))
    reader.records[sid]["npy_bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match="dtype"):
        reader.get(sid)
    np.save(path, np.full((3, 384, 40, 40), np.nan, dtype=np.float16))
    reader.records[sid].update(npy_bytes=path.stat().st_size, npy_sha256=sha256_file(path))
    with pytest.raises(ValueError, match="finite"):
        reader.verify_sample(sid)


def test_dataset_and_pin_memory_keep_one_feature_copy(converted, monkeypatch):
    source, target, data, _ = converted
    kwargs = {
        "img_path": str(data / "images/train2017"),
        "data": {"names": {i: str(i) for i in range(80)}, "nc": 80},
        "hyp": SimpleNamespace(**DEFAULT_CFG_DICT),
        "batch_size": 2,
    }
    old = D1FeatureCacheDataset(cache_dir=source, **kwargs)
    new = D1FeatureCacheDataset(cache_dir=target, **kwargs)
    batch = new.collate_fn([new[0], new[1]])
    reference = old.collate_fn([old[0], old[1]])
    calls = []

    def fake_pin(tensor, device=None):
        calls.append(tensor)
        return tensor.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin)
    pinned = pickle.loads(pickle.dumps(pin_memory(batch)))
    assert isinstance(pinned, D1TrainingBatch)
    assert "img" in pinned and pinned["img"] is pinned.get("features")
    assert "img" not in tuple(pinned.keys())
    assert len([t for t in calls if t.ndim == 4]) == 3
    for name in NAMES:
        assert torch.equal(pinned["features"][name], reference["features"][name])
    for key in ("cls", "bboxes", "batch_idx"):
        assert torch.equal(pinned[key], reference[key])
    assert torch.equal(pinned["bboxes"], reference["bboxes"])


def test_resume_does_not_reset_restored_amp_scaler(monkeypatch):
    trainer = object.__new__(D1FoundationDetectionTrainer)
    trainer.amp = True
    trainer.amp_init_scale = 16
    trainer.resume = "checkpoint.pt"
    restored = object()
    monkeypatch.setattr(DetectionTrainer, "_setup_train", lambda self: setattr(self, "scaler", restored))
    trainer._setup_train()
    assert trainer.scaler is restored


@pytest.mark.skipif(not os.environ.get("D1_NPY_CACHE"), reason="local NPY cache integration is opt-in")
def test_real_npy_model_backward_and_checkpoint(tmp_path):
    from ultralytics.nn import D1FoundationDetectionModel
    from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer

    root = Path(os.environ["D1_NPY_CACHE"])
    reader = NpyFeatureCacheReader(root)
    sid = next(iter(reader.records))
    reader.verify_sample(sid)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = D1FoundationDetectionModel().to(device).train()
    inputs = {k: v.unsqueeze(0).to(device=device, dtype=torch.float32) for k, v in reader.get(sid).items()}
    batch = {
        "img": inputs,
        "features": inputs,
        "batch_idx": torch.zeros(1, device=device),
        "cls": torch.zeros(1, 1, device=device),
        "bboxes": torch.full((1, 4), 0.5, device=device),
    }
    with torch.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
        loss, _ = model.loss(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)
    assert not any("teacher" in k.lower() for k in model.state_dict())
    state = tmp_path / "model.pt"
    torch.save(model.state_dict(), state)
    restored = D1FoundationDetectionModel().to(device)
    initialize_mixture_loss_ema_buffer(restored)
    restored.load_state_dict(torch.load(state, map_location=device, weights_only=True), strict=True)

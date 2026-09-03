"""Offline tests for the D1 WP8 formal-training contract and runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.d1 import run_wp8_train
from scripts.d1.run_wp8_train import (
    SCHEMA_VERSION,
    discover_cache_report,
    load_contract,
    training_overrides,
    validate_cache_evidence,
)
from ultralytics.nn.foundation.cache import FeatureCacheWriter, sha256_bytes, verify_feature_cache
from ultralytics.utils import YAML


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "ultralytics/cfg/experiments/d1/wp8-formal-coco2017.yaml"


def cache_contract():
    return {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-teacher",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": ["block4", "block8", "block12"],
        "dtype": "float16",
        "expected_shape": [384, 40, 40],
    }


def build_cache_and_report(tmp_path: Path, split: str) -> tuple[Path, Path]:
    cache_dir = tmp_path / split
    writer = FeatureCacheWriter(cache_dir, split=split, contract=cache_contract(), target_shard_bytes=4 * 1024**2)
    tensor = torch.zeros(384, 40, 40, dtype=torch.float16)
    writer.add(
        sample_id=f"{split}/000000000001",
        split=split,
        image_path=f"images/{split}/000000000001.jpg",
        image_sha256=sha256_bytes(b"image"),
        features={name: tensor for name in ("block4", "block8", "block12")},
    )
    writer.close()
    verification = verify_feature_cache(cache_dir)
    report_path = tmp_path / "manifests" / f"wp8-full-{split}-test.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "finalization": {"status": "passed", "split": split, "verification": verification},
            }
        ),
        encoding="utf-8",
    )
    return cache_dir, report_path


def test_formal_contract_locks_six_gpu_training_and_has_no_host_paths():
    contract = load_contract(CONFIG_PATH)

    assert contract["schema_version"] == SCHEMA_VERSION
    assert contract["hardware"] == {"world_size": 6, "devices": "0,1,2,3,4,5"}
    assert contract["cache_io"] == {
        "trusted": True,
        "max_open_shards_per_worker": 4,
        "prefetch_factor": 1,
    }
    assert contract["runtime"] == {"amp_init_scale": 16, "amp_growth_interval": 1_000_000}
    assert contract["train"]["batch"] == 48
    assert contract["train"]["batch"] // contract["hardware"]["world_size"] == 8
    assert contract["train"]["epochs"] == 100
    assert contract["train"]["amp"] is True
    assert contract["acceptance"]["accuracy_threshold"] is None
    serialized = CONFIG_PATH.read_text(encoding="utf-8")
    assert "/data/" not in serialized
    assert "/root/" not in serialized
    assert "10.210.22.36" not in serialized


def test_training_overrides_preserve_p0_geometry_and_lock_runtime(tmp_path):
    contract = load_contract(CONFIG_PATH)
    overrides = training_overrides(REPO_ROOT, contract, tmp_path / "data.yaml", tmp_path / "formal")

    assert overrides["device"] == "0,1,2,3,4,5"
    assert overrides["batch"] == 48
    assert overrides["nbs"] == 48
    assert overrides["mosaic"] == 0.0
    assert overrides["mixup"] == 0.0
    assert overrides["copy_paste"] == 0.0
    assert overrides["multi_scale"] == 0.0
    assert overrides["foundation_enabled"] is False
    assert overrides["pretrained"] is False
    assert Path(overrides["model"]).name == "yolo26-d1-dinov3-latent-n.yaml"


def test_cache_report_discovery_requires_one_passed_report(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        discover_cache_report(tmp_path, "train2017", None)
    path = manifests / "wp8-full-train2017-a.json"
    path.write_text(json.dumps({"status": "passed", "finalization": {"split": "train2017"}}), encoding="utf-8")
    assert discover_cache_report(tmp_path, "train2017", None) == path
    duplicate = manifests / "wp8-full-train2017-b.json"
    duplicate.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        discover_cache_report(tmp_path, "train2017", None)


def test_cache_evidence_matches_index_and_fails_on_report_drift(tmp_path, monkeypatch):
    cache_dir, report_path = build_cache_and_report(tmp_path, "train2017")
    monkeypatch.setitem(run_wp8_train.SPLIT_COUNTS, "train2017", 1)

    evidence = validate_cache_evidence(cache_dir, report_path, "train2017")
    assert evidence["sample_count"] == 1
    assert evidence["contract_sha256"] == verify_feature_cache(cache_dir)["contract_sha256"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["finalization"]["verification"]["content_sha256"] = "f" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        validate_cache_evidence(cache_dir, report_path, "train2017")


def test_train_command_requires_torchrun_environment(tmp_path, monkeypatch):
    args = type("Args", (), {"config": CONFIG_PATH})()
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(RuntimeError, match="torchrun"):
        run_wp8_train.train(args)


def test_script_can_be_executed_directly():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/d1/run_wp8_train.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout
    assert "summarize" in result.stdout
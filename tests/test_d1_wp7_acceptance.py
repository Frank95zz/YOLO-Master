"""Offline tests for the D1 WP7 acceptance runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.d1 import run_wp7
from scripts.d1.run_wp7 import (
    PROFILE_NAMES,
    evaluate_training_rows,
    load_contract,
    report_matches,
    select_records,
    validate_source_files,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "ultralytics/cfg/experiments/d1/wp7-minimal-tests.yaml"
COCO8_TRAIN = ("000000000009", "000000000025", "000000000030", "000000000034")
COCO8_VAL = ("000000000036", "000000000042", "000000000049", "000000000061")


def cache_records(image_ids):
    return {
        f"train2017/{image_id}": {
            "sample_id": f"train2017/{image_id}",
            "image_path": f"images/train2017/{image_id}.jpg",
            "split": "train2017",
        }
        for image_id in image_ids
    }


def result_rows(count: int = 20, *, first_loss: float = 10.0, last_loss: float = 4.0):
    rows = []
    for index in range(count):
        fraction = index / max(count - 1, 1)
        total = first_loss + fraction * (last_loss - first_loss)
        rows.append(
            {
                "train/box_loss": total * 0.5,
                "train/cls_loss": total * 0.3,
                "train/dfl_loss": total * 0.2,
                "train/latent_balance_loss": 0.01,
                "train/latent_z_loss": 0.02,
                "train/latent_aux_loss": 0.03,
                "train/mixture_aux_loss": 0.04,
                "metrics/mAP50(B)": 0.25,
                "metrics/mAP50-95(B)": 0.08,
            }
        )
    return rows


def test_contract_locks_wp7_profiles_and_has_no_host_paths() -> None:
    contract = load_contract(CONTRACT_PATH)

    assert contract["schema_version"] == "d1-wp7-v1"
    assert tuple(contract["profiles"]) == PROFILE_NAMES
    assert tuple(contract["profiles"]["coco8"]["train_ids"]) == COCO8_TRAIN
    assert tuple(contract["profiles"]["coco8"]["val_ids"]) == COCO8_VAL
    assert len(contract["profiles"]["overfit32"]["train_ids"]) == 32
    serialized = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "/data/" not in serialized
    assert "/root/" not in serialized
    assert "10.210.22.36" not in serialized


def test_record_selection_preserves_contract_order_and_rejects_missing() -> None:
    records = cache_records(COCO8_TRAIN)
    selected = select_records(records, tuple(reversed(COCO8_TRAIN)))

    assert [record["sample_id"].split("/")[1] for record in selected] == list(reversed(COCO8_TRAIN))
    with pytest.raises(FileNotFoundError, match="cache is missing"):
        select_records(records, ("999999999999",))
    with pytest.raises(ValueError, match="duplicates"):
        select_records(records, (COCO8_TRAIN[0], COCO8_TRAIN[0]))


def test_source_validation_requires_real_images_and_nonempty_labels(tmp_path: Path) -> None:
    image_id = COCO8_TRAIN[0]
    records = list(cache_records((image_id,)).values())
    image = tmp_path / "images" / "train2017" / f"{image_id}.jpg"
    label = tmp_path / "labels" / "train2017" / f"{image_id}.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg")
    label.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty label"):
        validate_source_files(tmp_path, records, require_nonempty_labels=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert validate_source_files(tmp_path, records, require_nonempty_labels=True) == [image.resolve()]


def test_overfit_metrics_apply_loss_and_map_thresholds() -> None:
    contract = load_contract(CONTRACT_PATH)
    profile = contract["profiles"]["overfit32"]
    metrics, failures = evaluate_training_rows(result_rows(last_loss=1.0), "overfit32", profile)

    assert failures == []
    assert metrics["final_to_initial_loss_ratio"] <= 0.5
    assert metrics["final_map50"] == 0.25
    assert metrics["final_map50_95"] == 0.08


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (result_rows(last_loss=8.0), "loss reduction"),
        (
            [dict(row, **{"metrics/mAP50(B)": 0.1}) for row in result_rows()],
            "mAP50 did not meet",
        ),
        (
            [dict(row, **{"metrics/mAP50-95(B)": 0.01}) for row in result_rows()],
            "mAP50-95 did not meet",
        ),
        (
            [dict(row, **{"train/box_loss": float("nan")}) for row in result_rows()],
            "NaN or Inf",
        ),
    ],
)
def test_overfit_metrics_fail_closed(rows, message: str) -> None:
    profile = load_contract(CONTRACT_PATH)["profiles"]["overfit32"]
    _metrics, failures = evaluate_training_rows(rows, "overfit32", profile)

    assert any(message in failure for failure in failures)


def test_coco8_has_no_accuracy_gate() -> None:
    profile = load_contract(CONTRACT_PATH)["profiles"]["coco8"]
    rows = result_rows(1)
    rows[0]["metrics/mAP50(B)"] = 0.0
    rows[0]["metrics/mAP50-95(B)"] = 0.0

    metrics, failures = evaluate_training_rows(rows, "coco8", profile)

    assert failures == []
    assert metrics["final_map50"] == 0.0


def test_runtime_identity_uses_reader_root(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "wp2-train100-a"
    cache_root.mkdir()
    (cache_root / "index.json").write_text(
        json.dumps({"contract_sha256": "b" * 64, "content_sha256": "c" * 64}),
        encoding="utf-8",
    )
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "model.safetensors").write_bytes(b"weights")
    reader = SimpleNamespace(
        root=cache_root,
        contract={"teacher_weights_sha256": "d" * 64},
    )
    monkeypatch.setattr(run_wp7, "git_commit", lambda _root: "a" * 40)
    monkeypatch.setattr(run_wp7, "sha256_file", lambda _path: "d" * 64)

    identity = run_wp7.runtime_identity(tmp_path, reader, weights_dir)

    assert identity["cache_id"] == "wp2-train100-a"
    assert identity["cache_contract_sha256"] == "b" * 64
    assert identity["cache_content_sha256"] == "c" * 64


def test_report_reuse_requires_exact_runtime_identity() -> None:
    identity = {
        "schema_version": "d1-wp7-v1",
        "code_commit": "a" * 40,
        "cache_id": "wp2-train100-a",
        "cache_contract_sha256": "b" * 64,
        "cache_content_sha256": "c" * 64,
        "teacher_weights_sha256": "d" * 64,
    }
    report = {"status": "passed", "identity": identity}

    assert report_matches(report, identity)
    assert not report_matches({**report, "status": "failed"}, identity)
    assert not report_matches(report, {**identity, "code_commit": "e" * 40})
    assert not Path(json.dumps(report)).is_absolute()

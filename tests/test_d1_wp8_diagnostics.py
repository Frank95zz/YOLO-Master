from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from scripts.d1.diagnose_wp8 import (
    numeric_summary,
    prediction_summary,
    router_summary,
    select_split_paths,
)


def test_numeric_summary_is_finite_and_stable():
    summary = numeric_summary([0.0, 1.0, 2.0, 3.0])
    assert summary["count"] == 4
    assert summary["median"] == pytest.approx(1.5)
    assert summary["mean"] == pytest.approx(1.5)
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        numeric_summary([float("nan")])


def test_prediction_summary_includes_zero_detection_images():
    predictions = [
        {"image_id": 9, "category_id": 1, "score": 0.75},
        {"image_id": 9, "category_id": 3, "score": 0.25},
        {"image_id": 25, "category_id": 3, "score": 0.5},
    ]
    summary = prediction_summary(predictions, ["000000000009", "000000000025", "000000000030"])
    assert summary["total_detections"] == 3
    assert summary["zero_detection_images"] == 1
    assert summary["detections_per_image"]["mean"] == pytest.approx(1.0)
    assert summary["confidence"]["median"] == pytest.approx(0.5)
    assert summary["coco_category_histogram"] == {"1": 1, "3": 2}


def _router_record(sample_id: str, scale: str, p3, p4, p5):
    def item(probabilities):
        entropy = -sum(value * math.log(value) for value in probabilities)
        return {"probabilities": probabilities, "entropy": entropy, "top1_expert": probabilities.index(max(probabilities))}

    return {
        "sample_id": sample_id,
        "object_count": 1,
        "bbox_area_counts": {"small": scale == "small", "medium": scale == "medium", "large": scale == "large"},
        "largest_bbox_scale": scale,
        "routing": {"p3": item(p3), "p4": item(p4), "p5": item(p5)},
    }


def test_router_summary_preserves_per_sample_specialization():
    records = [
        _router_record(
            "val2017/1",
            "small",
            [0.7, 0.1, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
        ),
        _router_record(
            "val2017/2",
            "large",
            [0.1, 0.7, 0.1, 0.1],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
        ),
    ]
    summary = router_summary(records)
    assert summary["p3"]["top1_counts"] == [1, 1, 0, 0]
    assert summary["p4"]["mean_normalized_entropy"] == pytest.approx(1.0)
    assert summary["p5"]["by_largest_bbox_scale"]["small"]["images"] == 1
    assert summary["p5"]["by_largest_bbox_scale"]["large"]["images"] == 1


def test_router_summary_rejects_non_normalized_probabilities():
    record = _router_record(
        "val2017/1",
        "small",
        [0.4, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    )
    with pytest.raises(ValueError, match="not normalized"):
        router_summary([record])


def test_select_split_paths_uses_sorted_deterministic_prefix(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "coco"
    manifest_dir = repo / "experiments/d1/manifests"
    manifest_dir.mkdir(parents=True)
    relative = [f"images/train2017/{index:012d}.jpg" for index in range(4)]
    (manifest_dir / "coco2017-train2017.txt").write_text("\n".join(relative) + "\n", encoding="utf-8")
    for name in relative:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    module = __import__("scripts.d1.diagnose_wp8", fromlist=["EXPECTED_SPLIT_COUNTS"])
    monkeypatch.setitem(module.EXPECTED_SPLIT_COUNTS, "train2017", 4)
    selected, absolute, digest = select_split_paths(repo, data, "train2017", 2)
    assert selected == relative[:2]
    assert absolute == [(data / name).resolve() for name in relative[:2]]
    expected = hashlib.sha256(("\n".join(relative[:2]) + "\n").encode()).hexdigest()
    assert digest == expected


def test_select_split_paths_rejects_unsorted_manifest(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    data = tmp_path / "coco"
    manifest_dir = repo / "experiments/d1/manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "coco2017-val2017.txt").write_text("b.jpg\na.jpg\n", encoding="utf-8")
    module = __import__("scripts.d1.diagnose_wp8", fromlist=["EXPECTED_SPLIT_COUNTS"])
    monkeypatch.setitem(module.EXPECTED_SPLIT_COUNTS, "val2017", 2)
    with pytest.raises(ValueError, match="sorted"):
        select_split_paths(repo, data, "val2017", None)

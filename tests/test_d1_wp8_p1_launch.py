"""Offline tests for the D1 scratch pipeline's evidence and failure gates."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.d1 import launch_wp8_p1 as launch
from ultralytics.cfg import get_cfg


def probe_reports():
    identity = {"commit": "a" * 40, "dirty": False}
    rows = [{
        "rank": rank, "identity": deepcopy(identity), "status": "passed", "workers": 4,
        "actual_workers": 4, "per_gpu_batch": 64, "prefetch_factor": 1, "amp": True,
        "steps": 40, "successful_steps": 40, "measured_steps": 20, "val_dataset_samples": 5000,
        "measured_seconds": 40.0, "validation_checkpoint_seconds": 20.0,
        "parameter_max_delta": 0.01, "final_amp_scale": 16.0, "val_seen": 5000 if rank == 0 else 833,
    } for rank in range(6)]
    return identity, rows


def test_aggregate_uses_slowest_rank_and_30_epoch_window():
    identity, rows = probe_reports()
    rows[5]["measured_seconds"] = 60.0
    result = launch.aggregate_probe(rows, identity, 4)
    assert result["images_per_second"] == pytest.approx(128)
    assert result["train_val_seconds_30_epochs"] == 30 * (309 * 3 + 20)
    assert result["estimated_hours_range"][0] < result["estimated_hours_range"][1]


@pytest.mark.parametrize("key,value", [
    ("identity", {}), ("status", "failed"), ("actual_workers", 2), ("amp", False),
    ("successful_steps", 39), ("per_gpu_batch", 32), ("measured_seconds", float("nan")),
    ("parameter_max_delta", 0.0), ("val_dataset_samples", 4999), ("val_seen", 4999),
])
def test_aggregate_rejects_bad_evidence(key, value):
    identity, rows = probe_reports()
    rows[0][key] = value
    with pytest.raises(ValueError):
        launch.aggregate_probe(rows, identity, 4)


def test_aggregate_requires_six_distinct_ranks():
    identity, rows = probe_reports()
    rows[5] = deepcopy(rows[0])
    with pytest.raises(ValueError):
        launch.aggregate_probe(rows, identity, 4)


def test_probe_cannot_run_without_ddp(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(RuntimeError, match="six torchrun"):
        launch.probe_worker(SimpleNamespace())


def test_prefix_warmup_matches_full_training():
    assert launch.TRAIN_BATCHES == 309
    assert round((3 * launch.TRAIN_BATCHES / launch.STEPS) * launch.STEPS) == 927


def test_evaluation_options_are_accepted():
    args = get_cfg(overrides={"task": "detect", "half": False, "quantize": None,
                             "rect": False, "batch": 64, "conf": 0.001, "iou": 0.7})
    assert args.quantize is None


def test_official_evaluator_handles_no_predictions(tmp_path):
    pytest.importorskip("faster_coco_eval")
    annotation = tmp_path / "instances.json"
    annotation.write_text(json.dumps({
        "info": {}, "images": [{"id": 1, "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
    }))
    predictions = tmp_path / "predictions.json"
    predictions.write_text("[]")
    metrics = launch.official_metrics(annotation, predictions, [1])
    assert metrics["AP_all"] == 0.0
    assert metrics["AP_50"] == 0.0


def test_atomic_status_and_inputs(tmp_path):
    path = tmp_path / "nested/status"
    launch.atomic_text(path, "PREPARING\n")
    launch.atomic_text(path, "TRAIN\n")
    assert path.read_text() == "TRAIN\n"
    assert not path.with_name(path.name + ".part").exists()


def test_keeper_command_does_not_use_default_pidfile():
    script = Path("/example/gpu_keeper/gpu_keeper.py")
    command = launch.keeper_command(script, "stop")
    assert command[2] == "stop"
    assert command[command.index("--pidfile") + 1] == "/example/gpu_keeper/gpu_keeper_v2.pid"

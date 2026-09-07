"""Offline guards for the isolated D1 two-GPU cache-layout experiment."""

import copy
from typing import ClassVar

import numpy as np
import pytest
import torch

from scripts.d1 import benchmark_cache_layout_gpu as bench


def reports(backend="shard", workers=0):
    return [
        {
            "rank": rank,
            "backend": backend,
            "workers": workers,
            "training_steps": 64,
            "optimizer_updates": 64,
            "measured_steps": 48,
            "weights_changed": True,
            "finite_losses": True,
            "training_seconds": 12 + rank,
            "loader_seconds": 3 + rank,
            "initial_parameters_sha256": "initial",
            "final_parameters_sha256": "final",
            "sample_order_sha256": f"train-{rank}",
            "loader_order_sha256": f"loader-{rank}",
            "loss_trace": [[1.0, 2.0]] * 64,
        }
        for rank in (0, 1)
    ]


def test_case_grid_is_paired_and_bounded():
    grid = bench.case_grid()
    assert len(grid) == 12
    for workers in (0, 2, 4):
        assert [kind for w, kind in grid if w == workers] == ["shard", "npy", "npy", "shard"]


def test_cleanup_handles_children_after_parent_exit(monkeypatch):
    calls = []

    class Process:
        pid = 12345

        def poll(self):
            return 0

        def wait(self, timeout):
            calls.append(("wait", timeout))

    times = iter([0, 1, 20])
    monkeypatch.setattr(bench.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(bench.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(bench.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    bench.stop_group(Process())
    assert (12345, bench.signal.SIGTERM) in calls
    assert (12345, bench.signal.SIGKILL) in calls


def test_aggregation_uses_slowest_rank_and_real_updates():
    result = bench.aggregate_case(reports())
    assert result["training_images_per_second"] == pytest.approx(6144 / 13)
    assert result["loader_images_per_second"] == 1536 / 4
    assert result["full_training_eta_available"] is False
    bad = reports()
    bad[1]["optimizer_updates"] = 63
    with pytest.raises(ValueError, match="optimizer"):
        bench.aggregate_case(bad)
    with pytest.raises(ValueError, match="two rank"):
        bench.aggregate_case(bad[:1])


def test_comparison_requires_same_inputs_and_loss():
    cases = [bench.aggregate_case(reports(kind, workers)) for workers, kind in bench.case_grid()]
    result = bench.compare_cases(cases)
    assert all(value["training_speedup"] == 1 for value in result.values())
    changed = copy.deepcopy(cases)
    changed[1]["ranks"][0]["sample_order_sha256"] = "bad"
    with pytest.raises(ValueError, match="inputs"):
        bench.compare_cases(changed)
    changed = copy.deepcopy(cases)
    changed[1]["ranks"][0]["loss_trace"][0][0] += 1
    with pytest.raises(ValueError, match="loss trajectories"):
        bench.compare_cases(changed)


def test_npy_reader_keeps_layer_order_and_original_metadata(tmp_path):
    class Base:
        records: ClassVar[dict] = {"train2017/000000000009": {"shard": "original"}}
        contract: ClassVar[dict] = {"feature_names": bench.NAMES}
        index: ClassVar[dict] = {"content_sha256": "source"}
        closed = False

        def close(self):
            self.closed = True

    path = bench.sample_path(tmp_path, "train2017/000000000009")
    path.parent.mkdir(parents=True)
    values = np.stack([np.full(bench.SHAPE, i, dtype=np.float16) for i in range(3)])
    np.save(path, values, allow_pickle=False)
    base = Base()
    reader = bench.NpyReader(base, tmp_path)
    result = reader.get("train2017/000000000009")
    assert tuple(result) == bench.NAMES
    assert reader.records is base.records
    for index, value in enumerate(result.values()):
        assert value.shape == bench.SHAPE and value.dtype == torch.float16
        assert torch.all(value == index)
    with pytest.raises(ValueError, match="CPU"):
        reader.get("train2017/000000000009", device="cuda:0")
    reader.close()
    assert base.closed

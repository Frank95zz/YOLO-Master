"""Offline correctness gates for the bounded NPY layout experiment."""

import json

import numpy as np
import pytest

from scripts.d1 import benchmark_cache_layout as bench


def sample():
    values = {name: np.full((2, 2, 2), index + 1, dtype=np.float16) for index, name in enumerate(bench.NAMES)}
    record = {
        "sample_id": "train2017/000000000009",
        "tensors": {name: {"sha256": bench.digest(value)} for name, value in values.items()},
    }
    return values, record


def test_selection_is_stable_and_excludes_val():
    records = {f"train2017/{i:012d}": {} for i in range(20)}
    records["val2017/000000000001"] = {}
    selected = bench.select_ids(records, 10, 0)
    assert selected == bench.select_ids(dict(reversed(list(records.items()))), 10, 0)
    assert len(selected) == len(set(selected)) == 10
    assert all(sid.startswith("train2017/") for sid in selected)
    assert selected != bench.select_ids(records, 10, 1)


@pytest.mark.parametrize("count", [0, -1, 2, True])
def test_bad_selection_count(count):
    with pytest.raises(ValueError):
        bench.select_ids({"train2017/000000000001": {}}, count, 0)


@pytest.mark.parametrize("sid", ["../bad", "train2017/../x", "/train2017/000000000009", "val2017/000000000009"])
def test_unsafe_sample_path(tmp_path, sid):
    with pytest.raises(ValueError):
        bench.sample_path(tmp_path, sid)


def test_lossless_export_readback_and_idempotent_resume(tmp_path):
    values, record = sample()
    assert bench.export_sample(tmp_path, record, dict(reversed(list(values.items()))), shape=(2, 2, 2))
    path = bench.sample_path(tmp_path, record["sample_id"])
    original = path.read_bytes()
    loaded = bench.read_npy(path, shape=(2, 2, 2))
    assert tuple(loaded) == bench.NAMES
    for name in bench.NAMES:
        assert np.array_equal(loaded[name], values[name])
        assert bench.digest(loaded[name]) == record["tensors"][name]["sha256"]
    assert not bench.export_sample(tmp_path, record, values, shape=(2, 2, 2))
    assert path.read_bytes() == original
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("fault", ["dtype", "shape", "nan", "inf", "checksum"])
def test_invalid_features_never_commit(tmp_path, fault):
    values, record = sample()
    if fault == "dtype":
        values = {name: value.astype(np.float32) for name, value in values.items()}
    elif fault == "shape":
        values = {name: value[:, :, :1] for name, value in values.items()}
    else:
        values["block8"][0, 0, 0] = {"nan": np.nan, "inf": np.inf, "checksum": 12}[fault]
    with pytest.raises(ValueError):
        bench.export_sample(tmp_path, record, values, shape=(2, 2, 2))
    assert not list(tmp_path.rglob("*.npy"))


def test_corrupt_existing_file_is_not_silently_replaced(tmp_path):
    values, record = sample()
    bench.export_sample(tmp_path, record, values, shape=(2, 2, 2))
    path = bench.sample_path(tmp_path, record["sample_id"])
    np.save(path, np.zeros((3, 2, 2, 2), dtype=np.float16), allow_pickle=False)
    with pytest.raises(ValueError, match="checksum"):
        bench.export_sample(tmp_path, record, values, shape=(2, 2, 2))


def test_memory_guard_distinguishes_file_cache_and_shmem():
    bench.guard_memory({"limit_bytes": 100, "usage_bytes": 99, "rss_bytes": 20, "shmem_bytes": 10}, 40)
    with pytest.raises(RuntimeError, match="Memory guard"):
        bench.guard_memory({"limit_bytes": 100, "usage_bytes": 99, "rss_bytes": 20, "shmem_bytes": 50}, 40)


def test_summary_is_paired_and_never_reports_training_eta():
    passes = [
        {
            "backend": kind,
            "elapsed_seconds": seconds,
            "sample_count": 4,
            "batch_size": 2,
            "order_sha256": "order",
            "sentinel_sha256": "sentinel",
        }
        for kind, seconds in [("shard", 4), ("npy", 2), ("npy", 2), ("shard", 6)]
    ]
    result = bench.summarize(passes)
    assert result["npy_speedup_vs_shard"] == 2.5
    assert result["full_training_eta_available"] is False
    passes[-1]["order_sha256"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        bench.summarize(passes)
    with pytest.raises(ValueError, match="ABBA"):
        bench.summarize(passes[:2])


def test_atomic_json(tmp_path):
    path = tmp_path / "status.json"
    bench.write_json(path, {"status": "RUNNING"})
    assert json.loads(path.read_text()) == {"status": "RUNNING"}
    assert not path.with_name(path.name + ".part").exists()

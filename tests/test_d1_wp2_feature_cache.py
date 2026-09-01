"""Offline WP2 tests for the D1 sharded feature-cache protocol."""

import json
from pathlib import Path

import pytest
import torch

from ultralytics.nn.foundation import cache as cache_module
from ultralytics.nn.foundation.cache import (
    FeatureCacheReader,
    FeatureCacheWriter,
    build_cache_key,
    compare_feature_caches,
    sha256_bytes,
    verify_feature_cache,
)


def contract(**updates):
    value = {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-teacher",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": ["block4", "block8", "block12"],
        "dtype": "float16",
        "expected_shape": [2, 2, 2],
    }
    value.update(updates)
    return value


def features(value):
    return {
        name: torch.full((2, 2, 2), float(value + index), dtype=torch.float32)
        for index, name in enumerate(("block4", "block8", "block12"))
    }


def image_sha(value):
    return sha256_bytes(f"image-{value}".encode())


def add_sample(writer, value):
    return writer.add(
        sample_id=f"train2017/{value:012d}",
        split="train2017",
        image_path=f"images/train2017/{value:012d}.jpg",
        image_sha256=image_sha(value),
        features=features(value),
    )


def build_cache(path: Path, values=(1, 2)):
    with FeatureCacheWriter(path, split="train2017", contract=contract(), target_shard_bytes=60) as writer:
        for value in values:
            assert add_sample(writer, value)


def test_cache_key_changes_with_every_content_identity_field():
    base = {
        "image_sha256": "1" * 64,
        "preprocessing_sha256": "2" * 64,
        "teacher_weights_sha256": "3" * 64,
        "output_layers": (4, 8, 12),
        "dtype": "float16",
    }
    baseline = build_cache_key(**base)
    for name, changed in (
        ("image_sha256", "4" * 64),
        ("preprocessing_sha256", "5" * 64),
        ("teacher_weights_sha256", "6" * 64),
        ("output_layers", (4, 12)),
    ):
        candidate = dict(base)
        candidate[name] = changed
        assert build_cache_key(**candidate) != baseline


@pytest.mark.parametrize(
    "updates, exception",
    [
        ({"dtype": "float32"}, ValueError),
        ({"output_layers": []}, ValueError),
        ({"output_layers": [8, 4]}, ValueError),
        ({"feature_names": ["block4"]}, ValueError),
        ({"expected_shape": [2, 2]}, ValueError),
        ({"teacher_weights_sha256": "bad"}, ValueError),
    ],
)
def test_invalid_cache_contract_fails_fast(tmp_path, updates, exception):
    with pytest.raises(exception):
        FeatureCacheWriter(tmp_path, split="train2017", contract=contract(**updates))


def test_sharded_roundtrip_index_and_full_verification(tmp_path):
    build_cache(tmp_path)

    report = verify_feature_cache(tmp_path)
    reader = FeatureCacheReader(tmp_path)
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))

    assert report["sample_count"] == 2
    assert report["shard_count"] == 2
    assert report["tensor_count"] == 6
    assert report["part_files"] == []
    assert index["split_counts"] == {"train2017": 2}
    assert index["content_sha256"] == report["content_sha256"]
    assert all(not Path(record["image_path"]).is_absolute() for record in reader.records.values())
    loaded = reader.get("train2017/000000000001")
    assert tuple(loaded) == ("block4", "block8", "block12")
    assert loaded["block4"].dtype == torch.float16
    assert torch.all(loaded["block4"] == 1)
    assert torch.all(loaded["block12"] == 3)


def test_resume_skips_committed_sample_and_appends_next_shard(tmp_path):
    build_cache(tmp_path, values=(1,))
    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)

    assert add_sample(writer, 1) is False
    assert add_sample(writer, 2) is True
    writer.close()

    report = verify_feature_cache(tmp_path)
    assert report["sample_count"] == 2
    assert report["shard_count"] == 2


def test_missing_index_is_rebuilt_from_committed_shard_headers(tmp_path):
    build_cache(tmp_path)
    expected = verify_feature_cache(tmp_path)["content_sha256"]
    (tmp_path / "index.json").unlink()
    (tmp_path / "samples.jsonl").unlink()

    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)
    writer.close()

    assert verify_feature_cache(tmp_path)["content_sha256"] == expected


def test_two_independent_builds_have_identical_content(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    build_cache(first)
    build_cache(second)

    comparison = compare_feature_caches(first, second)

    assert comparison["identical"] is True
    assert comparison["sample_count"] == 2
    assert comparison["first_shards"] == comparison["second_shards"] == 2


@pytest.mark.parametrize(
    "bad_features, message",
    [
        ({"block4": torch.zeros(2, 2, 2)}, "ordered"),
        (
            {
                "block4": torch.zeros(2, 2, 2),
                "block8": torch.zeros(2, 2, 2),
                "block12": torch.zeros(2, 2, 1),
            },
            "shape",
        ),
        (
            {
                "block4": torch.zeros(2, 2, 2),
                "block8": torch.full((2, 2, 2), float("nan")),
                "block12": torch.zeros(2, 2, 2),
            },
            "NaN or Inf",
        ),
    ],
)
def test_invalid_features_fail_before_commit(tmp_path, bad_features, message):
    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)
    with pytest.raises(ValueError, match=message):
        writer.add(
            sample_id="train2017/000000000001",
            split="train2017",
            image_path="images/train2017/000000000001.jpg",
            image_sha256=image_sha(1),
            features=bad_features,
        )


def test_corrupt_shard_fails_checksum_verification(tmp_path):
    build_cache(tmp_path, values=(1,))
    shard = next(tmp_path.glob("*.safetensors"))
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="shard checksum mismatch"):
        verify_feature_cache(tmp_path)


def test_failed_shard_save_preserves_part_file(tmp_path, monkeypatch):
    safe_open, _save_file = cache_module._safetensors_api()

    def fail_after_partial_write(_tensors, filename, metadata):
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cache_module, "_safetensors_api", lambda: (safe_open, fail_after_partial_write))
    writer = FeatureCacheWriter(tmp_path, split="train2017", contract=contract(), target_shard_bytes=60)
    add_sample(writer, 1)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        writer.flush()

    assert (tmp_path / ".train2017-00000.safetensors.part").read_bytes() == b"partial"
    assert not list(tmp_path.glob("*.safetensors"))

"""Offline tests for D1 WP8 deterministic multi-GPU cache orchestration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from scripts.d1.run_wp8 import (
    SCHEMA_VERSION,
    finalize_rank_caches,
    parse_devices,
    partition_paths,
    paths_sha256,
    rank_cache_dir,
    validate_rank_reports,
    validate_split_membership,
)
from ultralytics.nn.foundation.cache import FeatureCacheReader, FeatureCacheWriter, sha256_bytes


def contract():
    return {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-teacher",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": ["block4", "block8", "block12"],
        "dtype": "float16",
        "expected_shape": [2, 2, 2],
    }


def features(value: int):
    return {
        name: torch.full((2, 2, 2), value + index, dtype=torch.float32)
        for index, name in enumerate(("block4", "block8", "block12"))
    }


def image_sha(value: int) -> str:
    return sha256_bytes(f"image-{value}".encode())


def paths(count: int) -> list[str]:
    return [f"images/train2017/{value:012d}.jpg" for value in range(count)]


def build_rank(cache_dir: Path, all_paths: list[str], rank: int, world_size: int) -> dict:
    part_paths = partition_paths(all_paths, rank, world_size)
    part_dir = rank_cache_dir(cache_dir, rank)
    writer = FeatureCacheWriter(
        part_dir,
        split="train2017",
        contract=contract(),
        target_shard_bytes=60,
        shard_prefix=f"train2017-r{rank:02d}",
    )
    for image_path in part_paths:
        value = int(Path(image_path).stem)
        writer.add(
            sample_id=f"train2017/{value:012d}",
            split="train2017",
            image_path=image_path,
            image_sha256=image_sha(value),
            features=features(value),
        )
    writer.close()
    from ultralytics.nn.foundation.cache import verify_feature_cache

    return {
        "schema_version": SCHEMA_VERSION,
        "operation": "build-part",
        "status": "passed",
        "code_commit": "c" * 40,
        "split": "train2017",
        "selected_sample_count": len(all_paths),
        "selected_paths_sha256": paths_sha256(all_paths),
        "rank": rank,
        "world_size": world_size,
        "partition_sample_count": len(part_paths),
        "partition_paths_sha256": paths_sha256(part_paths),
        "target_shard_bytes": 60,
        "verification": verify_feature_cache(part_dir),
    }


def test_script_can_be_executed_directly():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "d1" / "run_wp8.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "benchmark-cache" in result.stdout


def test_partition_is_stable_disjoint_and_complete():
    source = paths(17)
    first = [partition_paths(source, rank, 6) for rank in range(6)]
    second = [partition_paths(source, rank, 6) for rank in range(6)]

    assert first == second
    assert sorted(item for partition in first for item in partition) == source
    assert sum(len(set(partition)) for partition in first) == len(source)
    assert not any(set(first[left]) & set(first[right]) for left in range(6) for right in range(left + 1, 6))


@pytest.mark.parametrize(("rank", "world_size"), ((-1, 6), (6, 6), (0, 0), (True, 6)))
def test_partition_rejects_invalid_rank_contract(rank, world_size):
    with pytest.raises(ValueError):
        partition_paths(paths(3), rank, world_size)


def test_split_membership_requires_counts_and_disjoint_ids():
    train = ["images/train2017/000000000001.jpg", "images/train2017/000000000002.jpg"]
    val = ["images/val2017/000000000003.jpg"]
    validate_split_membership(train, val, {"train2017": 2, "val2017": 1})
    with pytest.raises(ValueError, match="counts differ"):
        validate_split_membership(train, val, {"train2017": 3, "val2017": 1})
    with pytest.raises(ValueError, match="overlap"):
        validate_split_membership(train, ["images/val2017/000000000002.jpg"], {"train2017": 2, "val2017": 1})


def test_device_parser_requires_unique_numeric_devices():
    assert parse_devices("0,1,2,3,4,5") == ("0", "1", "2", "3", "4", "5")
    for value in ("", "0,0", "cuda:0,1"):
        with pytest.raises(ValueError):
            parse_devices(value)


def test_rank_reports_fail_closed_on_missing_or_mismatched_identity():
    source = paths(6)
    reports = []
    for rank in range(2):
        part = partition_paths(source, rank, 2)
        reports.append(
            {
                "schema_version": SCHEMA_VERSION,
                "operation": "build-part",
                "status": "passed",
                "code_commit": "c" * 40,
                "split": "train2017",
                "selected_sample_count": len(source),
                "selected_paths_sha256": paths_sha256(source),
                "rank": rank,
                "world_size": 2,
                "partition_sample_count": len(part),
                "partition_paths_sha256": paths_sha256(part),
            }
        )
    validate_rank_reports(reports, source, split="train2017", world_size=2, code_commit="c" * 40)
    with pytest.raises(ValueError, match="expected 2"):
        validate_rank_reports(reports[:1], source, split="train2017", world_size=2, code_commit="c" * 40)
    reports[1]["code_commit"] = "d" * 40
    with pytest.raises(ValueError, match="code_commit"):
        validate_rank_reports(reports, source, split="train2017", world_size=2, code_commit="c" * 40)


def test_finalize_hardlinks_rank_shards_and_builds_one_index(tmp_path):
    source = paths(8)
    cache_dir = tmp_path / "final"
    reports = [build_rank(cache_dir, source, rank, 2) for rank in range(2)]
    split_file = tmp_path / "experiments" / "d1" / "manifests" / "coco2017-train2017.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_text("\n".join(source) + "\n", encoding="utf-8")

    with patch("scripts.d1.run_wp8.cache_contract", return_value=contract()), patch(
        "subprocess.check_output", return_value="c" * 40 + "\n"
    ):
        result = finalize_rank_caches(
            repo_root=tmp_path,
            cache_dir=cache_dir,
            split="train2017",
            limit=None,
            world_size=2,
            reports=reports,
            target_shard_bytes=60,
        )

    assert result["status"] == "passed"
    assert result["verification"]["sample_count"] == 8
    assert set(FeatureCacheReader(cache_dir).records) == {f"train2017/{value:012d}" for value in range(8)}
    for rank in range(2):
        for source_shard in rank_cache_dir(cache_dir, rank).glob("*.safetensors"):
            assert (cache_dir / source_shard.name).samefile(source_shard)


def test_finalize_rejects_unfinished_part_file(tmp_path):
    source = paths(4)
    cache_dir = tmp_path / "final"
    reports = [build_rank(cache_dir, source, rank, 2) for rank in range(2)]
    split_file = tmp_path / "experiments" / "d1" / "manifests" / "coco2017-train2017.txt"
    split_file.parent.mkdir(parents=True)
    split_file.write_text("\n".join(source) + "\n", encoding="utf-8")
    (rank_cache_dir(cache_dir, 0) / ".unfinished.part").write_bytes(b"partial")

    with patch("subprocess.check_output", return_value="c" * 40 + "\n"), pytest.raises(
        ValueError, match="unfinished"
    ):
        finalize_rank_caches(
            repo_root=tmp_path,
            cache_dir=cache_dir,
            split="train2017",
            limit=None,
            world_size=2,
            reports=reports,
            target_shard_bytes=60,
        )

"""Tests for the D1 post-training page-cache release wrapper."""

from pathlib import Path

import pytest

from scripts.d1 import run_with_cache_cleanup as cleanup


def test_cgroup_memory_snapshot_reads_v1(tmp_path):
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "memory.usage_in_bytes").write_text("100\n", encoding="ascii")
    (v1 / "memory.limit_in_bytes").write_text("200\n", encoding="ascii")
    (v1 / "memory.kmem.usage_in_bytes").write_text("30\n", encoding="ascii")
    (v1 / "memory.stat").write_text("total_rss 40\ntotal_cache 50\n", encoding="ascii")

    assert cleanup.cgroup_memory_snapshot(v1, tmp_path / "missing") == {
        "cgroup_version": 1,
        "usage_bytes": 100,
        "limit_bytes": 200,
        "rss_bytes": 40,
        "cache_bytes": 50,
        "kernel_bytes": 30,
    }


def test_release_file_cache_targets_only_regular_matching_files(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    shard = cache / "shard.safetensors"
    shard.write_bytes(b"cache-data")
    (cache / "index.json").write_text("{}", encoding="ascii")
    nested = cache / "nested"
    nested.mkdir()
    nested_shard = nested / "nested.safetensors"
    nested_shard.write_bytes(b"nested")
    try:
        (cache / "link.safetensors").symlink_to(shard)
    except OSError:
        pass

    advised = []
    monkeypatch.setattr(cleanup.os, "posix_fadvise", lambda fd, offset, length, advice: advised.append((fd, advice)))
    snapshots = iter(
        [
            {"usage_bytes": 100, "cache_bytes": 50},
            {"usage_bytes": 60, "cache_bytes": 10},
        ]
    )
    monkeypatch.setattr(cleanup, "cgroup_memory_snapshot", lambda: next(snapshots))

    report = cleanup.release_file_cache([cache], wait_seconds=0)

    assert report["status"] == "passed"
    assert report["files_advised"] == 2
    assert report["logical_bytes"] == shard.stat().st_size + nested_shard.stat().st_size
    assert report["estimated_released_bytes"] == 40
    assert len(advised) == 2
    assert not report["errors"]


def test_normalize_cache_paths_rejects_root_and_missing_path(tmp_path):
    with pytest.raises(ValueError, match="filesystem root"):
        cleanup.normalize_cache_paths([Path("/")])
    with pytest.raises(FileNotFoundError):
        cleanup.normalize_cache_paths([tmp_path / "missing"])


def test_main_runs_cleanup_after_successful_command(tmp_path, monkeypatch):
    events = []
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    report_path = tmp_path / "report.json"

    def fake_run(command, *, termination_grace_seconds):
        events.append(("run", list(command), termination_grace_seconds))
        return {"executed": True, "exit_code": 0}

    def fake_release(paths, *, suffixes, wait_seconds):
        events.append(("cleanup", paths, suffixes, wait_seconds))
        return {"status": "passed", "files_advised": 1}

    monkeypatch.setattr(cleanup, "run_wrapped_command", fake_run)
    monkeypatch.setattr(cleanup, "release_file_cache", fake_release)

    result = cleanup.main(
        [
            "--cache-path",
            str(cache_path),
            "--wait-seconds",
            "0",
            "--report",
            str(report_path),
            "--",
            "python",
            "train.py",
        ]
    )

    assert result == 0
    assert [event[0] for event in events] == ["run", "cleanup"]
    assert report_path.is_file()


def test_main_preserves_training_failure_after_cleanup(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    monkeypatch.setattr(
        cleanup,
        "run_wrapped_command",
        lambda command, **kwargs: {"executed": True, "exit_code": 7},
    )
    monkeypatch.setattr(
        cleanup,
        "release_file_cache",
        lambda paths, **kwargs: {"status": "passed", "files_advised": 1},
    )

    assert cleanup.main(["--cache-path", str(cache_path), "--", "false"]) == 7


def test_cleanup_only_mode_is_supported(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    monkeypatch.setattr(
        cleanup,
        "release_file_cache",
        lambda paths, **kwargs: {"status": "passed", "files_advised": 1},
    )

    assert cleanup.main(["--cache-path", str(cache_path)]) == 0

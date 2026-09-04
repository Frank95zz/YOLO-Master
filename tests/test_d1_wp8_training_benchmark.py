"""Tests for the WP8 configurable-GPU training benchmark report logic."""

import pytest

from scripts.d1 import benchmark_wp8_training as benchmark

WORLD_SIZE = benchmark.WORLD_SIZE
aggregate_reports = benchmark.aggregate_reports
candidate_grid = benchmark.candidate_grid


def test_candidate_grid_is_complete_and_stable():
    assert candidate_grid([64, 128, 256], [8, 16]) == (
        (64, 8),
        (64, 16),
        (128, 8),
        (128, 16),
        (256, 8),
        (256, 16),
    )


def test_parser_accepts_world_size_seed_and_sample_offset():
    args = benchmark.parser().parse_args(
        [
            "--workspace",
            "/tmp/work",
            "--world-size",
            "2",
            "--seed",
            "1",
            "--sample-offset",
            "32000",
            "all",
        ]
    )
    assert (args.world_size, args.seed, args.sample_offset) == (2, 1, 32_000)


def test_aggregate_reports_uses_slowest_rank_and_reports_eta():
    reports = [
        {
            "rank": rank,
            "workers_per_rank": 8,
            "amp_enabled": True,
            "total_batches": 12,
            "actual_workers_per_rank": 8,
            "actual_per_gpu_batch": 64,
            "measured_batches": 10,
            "measured_seconds": 5.0 + rank,
            "data_wait_ratio": 0.1,
            "peak_gpu_bytes": 1_000 + rank,
        }
        for rank in range(WORLD_SIZE)
    ]
    result = aggregate_reports(reports, per_gpu_batch=64, warmup_steps=2)

    assert result["global_batch"] == 384
    assert result["measured_images"] == 3_840
    assert result["aggregate_images_per_second"] == pytest.approx(384.0)
    assert result["mean_data_wait_ratio"] == pytest.approx(0.1)
    assert result["estimated_train_hours_100_epochs"] > 0
    assert result["estimated_total_hours_with_15pct_overhead"] > result["estimated_train_plus_val_hours"]


def test_aggregate_reports_supports_two_gpu_global_batch_16():
    reports = [
        {
            "rank": rank,
            "workers_per_rank": 4,
            "amp_enabled": True,
            "measured_batches": 100,
            "measured_seconds": 10.0 + rank,
            "data_wait_ratio": 0.05,
            "peak_gpu_bytes": 2_000 + rank,
        }
        for rank in range(2)
    ]

    result = aggregate_reports(
        reports,
        per_gpu_batch=8,
        warmup_steps=10,
        world_size=2,
    )

    assert result["world_size"] == 2
    assert result["per_gpu_batch"] == 8
    assert result["global_batch"] == 16
    assert result["measured_images"] == 1_600
    assert result["aggregate_images_per_second"] == pytest.approx(1_600 / 11.0)


def test_aggregate_reports_rejects_missing_rank():
    with pytest.raises(ValueError, match="6 ranks"):
        aggregate_reports([], per_gpu_batch=64, warmup_steps=2)
    with pytest.raises(ValueError, match="positive integer"):
        aggregate_reports([], per_gpu_batch=8, warmup_steps=2, world_size=0)


def test_candidate_processes_requires_exact_script_and_token(tmp_path):
    commands = {
        "101": ["python", "/repo/scripts/d1/benchmark_wp8_training.py", "--candidate-token", "target"],
        "102": ["python", "/repo/scripts/d1/benchmark_wp8_training.py", "--candidate-token", "other"],
        "103": ["python", "/repo/scripts/d1/run_wp8_train.py", "--candidate-token", "target"],
    }
    for pid, tokens in commands.items():
        process = tmp_path / pid
        process.mkdir()
        (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in tokens) + b"\0")

    assert benchmark.candidate_processes("target", tmp_path) == (101,)
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark.candidate_processes("", tmp_path)


def test_cgroup_memory_snapshot_reads_v1_counters(tmp_path):
    (tmp_path / "memory.usage_in_bytes").write_text("100\n", encoding="utf-8")
    (tmp_path / "memory.limit_in_bytes").write_text("200\n", encoding="utf-8")
    (tmp_path / "memory.failcnt").write_text("3\n", encoding="utf-8")
    (tmp_path / "memory.oom_control").write_text("oom_kill_disable 0\noom_kill 4\n", encoding="utf-8")
    (tmp_path / "memory.stat").write_text("total_rss 60\ntotal_cache 30\n", encoding="utf-8")

    assert benchmark.cgroup_memory_snapshot(tmp_path) == {
        "usage_bytes": 100,
        "limit_bytes": 200,
        "fail_count": 3,
        "oom_kill_count": 4,
        "rss_bytes": 60,
        "cache_bytes": 30,
    }


def test_run_candidate_always_cleans_process_group_and_token(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 777
        returncode = 1

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            calls.append(("wait", timeout))
            return self.returncode

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        benchmark,
        "cgroup_memory_snapshot",
        lambda: {
            "usage_bytes": 100,
            "limit_bytes": 1_000,
            "fail_count": 0,
            "oom_kill_count": 0,
            "rss_bytes": 50,
            "cache_bytes": 25,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "terminate_process_group",
        lambda pgid: calls.append(("group", pgid)) or {"process_group": pgid, "sigkill": False},
    )
    monkeypatch.setattr(
        benchmark,
        "terminate_candidate_processes",
        lambda token: calls.append(("token", token))
        or {"matched_pids": [10], "sigkill_pids": [], "remaining_pids": []},
    )

    result = benchmark.run_candidate(
        ["fake-command"],
        log_path=tmp_path / "candidate.log",
        candidate_token="candidate-a",
        timeout_seconds=10,
        memory_headroom_bytes=10,
    )

    assert result["failure_reason"] == "subprocess_failed"
    assert calls == [("group", 777), ("wait", 2.0), ("token", "candidate-a")]

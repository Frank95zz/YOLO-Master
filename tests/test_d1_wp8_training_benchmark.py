"""Tests for the WP8 six-GPU training benchmark report logic."""

import pytest

from scripts.d1.benchmark_wp8_training import WORLD_SIZE, aggregate_reports, candidate_grid


def test_candidate_grid_is_complete_and_stable():
    assert candidate_grid([64, 128, 256], [8, 16]) == (
        (64, 8),
        (64, 16),
        (128, 8),
        (128, 16),
        (256, 8),
        (256, 16),
    )


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


def test_aggregate_reports_rejects_missing_rank():
    with pytest.raises(ValueError, match="six ranks"):
        aggregate_reports([], per_gpu_batch=64, warmup_steps=2)

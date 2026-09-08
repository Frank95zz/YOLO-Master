"""Consistency checks for the completed E2 engineering evidence, not accuracy gates."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.d1.evaluate_visdrone import validate_official_report

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "experiments/d1/manifests"


def read(name):
    return json.loads((MANIFESTS / name).read_text())


def test_e2_acceptance_links_immutable_evidence():
    acceptance = read("e2-acceptance.json")
    assert acceptance["status"] == "PASSED" and acceptance["e2_complete"]
    assert acceptance["formal_experiments_started"] is False and acceptance["test_dev_evaluated"] is False
    for name, digest in acceptance["evidence_sha256"].items():
        assert hashlib.sha256((MANIFESTS / name).read_bytes()).hexdigest() == digest
    technical = read("e2-engineering-results.json")
    assert technical["identity"]["commit"] == acceptance["training_code_commit"]
    assert technical["parameters"] == 3477987
    assert technical["contract"]["global_batch"] == 96
    assert technical["contract"]["schedule_epochs"] == 300


@pytest.mark.parametrize(("profile", "epoch", "count", "updates"), [("smoke", 2, 8, 2), ("benchmark", 3, 548, 204)])
def test_official_scores_bind_to_evaluated_checkpoint(profile, epoch, count, updates):
    official = read("e2-official-evaluation.json")["runs"][profile]
    result = read("e2-engineering-results.json")["profiles"][profile]
    validate_official_report(official)
    assert official["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert official["checkpoint_epoch"] == result["epochs"] == epoch
    assert official["image_count"] == result["val_samples"] == count
    assert result["optimizer_updates"] == updates and result["amp_retries"] == 0
    assert result["strict_reload"] and result["teacher_parameters"] == 0


def test_benchmark_eta_is_only_an_early_epoch_extrapolation():
    benchmark = read("e2-benchmark.json")
    assert benchmark["timed_epochs"] == [2, 3]
    assert benchmark["updates_per_epoch"] == 68 and len(benchmark["epochs"]) == 3
    median = sum(row["wall_seconds"] for row in benchmark["epochs"][1:]) / 2
    assert benchmark["median_epoch_seconds"] == pytest.approx(median)
    assert benchmark["estimated_300_epoch_hours"] == pytest.approx(median / 12)
    assert benchmark["estimated_300_epoch_gpu_hours"] == pytest.approx(median / 2)
    assert benchmark["resources"]["cpu_throttled_periods_delta"] == 0
    assert "warmup" in benchmark["limitation"]


def test_real_aux_and_parameter_updates_are_recorded():
    technical = read("e2-engineering-results.json")
    assert len(technical["aux_probes"]) == 6
    assert all(p["active_aux"] > 0 and p["zero_aux"] == 0 for p in technical["aux_probes"])
    result = technical["profiles"]["benchmark"]
    assert result["nine_adapter_branches_nonzero_gradient_all_ranks"]
    assert set(result["parameter_updates"]) == {"p3", "p4", "p5"}
    for value in result["parameter_updates"].values():
        assert value["router_delta_l2"] > 0 and value["residual_gain_abs_delta"] > 0

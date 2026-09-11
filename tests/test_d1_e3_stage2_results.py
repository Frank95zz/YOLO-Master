"""Offline regression for the E3 fixed-epoch official gain selection."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e3_stage2_summary", ROOT / "scripts/d1/summarize_e3_stage2.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def fixture_rows():
    rows = []
    for gain in summary.GAINS:
        for seed in summary.SEEDS:
            rows.append({
                "balance": 0.1, "z": 0.0, "gain": gain, "seed": seed,
                **{k: 0.08 + gain / 1000 + seed / 10000 for k in summary.METRICS},
                "training_gpu_hours": 2.0, "reused_stage1": gain == 0.1,
            })
    defaults = [{**r, "balance": 0.01, "z": 0.001, "gain": 0.1} for r in rows[:3]]
    return rows, defaults


def test_stable_order_cost_and_unrounded_selection():
    rows, defaults = fixture_rows()
    a = summary.summarize_rows(rows, defaults)
    assert a == summary.summarize_rows(list(reversed(rows)), list(reversed(defaults)))
    assert a["ranking"] == [0.3, 0.1, 0.03, 0.0]
    assert a["new_training_gpu_hours"] == 18
    assert a["AUX_STAR"]["latent_aux_gain"] == 0.3
    assert a["groups"][0]["paired_vs_gain0"]["AP_differences"] == [0, 0, 0]


@pytest.mark.parametrize("field,value", [
    ("AP_all", float("nan")), ("AP_50", float("inf")), ("AR_500", -0.1),
    ("AP_all", 8.0), ("AP_75", True), ("training_gpu_hours", 0),
    ("training_gpu_hours", float("inf")), ("seed", True), ("seed", 3),
    ("gain", True), ("balance", 0.01), ("z", 0.001),
    ("reused_stage1", True), ("reused_stage1", 0),
])
def test_invalid_rows_fail_closed(field, value):
    rows, defaults = fixture_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        summary.summarize_rows(rows, defaults)


@pytest.mark.parametrize("kind", ("missing", "duplicate", "default_missing", "default_duplicate", "wrong_default"))
def test_complete_paired_coverage_required(kind):
    rows, defaults = fixture_rows()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif kind == "default_missing":
        defaults.pop()
    elif kind == "default_duplicate":
        defaults[-1] = deepcopy(defaults[0])
    else:
        defaults[0]["balance"] = 0.1
    with pytest.raises(ValueError):
        summary.summarize_rows(rows, defaults)


def test_exact_ties_use_sd_then_cost_then_gain():
    rows, defaults = fixture_rows()
    for row in rows:
        row["AP_all"] = 0.08
    result = summary.summarize_rows(rows, defaults)
    assert result["ranking"] == [0, 0.03, 0.1, 0.3]
    for row in rows:
        if row["gain"] == 0.3:
            row["training_gpu_hours"] = 1.0
    assert summary.summarize_rows(rows, defaults)["ranking"][0] == 0.3
    for row in rows:
        if row["gain"] == 0.3:
            row["AP_all"] += (row["seed"] - 1) * 0.001
    assert summary.summarize_rows(rows, defaults)["ranking"][0] == 0


def test_archived_evidence_is_reproducible():
    evidence = ROOT / "experiments/d1/manifests/e3-stage2-official-20260911.json"
    if not evidence.exists():
        pytest.skip("Official evidence has not been archived yet")
    archived = json.loads(evidence.read_text(encoding="utf-8"))
    generated = summary.summarize_rows(archived["rows"], archived["default_rows"])
    for key, value in generated.items():
        assert archived[key] == value
    assert archived["AUX_STAR"]["latent_aux_gain"] == 0.1
    assert archived["new_runs"] == 9 and archived["unique_runs_both_stages"] == 36
    assert len(archived["official_reports"]) == 15
    for row in archived["rows"] + archived["default_rows"]:
        report = next(r for r in archived["official_reports"] if r["run_id"] == row["run_id"])
        assert {k: row[k] for k in summary.METRICS} == report["metrics"]
        assert report["checkpoint_epoch"] == 60 and report["image_count"] == 548


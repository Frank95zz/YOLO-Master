"""Stage2 orchestration tests do not launch GPU jobs or mutate stage1 evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.d1 import run_e3 as e3
from scripts.d1 import run_e3_stage2 as stage2


def summary():
    rows = [
        {**c, "AP_all": 0.08 + (0.002 if c["balance"] == 0.1 and c["z"] == 0 else 0),
         "training_gpu_hours": 2.0}
        for c in e3.candidates()
    ]
    groups, winner = e3.rank_groups(rows)
    return {"rows": rows, "groups": groups, "selected_nonzero": winner, "seeds": [0, 1, 2], "stage2_new_runs": 9}


def test_candidate_coverage_and_reuse():
    selected, new, reused = stage2.stage2_candidates(summary(), e3)
    assert (selected["balance"], selected["z"]) == (0.1, 0)
    assert [(c["gain"], c["seed"]) for c in new] == [(g, s) for g in (0, 0.03, 0.3) for s in (0, 1, 2)]
    assert [c["seed"] for c in reused] == [0, 1, 2]
    assert all(c["gain"] == 0.1 for c in reused)
    assert len({e3.key(c) for c in new + reused}) == 12


@pytest.mark.parametrize("fault", ["missing", "duplicate", "ranking", "winner", "seed", "count", "nan"])
def test_reject_invalid_summary(fault):
    value = summary()
    if fault == "missing":
        value["rows"].pop()
    elif fault == "duplicate":
        value["rows"][-1] = copy.deepcopy(value["rows"][0])
    elif fault == "ranking":
        value["groups"][0]["AP_mean"] += 0.1
    elif fault == "winner":
        value["selected_nonzero"] = {**value["selected_nonzero"], "z": 0.01}
    elif fault == "seed":
        value["seeds"] = [0, 1]
    elif fault == "count":
        value["stage2_new_runs"] = 12
    else:
        value["rows"][0]["AP_all"] = float("nan")
    with pytest.raises(ValueError):
        stage2.stage2_candidates(value, e3)


def test_all_zero_overall_winner_does_not_disable_gain_sweep():
    value = summary()
    for row in value["rows"]:
        if row["balance"] == row["z"] == 0:
            row["AP_all"] = 0.9
    value["groups"], value["selected_nonzero"] = e3.rank_groups(value["rows"])
    selected, new, _ = stage2.stage2_candidates(value, e3)
    assert selected["balance"] or selected["z"]
    assert len(new) == 9


def queue_fixture(tmp_path, monkeypatch, completed=False):
    _, new, reused = stage2.stage2_candidates(summary(), e3)
    plan = {"new_candidates": new, "reused_run_ids": [e3.key(c) for c in reused]}
    fake = SimpleNamespace(
        spec_for=lambda mat, c: {"run_id": e3.key(c), "output": str(tmp_path / "reports" / e3.key(c))},
        completed=Mock(return_value=completed), launch_train=Mock(), validate_run=Mock(), write_json=Mock(),
    )
    monkeypatch.setattr(stage2, "verify_plan", Mock(return_value={}))
    monkeypatch.setattr(stage2, "verify_exports", Mock())
    return plan, fake


def test_fresh_queue_runs_exactly_nine_without_resume(tmp_path, monkeypatch):
    plan, fake = queue_fixture(tmp_path, monkeypatch)
    result = stage2.run_queue(tmp_path, plan, fake)
    assert result["status"] == "STAGE2_TRAINED_AWAITING_OFFICIAL_MATLAB"
    assert result["AUX_STAR"] is None
    assert fake.launch_train.call_count == fake.validate_run.call_count == 9
    assert all(call.kwargs == {"resume": False} for call in fake.launch_train.call_args_list)


def test_completed_queue_is_revalidated_not_retrained(tmp_path, monkeypatch):
    plan, fake = queue_fixture(tmp_path, monkeypatch, completed=True)
    stage2.run_queue(tmp_path, plan, fake)
    fake.launch_train.assert_not_called()
    assert fake.validate_run.call_count == 9
    assert stage2.verify_exports.call_count == 9


@pytest.mark.parametrize("resume_file,authorized,allowed", [(True, False, False), (True, True, True), (False, True, False)])
def test_partial_trial_needs_explicit_validated_resume(tmp_path, monkeypatch, resume_file, authorized, allowed):
    plan, fake = queue_fixture(tmp_path, monkeypatch)
    output = Path(fake.spec_for({}, plan["new_candidates"][0])["output"])
    output.mkdir(parents=True)
    if resume_file:
        (output / "resume.pt").touch()
    if allowed:
        stage2.run_queue(tmp_path, plan, fake, allow_resume=authorized)
        assert fake.launch_train.call_args_list[0].kwargs == {"resume": True}
    else:
        with pytest.raises(ValueError):
            stage2.run_queue(tmp_path, plan, fake, allow_resume=authorized)
        fake.launch_train.assert_not_called()


def test_training_failure_stops_queue(tmp_path, monkeypatch):
    plan, fake = queue_fixture(tmp_path, monkeypatch)
    fake.launch_train.side_effect = RuntimeError("resource conflict or failed training")
    with pytest.raises(RuntimeError):
        stage2.run_queue(tmp_path, plan, fake)
    assert fake.launch_train.call_count == 1
    fake.validate_run.assert_not_called()


def test_changed_plan_stops_before_training(tmp_path, monkeypatch):
    plan, fake = queue_fixture(tmp_path, monkeypatch)
    stage2.verify_plan.side_effect = ValueError("Changed checksum")
    with pytest.raises(ValueError):
        stage2.run_queue(tmp_path, plan, fake)
    fake.launch_train.assert_not_called()


def test_prediction_exports_require_all_twelve_checkpoints(tmp_path):
    spec = {"identity": {"run_id": "test"}, "output": str(tmp_path)}
    fake = SimpleNamespace(file_sha=lambda path: "sha")
    for epoch in range(5, 61, 5):
        root = tmp_path / "official" / f"epoch-{epoch:03d}"
        root.mkdir(parents=True)
        (root / "report.json").write_text(json.dumps({
            "status": "PASSED", "run_identity": spec["identity"], "checkpoint_epoch": epoch,
            "seen": 548, "checkpoint_sha256": "sha", "prediction_export_sha256": "sha",
        }))
    stage2.verify_exports(spec, fake)
    (tmp_path / "official/epoch-060/report.json").unlink()
    with pytest.raises(FileNotFoundError):
        stage2.verify_exports(spec, fake)


def test_no_approval_fails_before_import_or_launch(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        stage2.main(["run", "--workspace", str(tmp_path), "--code-root", str(tmp_path / "code")])

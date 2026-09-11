"""Run the nine registered gain trials against an unchanged E3 training checkout."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path


def stage2_candidates(summary, e3):
    groups, selected = e3.rank_groups(summary["rows"])
    if groups != summary["groups"] or selected != summary["selected_nonzero"]:
        raise ValueError("Stage1 ranking does not match its complete official rows")
    if summary["seeds"] != [0, 1, 2] or summary["stage2_new_runs"] != 9:
        raise ValueError("Unregistered stage2 coverage")
    new = [
        {"balance": selected["balance"], "z": selected["z"], "gain": gain, "seed": seed}
        for gain in e3.EXPECTED["stage2_gains"]
        if gain != e3.EXPECTED["stage1_gain"]
        for seed in e3.EXPECTED["seeds"]
    ]
    if len(new) != 9 or len({e3.key(c) for c in new}) != 9:
        raise ValueError("Expected nine unique new trials")
    reused = [{**new[seed], "gain": e3.EXPECTED["stage1_gain"]} for seed in range(3)]
    return selected, new, reused


def verify_exports(spec, e3):
    for epoch in range(5, 61, 5):
        root = Path(spec["output"]) / "official" / f"epoch-{epoch:03d}"
        report = json.loads((root / "report.json").read_text())
        if (
            report.get("status") != "PASSED"
            or report.get("run_identity") != spec["identity"]
            or report.get("checkpoint_epoch") != epoch
            or report.get("seen") != 548
            or report.get("checkpoint_sha256") != e3.file_sha(root / "checkpoint.pt")
            or report.get("prediction_export_sha256") != e3.file_sha(root / "predictions-txt/export.json")
        ):
            raise ValueError("Missing or mismatched scheduled prediction export")


def make_plan(workspace, e3):
    mat = e3.matrix(workspace)
    gate = e3.gates(workspace)
    summary = e3.summarize_stage1(workspace)
    if summary["identity"] != mat["identity"] or gate["identity"] != mat["identity"]:
        raise ValueError("Stage1/gate identity changed")
    selected, new, reused = stage2_candidates(summary, e3)
    reused_ids = [e3.spec_for(mat, c)["run_id"] for c in reused]
    seconds = []
    for run_id in reused_ids:
        for path in (workspace / "jobs").glob(f"{run_id}-*.json"):
            attempt = json.loads(path.read_text())
            if attempt["status"] == "RETURNED" and attempt["returncode"] == 0:
                seconds.append(attempt["seconds"])
    if len(seconds) != 3:
        raise ValueError("Expected three complete successful reference attempts for ETA")
    return {
        "schema_version": "d1-e3-stage2-plan-v1",
        "identity": mat["identity"],
        "execution_identity": mat["execution_identity"],
        "orchestrator_sha256": e3.file_sha(Path(__file__)),
        "matrix_sha256": e3.file_sha(workspace / "matrix.json"),
        "gates_sha256": e3.file_sha(workspace / "gates.json"),
        "stage1_summary_sha256": e3.file_sha(workspace / "stage1-summary.json"),
        "selected_nonzero": selected,
        "new_candidates": new,
        "reused_run_ids": reused_ids,
        "screen_epochs": 60,
        "schedule_epochs": 300,
        "initialization": "same-seed original fresh initialization, not stage1 trained weights",
        "estimated_new_training_seconds": statistics.mean(seconds) * 9,
        "estimated_reference_run_seconds": seconds,
        "eta_scope": "Training, diagnostics, validation and GPU exports; MATLAB/transfer separate",
        "AUX_STAR": None,
    }


def verify_plan(workspace, plan, e3):
    mat = e3.matrix(workspace)
    if mat["identity"] != plan["identity"] or mat["execution_identity"] != plan["execution_identity"]:
        raise ValueError("Training identity changed")
    for name, field in (
        ("matrix.json", "matrix_sha256"),
        ("gates.json", "gates_sha256"),
        ("stage1-summary.json", "stage1_summary_sha256"),
    ):
        if e3.file_sha(workspace / name) != plan[field]:
            raise ValueError(f"Stage2 input changed: {name}")
    if e3.file_sha(Path(__file__)) != plan["orchestrator_sha256"]:
        raise ValueError("Stage2 orchestrator changed")
    summary = json.loads((workspace / "stage1-summary.json").read_text())
    selected, new, reused = stage2_candidates(summary, e3)
    if (
        selected != plan["selected_nonzero"]
        or new != plan["new_candidates"]
        or [e3.spec_for(mat, c)["run_id"] for c in reused] != plan["reused_run_ids"]
        or plan["screen_epochs"] != 60
        or plan["schedule_epochs"] != 300
    ):
        raise ValueError("Unregistered stage2 plan")
    return mat


def run_queue(workspace, plan, e3, allow_resume=False):
    for index, candidate in enumerate(plan["new_candidates"]):
        mat = verify_plan(workspace, plan, e3)
        spec = e3.spec_for(mat, candidate)
        output = Path(spec["output"])
        state = {
            "status": "STAGE2_RUNNING", "completed": index, "total": 9,
            "run_id": spec["run_id"], "candidate": candidate, "updated_unix": time.time(),
        }
        e3.write_json(workspace / "stage2-status.json", state)
        if not e3.completed(workspace, spec):
            has_resume = (output / "resume.pt").exists()
            if has_resume and not allow_resume:
                raise ValueError("Interrupted stage2 trial requires explicit --resume")
            if not has_resume and (output.exists() or any((workspace / "jobs").glob(f"{spec['run_id']}-*.json"))):
                raise ValueError("Partial trial without a resume checkpoint; inspect before retry")
            e3.launch_train(workspace, candidate, "screen", resume=has_resume)
        e3.validate_run(workspace, spec)
        verify_exports(spec, e3)
        state.update(status="STAGE2_RUN_COMPLETE", completed=index + 1, updated_unix=time.time())
        e3.write_json(workspace / "stage2-status.json", state)
        print(json.dumps(state), flush=True)
    result = {
        "status": "STAGE2_TRAINED_AWAITING_OFFICIAL_MATLAB",
        "completed": 9, "total": 9, "reused_stage1_runs": plan["reused_run_ids"],
        "AUX_STAR": None, "updated_unix": time.time(),
    }
    e3.write_json(workspace / "stage2-status.json", result)
    e3.write_json(workspace / "status.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.approved:
        raise ValueError("Explicit stage2 approval required")
    workspace, code_root = args.workspace.resolve(), args.code_root.resolve()
    if workspace.is_relative_to(code_root):
        raise ValueError("Use the existing external E3 workspace")
    sys.path.insert(0, str(code_root))
    e3 = importlib.import_module("scripts.d1.run_e3")
    if e3.ROOT.resolve() != code_root:
        raise ValueError("Imported a different training checkout")
    import fcntl
    import os

    with (workspace / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = workspace / "stage2-plan.json"
        if args.mode == "prepare":
            plan = make_plan(workspace, e3)
            e3.immutable(plan_path, e3.encoded(plan))
            resources = e3.check_resources(workspace)
            e3.write_json(workspace / "stage2-preflight.json", {
                "status": "PASSED", "plan_sha256": e3.file_sha(plan_path),
                "resources": resources, "updated_unix": time.time(),
            })
            result = plan
        else:
            plan = json.loads(plan_path.read_text())
            verify_plan(workspace, plan, e3)
            receipt = json.loads((workspace / "stage2-preflight.json").read_text())
            if receipt["status"] != "PASSED" or receipt["plan_sha256"] != e3.file_sha(plan_path):
                raise ValueError("Stage2 preflight does not match the approved plan")
            resources = e3.check_resources(workspace)
            e3.write_json(workspace / f"stage2-launch-{time.time_ns()}.json", {
                "pid": os.getpid(), "plan_sha256": e3.file_sha(plan_path),
                "orchestrator_sha256": plan["orchestrator_sha256"],
                "execution_identity": plan["execution_identity"], "resources": resources,
                "resume_authorized": args.resume, "updated_unix": time.time(),
            })
            try:
                result = run_queue(workspace, plan, e3, allow_resume=args.resume)
            except Exception as exc:
                result = {"status": "STAGE2_FAILED_OR_BLOCKED", "error": repr(exc), "updated_unix": time.time()}
                e3.write_json(workspace / "stage2-status.json", result)
                e3.write_json(workspace / "status.json", result)
                raise
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

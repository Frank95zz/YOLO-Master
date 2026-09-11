"""Verify and archive the registered E3 gain sweep without launching training."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

METRICS = ("AP_all", "AP_50", "AP_75", "AR_1", "AR_10", "AR_100", "AR_500")
GAINS = (0.0, 0.03, 0.1, 0.3)
SEEDS = (0, 1, 2)


def check_row(row):
    if type(row["seed"]) is not int or row["seed"] not in SEEDS:
        raise ValueError("Expected integer seed 0/1/2")
    for name in METRICS:
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Official metrics must be finite ratios in 0..1")
    cost = row["training_gpu_hours"]
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
        raise ValueError("Missing or invalid GPU-hours")


def paired(values, reference):
    differences = [a - b for a, b in zip(values, reference)]
    mean, std = statistics.mean(differences), statistics.stdev(differences)
    margin = 4.302652729911275 * std / math.sqrt(3)
    return {
        "AP_differences": differences,
        "AP_mean_difference": mean,
        "AP_sample_std": std,
        "positive_seeds": sum(x > 0 for x in differences),
        "exploratory_95pct_t_interval": [mean - margin, mean + margin],
    }


def summarize_rows(rows, default_rows):
    """Rank raw official AP, not rounded display scores or post-hoc tolerances."""
    expected = {(gain, seed) for gain in GAINS for seed in SEEDS}
    actual = [(r["gain"], r["seed"]) for r in rows]
    if len(rows) != 12 or set(actual) != expected:
        raise ValueError("Expected all four gains and three seeds, without duplicates")
    for row in rows:
        check_row(row)
        if row["balance"] != 0.1 or row["z"] != 0 or type(row["gain"]) not in (int, float):
            raise ValueError("Unregistered stage2 candidate")
        if type(row["reused_stage1"]) is not bool or row["reused_stage1"] != (row["gain"] == 0.1):
            raise ValueError("Reuse identity mismatch")
    if len(default_rows) != 3 or {r["seed"] for r in default_rows} != set(SEEDS):
        raise ValueError("Expected three paired default seeds")
    for row in default_rows:
        check_row(row)
        if (row["balance"], row["z"], row["gain"]) != (0.01, 0.001, 0.1):
            raise ValueError("Wrong default reference")
    rows = sorted(rows, key=lambda r: (r["gain"], r["seed"]))
    default_rows = sorted(default_rows, key=lambda r: r["seed"])
    zero = [r["AP_all"] for r in rows if r["gain"] == 0]
    default = [r["AP_all"] for r in default_rows]
    groups = []
    for gain in GAINS:
        selected = [r for r in rows if r["gain"] == gain]
        aps = [r["AP_all"] for r in selected]
        groups.append({
            "gain": gain,
            "AP_values": aps,
            "metrics_mean": {k: statistics.mean(r[k] for r in selected) for k in METRICS},
            "metrics_sample_std": {k: statistics.stdev(r[k] for r in selected) for k in METRICS},
            "training_gpu_hours_mean": statistics.mean(r["training_gpu_hours"] for r in selected),
            "paired_vs_gain0": paired(aps, zero),
            "paired_vs_default": paired(aps, default),
        })
    ranked = sorted(groups, key=lambda g: (
        -g["metrics_mean"]["AP_all"], g["metrics_sample_std"]["AP_all"],
        g["training_gpu_hours_mean"], g["gain"],
    ))
    return {
        "rows": rows, "default_rows": default_rows, "groups": groups,
        "ranking": [g["gain"] for g in ranked],
        "AUX_STAR": {
            "balance_loss_coeff": 0.1, "router_z_loss_coeff": 0.0,
            "latent_aux_gain": ranked[0]["gain"], "mixture_aux_budget": 3.0,
        },
        "new_training_gpu_hours": sum(r["training_gpu_hours"] for r in rows if not r["reused_stage1"]),
        "statistical_scope": "n=3 paired training seeds used in selection; exploratory intervals, not independent confirmation",
        "metric_units": "ratios 0..1; multiply by 100 for AP points",
        "selection_rule": "raw AP mean descending; sample SD, comparable mean GPU-hours and gain ascending",
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_report(report, spec, output, input_sha, e3):
    official = e3.validate_official_report(report)
    exported = read_json(output / "report.json")
    export_path = output / "predictions-txt/export.json"
    export = read_json(export_path)
    if (
        official["run_id"] != spec["run_id"]
        or official["candidate"] != spec["identity"]["candidate"]
        or official["checkpoint_epoch"] != 60
        or official["image_count"] != 548
        or official["input_manifest_sha256"] != input_sha
        or exported["run_identity"] != spec["identity"]
        or exported["checkpoint_epoch"] != 60
        or exported["seen"] != 548
        or exported["status"] != "PASSED"
        or exported["strict_reload"] is not True
        or exported["teacher_parameters"] != 0
        or official["checkpoint_sha256"] != exported["checkpoint_sha256"]
        or e3.file_sha(output / "checkpoint.pt") != official["checkpoint_sha256"]
        or e3.file_sha(export_path) != official["export_sha256"]
        or official["toolkit_manifest_sha256"] != e3.file_sha(e3.ROOT / "experiments/d1/manifests/visdrone-toolkit.json")
    ):
        raise ValueError("Official report does not match its registered run/checkpoint/export/toolkit")
    files = export["files_sha256"]
    if len(files) != 548 or set(files) != {p.name for p in export_path.parent.glob("*.txt")}:
        raise ValueError("Prediction file coverage differs")
    for name, digest in files.items():
        if Path(name).name != name or e3.file_sha(export_path.parent / name) != digest:
            raise ValueError("Prediction checksum differs")
    for metric in METRICS:
        if not math.isclose(official["metrics"][metric] * 100, official["metrics_percent"][metric], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Official ratio/percent conversion differs")
    return official, files


def archive(args):
    sys.path.insert(0, str(args.code_root.resolve()))
    e3 = importlib.import_module("scripts.d1.run_e3")
    if e3.ROOT.resolve() != args.code_root.resolve():
        raise ValueError("Imported a different training checkout")
    # Load the immutable queue copy so its self-hash can still verify the approved plan.
    module_spec = importlib.util.spec_from_file_location("e3_stage2_queue", args.queue_script)
    queue = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(queue)
    w = args.workspace.resolve()
    plan = read_json(w / "stage2-plan.json")
    mat = queue.verify_plan(w, plan, e3)
    manifest = read_json(args.input_manifest)
    manifest_sha = e3.file_sha(args.input_manifest)
    if (
        manifest["identity"] != plan["identity"]
        or manifest["execution_identity"] != plan["execution_identity"]
        or manifest["stage2_plan_sha256"] != e3.file_sha(w / "stage2-plan.json")
        or len(manifest["runs"]) != 9
        or {r["run_id"] for r in manifest["runs"]} != {e3.spec_for(mat, c)["run_id"] for c in plan["new_candidates"]}
    ):
        raise ValueError("Official input manifest does not match the registered stage2")
    stage1 = read_json(w / "stage1-summary.json")
    candidates = plan["new_candidates"] + [
        {"balance": 0.1, "z": 0.0, "gain": 0.1, "seed": s} for s in SEEDS
    ]
    default = [{"balance": 0.01, "z": 0.001, "gain": 0.1, "seed": s} for s in SEEDS]
    rows, default_rows, raw, installs, prediction_hashes = [], [], [], [], {}
    for candidate in candidates + default:
        spec = e3.spec_for(mat, candidate)
        e3.validate_run(w, spec)
        queue.verify_exports(spec, e3)
        out = Path(spec["output"]) / "official/epoch-060"
        is_new = candidate in plan["new_candidates"]
        source = args.official_results / spec["run_id"] / "official-matlab.json" if is_new else out / "official-matlab.json"
        report = read_json(source)
        if is_new:
            expected_input = manifest_sha
            manifest_run = next(r for r in manifest["runs"] if r["run_id"] == spec["run_id"])
            if any(report[k] != manifest_run[k] for k in ("candidate", "checkpoint_epoch", "checkpoint_sha256", "export_sha256")):
                raise ValueError("Scoring report differs from its input manifest")
        else:
            # First-stage reports remain immutable; bind them to the recorded stage1 archive.
            archived = read_json(args.stage1_evidence)
            matches = [r for r in archived["official_reports"] if r["run_id"] == spec["run_id"]]
            if len(matches) != 1:
                raise ValueError("Missing stage1 official provenance")
            expected_input = matches[0]["input_manifest_sha256"]
            for name in ("metrics", "checkpoint_sha256", "export_sha256", "candidate"):
                if report[name] != matches[0][name]:
                    raise ValueError("Reused/default official provenance changed")
        official, files = verify_report(report, spec, out, expected_input, e3)
        prediction_hashes[spec["run_id"]] = files
        attempts = [read_json(p) for p in sorted((w / "jobs").glob(f"{spec['run_id']}-*.json"))]
        if not attempts or any(
            a["identity"] != spec["identity"]
            or (is_new and a.get("execution_identity") != plan["execution_identity"])
            or not math.isfinite(a["gpu_hours"]) or a["gpu_hours"] <= 0
            or not math.isclose(a["gpu_hours"], a["seconds"] * 6 / 3600, rel_tol=1e-12)
            for a in attempts
        ):
            raise ValueError("Cost ledger mismatch")
        if not any(a.get("status") == "RETURNED" and a.get("returncode") == 0 for a in attempts):
            raise ValueError("No successful training attempt")
        row = {
            **candidate, **official["metrics"], "run_id": spec["run_id"],
            "training_gpu_hours": sum(a["gpu_hours"] for a in attempts),
            "attempt_count": len(attempts),
            "reused_stage1": not is_new,
        }
        if not is_new:
            reference = next(r for r in stage1["rows"] if all(r[k] == candidate[k] for k in candidate))
            if any(row[k] != reference[k] for k in (*METRICS, "training_gpu_hours")):
                raise ValueError("Reused metrics/cost differ from the immutable stage1 summary")
        (default_rows if candidate in default else rows).append(row)
        sanitized = dict(official)
        sanitized["mean2_implementation"] = "scripts/d1/visdrone_matlab_compat/mean2.m"
        sanitized["source_report_sha256"] = e3.file_sha(source)
        raw.append(sanitized)
        if is_new:
            destination = out / "official-matlab.json"
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError("Refusing to replace a different existing official result")
            installs.append((source, destination))
    result = summarize_rows(rows, default_rows)
    result.update({
        "schema_version": "d1-e3-stage2-official-v1", "status": "E3_AUX_SELECTION_COMPLETE",
        "identity": plan["identity"], "execution_identity": plan["execution_identity"],
        "stage2_plan_sha256": e3.file_sha(w / "stage2-plan.json"),
        "input_manifest_sha256": manifest_sha, "summary_script_sha256": e3.file_sha(Path(__file__)),
        "stage1_evidence_sha256": e3.file_sha(args.stage1_evidence),
        "official_reports": sorted(raw, key=lambda r: r["run_id"]),
        "new_runs": 9, "reused_stage1_runs": 3, "unique_runs_both_stages": 36,
        "screen_epochs": 60, "schedule_epochs": 300, "validation_images": 548,
        "official_checkpoint_policy": "fixed epoch 60; no best-seed or best-epoch selection",
        "all36_training_gpu_hours": sum(r["training_gpu_hours"] for r in stage1["rows"]) + result["new_training_gpu_hours"],
        "cost_scope": "Six reserved GPUs times training job wall seconds including diagnostics/validation/exports and prior attempts; excludes preparation, independent diagnostics, transfer and MATLAB",
        "stage2_matlab_seconds": args.matlab_seconds,
        "stage2_queue_wall_seconds": args.queue_seconds,
        "P1_cost_reduction_claim": False, "new_training_started": False,
        "official_curve_scope": "Only fixed epoch60 scored; all 432 scheduled exports retained, not a complete official standard-best curve",
    })
    # Compare exported bytes, not just identical aggregate scores, for the aux-off control.
    equivalence = []
    for seed in SEEDS:
        zero_id = e3.spec_for(mat, {"balance": 0.0, "z": 0.0, "gain": 0.1, "seed": seed})["run_id"]
        old = read_json(w / "reports" / zero_id / "official/epoch-060/predictions-txt/export.json")
        for name, digest in old["files_sha256"].items():
            if e3.file_sha(w / "reports" / zero_id / "official/epoch-060/predictions-txt" / name) != digest:
                raise ValueError("Stage1 aux-off prediction changed")
        new_id = e3.spec_for(mat, {"balance": 0.1, "z": 0.0, "gain": 0.0, "seed": seed})["run_id"]
        equivalence.append({"seed": seed, "prediction_files": 548, "byte_identical": prediction_hashes[new_id] == old["files_sha256"]})
    result["gain0_vs_stage1_zero_prediction_check"] = equivalence
    # Publish only after every candidate has passed; raw scoring files are never rewritten.
    for source, destination in installs:
        e3.immutable(destination, source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    e3.write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "code-root", "queue-script", "official-results", "input-manifest", "stage1-evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--matlab-seconds", type=float, required=True)
    parser.add_argument("--queue-seconds", type=float, required=True)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in (args.matlab_seconds, args.queue_seconds)):
        raise ValueError("Measured timing must be finite and positive")
    result = archive(args)
    print(json.dumps({k: result[k] for k in ("status", "ranking", "AUX_STAR", "new_training_gpu_hours", "all36_training_gpu_hours")}, indent=2))


if __name__ == "__main__":
    main()

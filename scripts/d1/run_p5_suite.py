"""Run the approved three-architecture P5 screen with a common accelerated P3 implementation."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time

from scripts.d1 import run_p1p2 as e1
from scripts.d1 import run_p5_ablation as p5
from scripts.d1.run_wp8_p1_control import write_json
from ultralytics.nn.foundation.cache import sha256_file

ORDER = ("BASE", "DW", "BN64")


def comparison(results, selection):
    """Apply the preregistered fixed-epoch AP and measured-cost gates without selecting a new budget."""
    if set(results) != set(ORDER):
        raise ValueError("All three completed variants are required")
    base = results["BASE"]
    rows = {}
    for variant in ORDER:
        row = results[variant]
        metrics = row["final"]["last"]["official"]
        cost = row["active_job_GPUh"]
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError("Invalid measured GPU-hours")
        for key in ("AP_all", "AP_75", "AP_large", "AP_small"):
            if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
                raise ValueError("Invalid official AP")
        drops = {
            key: 100 * (base["final"]["last"]["official"][key] - metrics[key])
            for key in ("AP_all", "AP_75", "AP_large")
        }
        rows[variant] = {
            "official": metrics,
            "drop_points_vs_BASE": drops,
            "active_job_GPUh": cost,
            "retained": variant == "BASE"
            or (
                drops["AP_all"] <= selection["max_ap_drop_points"]
                and drops["AP_75"] <= selection["max_ap75_drop_points"]
                and drops["AP_large"] <= selection["max_ap_large_drop_points"]
                and cost < base["active_job_GPUh"]
            ),
        }
    return rows


class P5Pipeline(e1.Pipeline):
    """Reuse E1 job accounting, with safe epoch-boundary stop and strict P5 result validation."""

    def stop(self, *_):
        self.interrupted = True
        if self.child is None or self.child.poll() is not None:
            return
        import psutil

        for process in psutil.Process(self.child.pid).children(recursive=False):
            try:
                command = process.cmdline()
                if "scripts.d1.run_p5_ablation" not in command or "train" not in command:
                    continue
                status = Path(f"/proc/{process.pid}/status").read_text()
                caught = next(line.split()[1] for line in status.splitlines() if line.startswith("SigCgt:"))
                if int(caught, 16) & (1 << (signal.SIGUSR1 - 1)):
                    process.send_signal(signal.SIGUSR1)
            except (psutil.NoSuchProcess, FileNotFoundError):
                pass

    def execute(self):
        if not self.args.approved:
            raise ValueError("Explicit --approved is required")
        self.matrix = p5.load_matrix(self.workspace)
        if self.matrix.get("p3_upsample_mode") != "separable_bilinear2x":
            raise ValueError("This suite requires a uniformly registered accelerated P3")
        launch = self.workspace / "suite-launch.json"
        if launch.exists():
            raise FileExistsError("Suite already launched; inspect state before explicit recovery")
        for variant in ORDER:
            for parent in ("runs", "reports"):
                if (self.workspace / parent / f"P5-{variant}").exists():
                    raise FileExistsError(f"Existing state for {variant}; refusing overwrite")
        write_json(
            launch,
            {
                "identity": self.matrix["identity"],
                "variants": list(ORDER),
                "approved": True,
                "p3_upsample_mode": self.matrix["p3_upsample_mode"],
                "seed": 0,
                "screen_epochs": 50,
                "schedule_epochs": 100,
                "global_batch": 384,
                "world_size": 6,
                "workers_per_rank": 4,
                "started_unix": time.time(),
                "scope": "Section 6.5 fresh paired screen; old E1 and paused P5 results are not reused",
            },
        )
        results = {}
        for variant in ORDER:
            if self.interrupted:
                raise InterruptedError("Queue interrupted before the next variant")
            run_id = f"P5-{variant}"
            output = self.workspace / "reports" / run_id
            weights = self.workspace / "runs" / run_id / "weights"
            seconds = self.child_run(
                run_id,
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    "--nproc-per-node=6",
                    "--module",
                    "scripts.d1.run_p5_ablation",
                    "train",
                    "--workspace",
                    str(self.workspace),
                    "--variant",
                    variant,
                    "--approved",
                ],
            )
            result = e1.read_json(output / "training-result.json")
            if (result.get("status"), result.get("epochs"), result.get("optimizer_steps")) != ("completed", 50, 15450):
                raise ValueError(f"{run_id} did not complete its exact screening budget")
            if result["last_sha256"] != sha256_file(weights / "last.pt"):
                raise ValueError("Training checkpoint checksum mismatch")
            if result["resume_sha256"] != sha256_file(output / "resume.pt"):
                raise ValueError("Resume checksum mismatch")
            write_json(
                output / "cost.json",
                {
                    "seconds": seconds,
                    "reserved_gpus": 6,
                    "GPUh": seconds * 6 / 3600,
                    "scope": "Training job including startup and periodic evaluations; final evaluations separate",
                },
            )
            finals, final_seconds = {}, {}
            for name in ("last", "standard-best"):
                destination = output / "final" / name
                final_seconds[name] = self.child_run(
                    f"{run_id}-final-{name}",
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.d1.run_p5_ablation",
                        "evaluate",
                        "--workspace",
                        str(self.workspace),
                        "--run-id",
                        run_id,
                        "--checkpoint",
                        str(weights / f"{name}.pt"),
                        "--output",
                        str(destination),
                    ],
                )
                report = e1.read_json(destination / "report.json")
                e1.validate_evaluation(report, self.matrix, run_id, weights / f"{name}.pt")
                if name == "last" and report["checkpoint_epoch"] != 50:
                    raise ValueError("Final last checkpoint is not epoch 50")
                finals[name] = report
            results[variant] = {
                "training": result,
                "final": finals,
                "train_seconds": seconds,
                "final_job_seconds": final_seconds,
                "active_job_GPUh": (6 * seconds + sum(final_seconds.values())) / 3600,
                "six_gpu_reserved_hours": 6 * (seconds + sum(final_seconds.values())) / 3600,
                "downstream_parameters": self.matrix["parameters"][variant],
            }
            write_json(output / "completed-summary.json", results[variant])
        write_json(
            self.workspace / "suite-summary.json",
            {
                "status": "completed",
                "identity": self.matrix["identity"],
                "results": results,
                "comparison": comparison(results, self.matrix["contract"]["selection"]),
                "scope": "Single-seed section 6.5 screen, not statistical equivalence or P1 acceptance",
            },
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args(argv)
    if not args.approved:
        parser.error("Explicit --approved is required")
    args.workspace = args.workspace.resolve()
    pipeline = P5Pipeline(args)
    try:
        pipeline.execute()
    except BaseException as error:
        write_json(
            args.workspace / "status.json",
            {
                "status": "interrupted" if pipeline.interrupted else "failed",
                "error": repr(error),
                "pid": os.getpid(),
                "updated_unix": time.time(),
                "started_unix": pipeline.started,
            },
        )
        raise
    write_json(
        args.workspace / "status.json",
        {
            "status": "completed",
            "variants": list(ORDER),
            "pid": os.getpid(),
            "updated_unix": time.time(),
            "started_unix": pipeline.started,
        },
    )


if __name__ == "__main__":
    main()

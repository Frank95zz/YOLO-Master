#!/usr/bin/env python3
"""Benchmark D1 WP8 cached training across configurable GPU and DataLoader settings."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from itertools import product
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.d1.run_wp8_train import git_commit, write_json
except ModuleNotFoundError:
    from run_wp8_train import git_commit, write_json

from ultralytics.models.yolo.detect import D1FoundationDetectionTrainer
from ultralytics.utils import YAML


WORLD_SIZE = 6
TRAIN_SAMPLES = 118_287
VAL_SAMPLES = 5_000
FORMAL_EPOCHS = 100
DEFAULT_BATCHES = (64, 128, 256)
DEFAULT_WORKERS = (8, 16)
CGROUP_MEMORY_ROOT = Path("/sys/fs/cgroup/memory")
PROC_ROOT = Path("/proc")


def candidate_grid(
    per_gpu_batches: tuple[int, ...] | list[int], workers: tuple[int, ...] | list[int]
) -> tuple[tuple[int, int], ...]:
    if not per_gpu_batches or not workers:
        raise ValueError("batch and worker candidate lists must not be empty")
    if any(type(value) is not int or value <= 0 for value in (*per_gpu_batches, *workers)):
        raise ValueError("batch and worker candidates must be positive integers")
    return tuple(product(per_gpu_batches, workers))


def cgroup_memory_snapshot(root: Path = CGROUP_MEMORY_ROOT) -> dict[str, int]:
    """Return the cgroup-v1 memory counters needed to detect host-memory failures."""

    def read_int(name: str) -> int:
        path = root / name
        return int(path.read_text(encoding="utf-8").strip()) if path.is_file() else 0

    stats: dict[str, int] = {}
    stat_path = root / "memory.stat"
    if stat_path.is_file():
        for line in stat_path.read_text(encoding="utf-8").splitlines():
            name, value = line.split()
            if name in {"rss", "cache", "total_rss", "total_cache"}:
                stats[name] = int(value)
    oom_kills = 0
    oom_path = root / "memory.oom_control"
    if oom_path.is_file():
        for line in oom_path.read_text(encoding="utf-8").splitlines():
            name, value = line.split()
            if name == "oom_kill":
                oom_kills = int(value)
                break
    return {
        "usage_bytes": read_int("memory.usage_in_bytes"),
        "limit_bytes": read_int("memory.limit_in_bytes"),
        "fail_count": read_int("memory.failcnt"),
        "oom_kill_count": oom_kills,
        "rss_bytes": stats.get("total_rss", stats.get("rss", 0)),
        "cache_bytes": stats.get("total_cache", stats.get("cache", 0)),
    }


def candidate_processes(candidate_token: str, proc_root: Path = PROC_ROOT) -> tuple[int, ...]:
    """Find only benchmark descendants carrying the exact per-candidate token."""
    if not candidate_token:
        raise ValueError("candidate token must not be empty")
    if not proc_root.is_dir():
        return ()
    matches = []
    own_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            tokens = [value.decode(errors="replace") for value in (entry / "cmdline").read_bytes().split(b"\0") if value]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(Path(value).name == Path(__file__).name for value in tokens):
            continue
        if any(
            tokens[index] == "--candidate-token" and index + 1 < len(tokens) and tokens[index + 1] == candidate_token
            for index in range(len(tokens))
        ):
            matches.append(int(entry.name))
    return tuple(sorted(matches))


def _signal_processes(pids: tuple[int, ...], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def terminate_candidate_processes(
    candidate_token: str, *, proc_root: Path = PROC_ROOT, grace_seconds: float = 5.0
) -> dict[str, Any]:
    """Terminate reparented DataLoader workers without touching another candidate or run."""
    matched = candidate_processes(candidate_token, proc_root)
    _signal_processes(matched, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    remaining = candidate_processes(candidate_token, proc_root)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = candidate_processes(candidate_token, proc_root)
    forced = remaining
    _signal_processes(forced, signal.SIGKILL)
    if forced:
        time.sleep(0.2)
    final = candidate_processes(candidate_token, proc_root)
    return {
        "matched_pids": list(matched),
        "sigkill_pids": list(forced),
        "remaining_pids": list(final),
    }


def terminate_process_group(process_group: int, *, grace_seconds: float = 5.0) -> dict[str, Any]:
    """Terminate the isolated torchrun process group, including DataLoader descendants."""
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return {"process_group": process_group, "sigkill": False}
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return {"process_group": process_group, "sigkill": False}
        time.sleep(0.1)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return {"process_group": process_group, "sigkill": True}


def run_candidate(
    command: list[str],
    *,
    log_path: Path,
    candidate_token: str,
    timeout_seconds: float,
    memory_headroom_bytes: int,
) -> dict[str, Any]:
    """Run one candidate in isolation and always reclaim its complete process tree."""
    before = cgroup_memory_snapshot()
    peak_usage = before["usage_bytes"]
    peak_rss = before["rss_bytes"]
    reason = None
    returncode = 1
    process: subprocess.Popen | None = None
    group_cleanup: dict[str, Any] = {}
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = started + timeout_seconds
            while process.poll() is None:
                snapshot = cgroup_memory_snapshot()
                peak_usage = max(peak_usage, snapshot["usage_bytes"])
                peak_rss = max(peak_rss, snapshot["rss_bytes"])
                limit = snapshot["limit_bytes"]
                if limit and snapshot["rss_bytes"] >= limit - memory_headroom_bytes:
                    reason = "anonymous_memory_headroom_exhausted"
                    break
                if time.monotonic() >= deadline:
                    reason = "timeout"
                    break
                time.sleep(0.25)
            if reason is None:
                returncode = int(process.returncode)
            else:
                returncode = 124 if reason == "timeout" else 137
    finally:
        if process is not None:
            group_cleanup = terminate_process_group(process.pid)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                group_cleanup["direct_child_reaped"] = False
            else:
                group_cleanup["direct_child_reaped"] = True
        token_cleanup = terminate_candidate_processes(candidate_token)
    after = cgroup_memory_snapshot()
    if after["oom_kill_count"] > before["oom_kill_count"]:
        reason = "cgroup_oom_kill"
    elif returncode and reason is None:
        reason = "subprocess_failed"
    if token_cleanup["remaining_pids"]:
        reason = "process_cleanup_failed"
    return {
        "returncode": returncode,
        "failure_reason": reason,
        "wall_seconds_including_cleanup": time.monotonic() - started,
        "cgroup_memory_before": before,
        "cgroup_memory_after": after,
        "cgroup_peak_usage_bytes": peak_usage,
        "cgroup_peak_rss_bytes": peak_rss,
        "process_group_cleanup": group_cleanup,
        "candidate_process_cleanup": token_cleanup,
    }


def aggregate_reports(
    reports: list[dict[str, Any]],
    *,
    per_gpu_batch: int,
    warmup_steps: int,
    world_size: int = WORLD_SIZE,
) -> dict[str, Any]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("benchmark world size must be a positive integer")
    if len(reports) != world_size or {item["rank"] for item in reports} != set(range(world_size)):
        raise ValueError(f"benchmark requires exactly one report for each of {world_size} ranks")
    if not all(item.get("amp_enabled") is True for item in reports):
        raise ValueError("benchmark requires AMP to remain enabled on every rank")
    measured_batches = {item["measured_batches"] for item in reports}
    if len(measured_batches) != 1 or next(iter(measured_batches)) <= 0:
        raise ValueError("rank benchmark reports disagree on measured batch count")
    count = next(iter(measured_batches))
    elapsed = max(item["measured_seconds"] for item in reports)
    if elapsed <= 0:
        raise ValueError("benchmark elapsed time must be positive")
    images = per_gpu_batch * world_size * count
    throughput = images / elapsed
    formal_images = FORMAL_EPOCHS * (TRAIN_SAMPLES + VAL_SAMPLES)
    training_hours = FORMAL_EPOCHS * TRAIN_SAMPLES / throughput / 3600
    total_hours = formal_images / throughput / 3600
    return {
        "status": "passed",
        "world_size": world_size,
        "per_gpu_batch": per_gpu_batch,
        "global_batch": per_gpu_batch * world_size,
        "workers_per_rank": reports[0]["workers_per_rank"],
        "warmup_steps": warmup_steps,
        "measured_batches": count,
        "measured_images": images,
        "measured_seconds": elapsed,
        "aggregate_images_per_second": throughput,
        "mean_data_wait_ratio": sum(item["data_wait_ratio"] for item in reports) / world_size,
        "peak_gpu_bytes_by_rank": {str(item["rank"]): item["peak_gpu_bytes"] for item in reports},
        "estimated_train_hours_100_epochs": training_hours,
        "estimated_train_plus_val_hours": total_hours,
        "estimated_total_hours_with_15pct_overhead": total_hours * 1.15,
    }


class Timing:
    def __init__(self, warmup_steps: int) -> None:
        self.warmup_steps = warmup_steps
        self.batch_index = 0
        self.batch_started: float | None = None
        self.previous_batch_ended: float | None = None
        self.step_seconds = 0.0
        self.wait_seconds = 0.0
        self.measured_batches = 0

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_start(self, trainer) -> None:
        self._sync()
        self.previous_batch_ended = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_train_batch_start(self, _trainer) -> None:
        self._sync()
        now = time.perf_counter()
        wait = 0.0 if self.previous_batch_ended is None else now - self.previous_batch_ended
        self.batch_started = now
        if self.batch_index >= self.warmup_steps:
            self.wait_seconds += wait

    def on_train_batch_end(self, _trainer) -> None:
        self._sync()
        now = time.perf_counter()
        if self.batch_index >= self.warmup_steps and self.batch_started is not None:
            self.step_seconds += now - self.batch_started
            self.measured_batches += 1
        self.previous_batch_ended = now
        self.batch_index += 1


class BenchmarkTrainer(D1FoundationDetectionTrainer):
    def final_eval(self) -> None:
        """Skip checkpoint selection and validation in the throughput-only benchmark."""


def _wait_for(path: Path, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.2)


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.world_size == 1 and rank == -1 and local_rank == -1:
        rank = local_rank = 0
    elif rank < 0 or local_rank < 0 or world_size != args.world_size:
        raise RuntimeError(f"worker must run under torchrun with exactly {args.world_size} processes")
    global_batch = args.per_gpu_batch * world_size
    sample_count = global_batch * args.steps
    if args.sample_offset + sample_count > TRAIN_SAMPLES:
        raise ValueError("benchmark sample range exceeds COCO train2017")

    run_dir = args.workspace / "benchmarks" / "wp8-training" / args.run_id / f"b{args.per_gpu_batch}-w{args.workers}"
    split_file = run_dir / "inputs" / "train.txt"
    data_yaml = run_dir / "inputs" / "coco.yaml"
    if rank == 0:
        manifest = args.repo_root / "experiments/d1/manifests/coco2017-train2017.txt"
        all_paths = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        relative_paths = all_paths[args.sample_offset : args.sample_offset + sample_count]
        if len(relative_paths) != sample_count:
            raise ValueError("tracked COCO train manifest is shorter than the requested benchmark")
        absolute_paths = [str(args.data_root / path) for path in relative_paths]
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text("\n".join(absolute_paths) + "\n", encoding="utf-8")
        names = YAML.load(args.repo_root / "ultralytics/cfg/datasets/coco.yaml")["names"]
        YAML.save(data_yaml, {"path": str(args.data_root), "train": str(split_file), "val": str(split_file), "names": names})
    _wait_for(data_yaml)

    experiment = YAML.load(args.repo_root / "ultralytics/cfg/experiments/d1/p0-dinov3-vits16-coco2017.yaml")
    overrides = {
        **experiment,
        "data": str(data_yaml),
        "model": str((args.repo_root / experiment["model"]).resolve()),
        "seed": args.seed,
        "epochs": 1,
        "batch": global_batch,
        "nbs": global_batch,
        "workers": args.workers,
        "amp": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "warmup_epochs": 0.0,
        "cos_lr": False,
        "patience": 1,
        "save": False,
        "val": False,
        "plots": False,
        "pretrained": False,
        "fraction": 1.0,
        "cache": False,
        "compile": False,
        "device": ",".join(str(index) for index in range(world_size)),
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "verbose": False,
        "rect": False,
        "multi_scale": 0.0,
    }
    timing = Timing(args.warmup_steps)
    trainer = BenchmarkTrainer(
        overrides=overrides,
        feature_caches={"train": args.train_cache, "val": args.train_cache},
        trusted_feature_cache=True,
        max_open_feature_shards=args.max_open_shards,
        feature_prefetch_factor=args.prefetch_factor,
        amp_init_scale=args.amp_init_scale,
        amp_growth_interval=args.amp_growth_interval,
    )
    trainer.add_callback("on_train_start", timing.on_train_start)
    trainer.add_callback("on_train_batch_start", timing.on_train_batch_start)
    trainer.add_callback("on_train_batch_end", timing.on_train_batch_end)
    trainer.train()
    measured_seconds = timing.step_seconds + timing.wait_seconds
    report = {
        "status": "passed",
        "rank": rank,
        "local_rank": local_rank,
        "seed": args.seed,
        "sample_offset": args.sample_offset,
        "per_gpu_batch": args.per_gpu_batch,
        "workers_per_rank": args.workers,
        "actual_workers_per_rank": trainer.train_loader.num_workers,
        "prefetch_factor": trainer.train_loader.prefetch_factor,
        "amp_enabled": bool(trainer.amp),
        "total_batches": timing.batch_index,
        "actual_per_gpu_batch": trainer.train_loader.batch_size,
        "final_amp_scale": float(trainer.scaler.get_scale()),
        "measured_batches": timing.measured_batches,
        "step_seconds": timing.step_seconds,
        "data_wait_seconds": timing.wait_seconds,
        "measured_seconds": measured_seconds,
        "data_wait_ratio": timing.wait_seconds / measured_seconds if measured_seconds else 0.0,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(trainer.device),
    }
    report_path = run_dir / "reports" / f"rank-{rank:02d}.json"
    write_json(report_path, report)
    if report["actual_workers_per_rank"] != args.workers:
        raise RuntimeError("DataLoader capped the requested worker count")
    if report["actual_per_gpu_batch"] != args.per_gpu_batch:
        raise RuntimeError("trainer changed the requested per-GPU batch")
    if report["total_batches"] != args.steps or report["amp_enabled"] is not True:
        raise RuntimeError("benchmark retried an epoch or disabled AMP")
    if rank == 0:
        paths = [run_dir / "reports" / f"rank-{value:02d}.json" for value in range(world_size)]
        for path in paths:
            _wait_for(path)
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        aggregate = aggregate_reports(
            reports,
            per_gpu_batch=args.per_gpu_batch,
            warmup_steps=args.warmup_steps,
            world_size=world_size,
        )
        aggregate["code_commit"] = git_commit(args.repo_root)
        write_json(run_dir / "aggregate.json", aggregate)
        return aggregate
    return report


def _all(args: argparse.Namespace) -> dict[str, Any]:
    root = args.workspace / "benchmarks" / "wp8-training" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for per_gpu_batch, workers in candidate_grid(args.per_gpu_batches, args.worker_candidates):
        log_path = root / f"b{per_gpu_batch}-w{workers}.log"
        candidate_token = f"{args.run_id}-b{per_gpu_batch}-w{workers}-{uuid.uuid4().hex[:12]}"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.world_size}",
            str(Path(__file__).resolve()),
            "--repo-root",
            str(args.repo_root),
            "--workspace",
            str(args.workspace),
            "--data-root",
            str(args.data_root),
            "--train-cache",
            str(args.train_cache),
            "--run-id",
            args.run_id,
            "--world-size",
            str(args.world_size),
            "--seed",
            str(args.seed),
            "--sample-offset",
            str(args.sample_offset),
            "worker",
            "--per-gpu-batch",
            str(per_gpu_batch),
            "--workers",
            str(workers),
            "--candidate-token",
            candidate_token,
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--max-open-shards",
            str(args.max_open_shards),
            "--amp-init-scale",
            str(args.amp_init_scale),
            "--prefetch-factor",
            str(args.prefetch_factor),
            "--amp-growth-interval",
            str(args.amp_growth_interval),
        ]
        if args.world_size == 1:
            command = [sys.executable, *command[5:]]
        started = time.time()
        lifecycle = run_candidate(
            command,
            log_path=log_path,
            candidate_token=candidate_token,
            timeout_seconds=args.candidate_timeout_seconds,
            memory_headroom_bytes=int(args.memory_headroom_gib * 1024**3),
        )
        aggregate_path = root / f"b{per_gpu_batch}-w{workers}" / "aggregate.json"
        if lifecycle["returncode"] == 0 and aggregate_path.is_file():
            result = json.loads(aggregate_path.read_text(encoding="utf-8"))
        else:
            result = {
                "status": "failed",
                "per_gpu_batch": per_gpu_batch,
                "global_batch": per_gpu_batch * args.world_size,
                "workers_per_rank": workers,
                "log": log_path.name,
            }
        result.update(lifecycle)
        result["wall_seconds_including_startup"] = time.time() - started
        results.append(result)
        write_json(root / "progress.json", {"status": "running", "completed": results})
        if lifecycle["candidate_process_cleanup"]["remaining_pids"]:
            break
    passed = [item for item in results if item["status"] == "passed"]
    best = max(passed, key=lambda item: item["aggregate_images_per_second"]) if passed else None
    summary = {
        "status": "passed" if best else "failed",
        "code_commit": git_commit(args.repo_root),
        "run_id": args.run_id,
        "candidates": results,
        "best": best,
    }
    write_json(root / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--data-root", type=Path)
    result.add_argument("--train-cache", type=Path)
    result.add_argument("--run-id")
    result.add_argument("--world-size", type=int, default=WORLD_SIZE)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--sample-offset", type=int, default=0)
    subparsers = result.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--per-gpu-batch", type=int, required=True)
    worker.add_argument("--workers", type=int, required=True)
    worker.add_argument("--candidate-token")
    worker.add_argument("--steps", type=int, default=20)
    worker.add_argument("--warmup-steps", type=int, default=3)
    worker.add_argument("--max-open-shards", type=int, default=4)
    worker.add_argument("--amp-init-scale", type=float, default=16.0)
    worker.add_argument("--amp-growth-interval", type=int, default=1_000_000)
    worker.add_argument("--prefetch-factor", type=int, default=1)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--per-gpu-batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    all_parser.add_argument("--worker-candidates", type=int, nargs="+", default=list(DEFAULT_WORKERS))
    all_parser.add_argument("--steps", type=int, default=20)
    all_parser.add_argument("--warmup-steps", type=int, default=3)
    all_parser.add_argument("--max-open-shards", type=int, default=4)
    all_parser.add_argument("--amp-init-scale", type=float, default=16.0)
    all_parser.add_argument("--amp-growth-interval", type=int, default=1_000_000)
    all_parser.add_argument("--prefetch-factor", type=int, default=1)
    all_parser.add_argument("--candidate-timeout-seconds", type=float, default=600.0)
    all_parser.add_argument("--memory-headroom-gib", type=float, default=8.0)
    return result


def main() -> None:
    args = parser().parse_args()
    args.repo_root = args.repo_root.resolve()
    args.workspace = args.workspace.resolve()
    args.data_root = (args.data_root or args.workspace / "datasets/coco").resolve()
    args.train_cache = (
        args.train_cache or args.workspace / "feature_cache/coco2017-train2017-d1-cache-v1"
    ).resolve()
    if type(args.world_size) is not int or args.world_size <= 0:
        raise ValueError("world size must be a positive integer")
    if args.world_size > torch.cuda.device_count():
        raise ValueError(f"world size {args.world_size} exceeds {torch.cuda.device_count()} visible CUDA devices")
    if args.seed < 0 or args.sample_offset < 0:
        raise ValueError("seed and sample offset must be non-negative")
    args.run_id = args.run_id or f"{git_commit(args.repo_root)[:7]}-{int(time.time())}"
    if args.steps <= args.warmup_steps:
        raise ValueError("steps must be greater than warmup_steps")
    if args.command == "all" and (args.candidate_timeout_seconds <= 0 or args.memory_headroom_gib <= 0):
        raise ValueError("candidate timeout and memory headroom must be positive")
    requested_workers = args.workers if args.command == "worker" else max(args.worker_candidates)
    if args.steps < requested_workers:
        raise ValueError("steps must be at least the requested workers so DataLoader does not cap them")
    report = _worker(args) if args.command == "worker" else _all(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

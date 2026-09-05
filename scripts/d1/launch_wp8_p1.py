"""Run the approved D1 scratch gates, training and official evaluation in isolated subprocesses."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from scripts.d1 import run_wp8_p1_control as control
from scripts.d1.benchmark_wp8_training import Timing, cgroup_memory_snapshot, terminate_process_group
from ultralytics.data import converter
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import unwrap_model

ROOT = control.ROOT
STEPS = 40
WARMUP_STEPS = 20
TRAIN_BATCHES = math.ceil(118287 / 384)


def read_json(path):
    """Read a JSON evidence file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, text):
    """Publish a small status or input list atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def require_clean():
    """Require the frozen code version used by all stages."""
    identity = control.code_identity()
    if identity["dirty"]:
        raise RuntimeError("Commit the reviewed code before starting the pipeline")
    return identity


class ProbeTelemetry(control.TrainingTelemetry):
    """Measure a real training prefix and full validation without claiming formal epochs."""

    def __init__(self, output):
        super().__init__(output)
        self.timing = Timing(WARMUP_STEPS)
        self.validation_started = None
        self.validation_seconds = None
        self.before_parameter = None

    def on_train_start(self, trainer):
        super().on_train_start(trainer)
        self.before_parameter = next(unwrap_model(trainer.model).parameters()).detach().clone()
        self.timing.on_train_start(trainer)

    def on_train_batch_start(self, trainer):
        self.timing.on_train_batch_start(trainer)

    def on_train_batch_end(self, trainer):
        super().on_train_batch_end(trainer)
        self.timing.on_train_batch_end(trainer)

    def on_train_epoch_end(self, trainer):
        torch.cuda.synchronize(trainer.device)
        self.validation_started = time.perf_counter()

    def on_fit_epoch_end(self, trainer):
        super().on_fit_epoch_end(trainer)
        torch.cuda.synchronize(trainer.device)
        self.validation_seconds = time.perf_counter() - self.validation_started
        trainer.stop = True


def probe_worker(args):
    """Train exactly 40 real six-rank batches and run full val2017 once."""
    if int(os.environ.get("WORLD_SIZE", "0")) != 6:
        raise RuntimeError("Probe requires six torchrun ranks")
    rank = int(os.environ["RANK"])
    candidate = args.output_dir
    data_yaml = candidate / "inputs/coco2017.yaml"
    overrides = control.training_overrides(control.load_contract(), data_yaml, candidate / "run", args.workers)
    # The prefix has 40 batches, but formal warmup spans 3 * 309 batches.
    overrides["warmup_epochs"] = 3 * TRAIN_BATCHES / STEPS
    trainer = control.ScratchTrainer(overrides=overrides)
    telemetry = ProbeTelemetry(candidate / "telemetry")
    for event in ("on_train_start", "on_train_batch_start", "on_train_batch_end",
                  "on_train_epoch_end", "on_fit_epoch_end"):
        trainer.add_callback(event, getattr(telemetry, event))
    trainer.train()
    parameter = next(unwrap_model(trainer.model).parameters())
    delta = float((parameter.detach() - telemetry.before_parameter).abs().max())
    timing = telemetry.timing
    report = {
        "status": "passed", "identity": control.code_identity(), "rank": rank, "workers": args.workers,
        "actual_workers": trainer.train_loader.num_workers, "per_gpu_batch": trainer.train_loader.batch_size,
        "prefetch_factor": trainer.train_loader.prefetch_factor, "amp": bool(trainer.amp),
        "final_amp_scale": float(trainer.scaler.get_scale()), "steps": timing.batch_index,
        "successful_steps": int(trainer.optimizer_steps), "measured_steps": timing.measured_batches,
        "measured_seconds": timing.step_seconds + timing.wait_seconds,
        "data_wait_seconds": timing.wait_seconds, "validation_checkpoint_seconds": telemetry.validation_seconds,
        "val_seen": int(trainer.validator.seen), "val_dataset_samples": len(trainer.test_loader.dataset),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device), "parameter_max_delta": delta,
    }
    if not math.isfinite(delta) or delta <= 0:
        raise RuntimeError("Probe did not update the model")
    if (report["steps"], report["successful_steps"], report["per_gpu_batch"],
            report["actual_workers"], report["prefetch_factor"], report["amp"]) != (
            STEPS, STEPS, 64, args.workers, 1, True):
        raise RuntimeError("Probe changed batch/workers/AMP or skipped optimizer steps")
    control.write_json(candidate / f"rank-{rank}.json", report)
    return report


def aggregate_probe(reports, identity, workers):
    """Reject incomplete/inconsistent evidence; use the slowest rank for throughput."""
    if len(reports) != 6 or {row["rank"] for row in reports} != set(range(6)):
        raise ValueError("Probe must contain exactly six unique ranks")
    for row in reports:
        if row.get("identity") != identity or row.get("status") != "passed":
            raise ValueError("Probe identity or status differs")
        for key, expected in {"workers": workers, "actual_workers": workers, "per_gpu_batch": 64,
                              "prefetch_factor": 1, "amp": True, "steps": STEPS,
                              "successful_steps": STEPS, "measured_steps": STEPS - WARMUP_STEPS,
                              "val_dataset_samples": 5000}.items():
            if row.get(key) != expected:
                raise ValueError(f"Invalid probe field: {key}")
        for key in ("measured_seconds", "validation_checkpoint_seconds", "parameter_max_delta", "final_amp_scale"):
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"Invalid probe numeric field: {key}")
    if next(row for row in reports if row["rank"] == 0)["val_seen"] != 5000:
        raise ValueError("Probe did not validate all 5000 images")
    elapsed = max(row["measured_seconds"] for row in reports)
    step_seconds = elapsed / (STEPS - WARMUP_STEPS)
    validation = max(row["validation_checkpoint_seconds"] for row in reports)
    train_val_seconds = 30 * (TRAIN_BATCHES * step_seconds + validation)
    return {
        "status": "passed", "identity": identity, "workers": workers, "ranks": reports,
        "images_per_second": 384 / step_seconds, "step_seconds": step_seconds,
        "validation_checkpoint_seconds": validation, "train_val_seconds_30_epochs": train_val_seconds,
        "estimated_hours_range": [train_val_seconds * 1.15 / 3600, train_val_seconds * 1.35 / 3600],
        "eta_caveat": "40-batch prefix, 20 timed batches, hot local data; final official evaluation is additional",
    }


class OfficialValidator(control.ScratchValidator):
    """Emit standard COCO class IDs and propagate evaluator failures."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.is_coco = True
        self.is_lvis = False
        self.class_map = converter.coco80_to_coco91_class()
        self.args.save_json = True

    def eval_json(self, stats):
        return stats


def official_metrics(annotation, predictions, image_ids):
    """Run the same official COCO-compatible evaluator used by the P0 diagnosis."""
    from faster_coco_eval import COCO, COCOeval_faster

    ground_truth = COCO(str(annotation))
    if read_json(predictions):
        detections = ground_truth.loadRes(str(predictions))
    else:
        detections = COCO()
        detections.dataset = {
            "images": ground_truth.dataset["images"], "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval_faster(ground_truth, detections, iouType="bbox", lvis_style=False)
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ("AP_all", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large")
    result = {key: float(evaluator.stats_as_dict[key]) for key in keys}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Official evaluator returned non-finite metrics")
    return result


def evaluate(args):
    """Strictly reload a checkpoint and independently evaluate all 5000 validation images."""
    metadata = control.strict_checkpoint(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("ema")
    if saved is None:
        saved = checkpoint["model"]
    model = DetectionModel(YAML.load(ROOT / control.MODEL), nc=80, ch=3, verbose=False)
    model.load_state_dict(saved.float().state_dict(), strict=True)
    del checkpoint, saved
    model.eval()
    options = {
        "task": "detect", "data": str(args.data_yaml), "imgsz": 640, "batch": 64, "workers": 4,
        "device": "0", "rect": False, "half": False, "quantize": None, "save_json": True,
        "plots": False, "verbose": False, "conf": 0.001, "iou": 0.7, "max_det": 300,
    }
    validator = OfficialValidator(save_dir=args.output_dir, args=options)
    started = time.monotonic()
    internal = validator(model=model)
    if validator.seen != 5000:
        raise RuntimeError(f"Expected 5000 validation images, got {validator.seen}")
    expected, _ = control.read_splits(ROOT / "experiments/d1/manifests")
    expected_ids = sorted(int(Path(path).stem) for path in expected["val2017"])
    seen_ids = sorted(int(Path(path).stem) for path in validator.dataloader.dataset.im_files)
    if seen_ids != expected_ids:
        raise RuntimeError("Validation IDs differ from WP0")
    prediction_path = args.output_dir / "predictions.json"
    atomic_text(prediction_path, json.dumps(validator.jdict, separators=(",", ":"), allow_nan=False) + "\n")
    metrics = official_metrics(args.data_root / "annotations/instances_val2017.json", prediction_path, seen_ids)
    internal = {key: float(value) for key, value in internal.items()}
    if not all(math.isfinite(value) for value in internal.values()):
        raise FloatingPointError("Non-finite internal validation metric")
    report = {"status": "passed", "identity": control.code_identity(), "checkpoint": metadata,
              "seen": validator.seen, "internal": internal, "official": metrics,
              "elapsed_seconds": time.monotonic() - started,
              "prediction_sha256": control.sha256_file(prediction_path)}
    control.write_json(args.output_dir / "report.json", report)
    return report


def make_probe_inputs(data_root, candidate):
    """Generate an isolated prefix view; never modify the dataset split itself."""
    paths, _ = control.read_splits(ROOT / "experiments/d1/manifests")
    inputs = candidate / "inputs"
    inputs.mkdir(parents=True)
    for name, split, limit in (("train", "train2017", STEPS * 384), ("val", "val2017", 5000)):
        atomic_text(inputs / f"{name}.txt", "".join(str(data_root / path) + "\n" for path in paths[split][:limit]))
    YAML.save(inputs / "coco2017.yaml", {
        "path": str(data_root), "train": str(inputs / "train.txt"), "val": str(inputs / "val.txt"),
        "names": YAML.load(ROOT / "ultralytics/cfg/datasets/coco.yaml")["names"],
    })


def keeper_command(script, command):
    """Use the existing keeper's own stop/start API and original runtime arguments."""
    return [str(script.parent / ".venv/bin/python3"), str(script), command,
            "--target", "10", "--cycle", "0.1", "--control-every", "1",
            "--pidfile", str(script.parent / "gpu_keeper_v2.pid"),
            "--log", str(script.parent / "gpu_keeper_v2.log"),
            "--status-file", str(script.parent / "gpu_keeper_v2_status.json")]


class Pipeline:
    """Own child process groups and write terminal status on success or failure."""

    def __init__(self, args):
        self.args = args
        self.identity = require_clean()
        self.output = args.output_dir
        self.output.mkdir(parents=True, exist_ok=False)
        self.logs = args.workspace / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.child = None
        self.monitor = None
        self.restored_keeper = False
        self.state = {"run_id": args.run_id, "identity": self.identity, "pid": os.getpid(),
                      "approved": args.approved, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def phase(self, name, **fields):
        self.state.update(fields)
        self.state.update(phase=name, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        control.write_json(self.output / "status.json", self.state)
        atomic_text(self.logs / f"{self.args.run_id}.status", name + "\n")
        print(json.dumps(self.state, sort_keys=True), flush=True)

    def run_child(self, name, command, timeout=1800):
        if require_clean() != self.identity:
            raise RuntimeError("Code changed between stages")
        log = self.logs / f"{self.args.run_id}-{name}.log"
        memory = cgroup_memory_snapshot()
        started = time.monotonic()
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   PYTHONUNBUFFERED="1", MPLBACKEND="Agg", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        with log.open("x", encoding="utf-8") as stream:
            self.child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            atomic_text(self.logs / f"{self.args.run_id}-{name}.pid", str(self.child.pid) + "\n")
            self.phase(name.upper(), child_pid=self.child.pid, child_log=str(log))
            try:
                while self.child.poll() is None:
                    now = cgroup_memory_snapshot()
                    if now["oom_kill_count"] > memory["oom_kill_count"]:
                        raise RuntimeError("Cgroup OOM occurred; stopping the pipeline")
                    if now["rss_bytes"] > now["limit_bytes"] - 8 * 1024**3:
                        raise RuntimeError("Anonymous memory leaves insufficient safety headroom")
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(name)
                    time.sleep(5)
                if self.child.returncode:
                    raise RuntimeError(f"{name} exited with {self.child.returncode}; see {log}")
            finally:
                terminate_process_group(self.child.pid, grace_seconds=5)
                self.child.wait()
                self.child = None
        return time.monotonic() - started

    def base(self, command):
        return [sys.executable, "-m", "scripts.d1.launch_wp8_p1", command,
                "--workspace", str(self.args.workspace), "--data-root", str(self.args.data_root),
                "--run-id", self.args.run_id]

    def evaluate_command(self, checkpoint, output, data_yaml):
        return [*self.base("evaluate"), "--checkpoint", str(checkpoint), "--output-dir", str(output),
                "--data-yaml", str(data_yaml)]

    def execute(self):
        if not self.args.approved:
            raise RuntimeError("The pipeline requires explicit approval")
        if importlib.util.find_spec("faster_coco_eval") is None:
            raise RuntimeError("faster-coco-eval is required before launch")
        names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
        if len(names) != 6 or any("A40" not in name for name in names):
            raise RuntimeError("The contract requires six A40 GPUs")
        atomic_text(self.logs / f"{self.args.run_id}.pid", str(os.getpid()) + "\n")
        self.phase("PREPARING")
        args = self.args
        args.run_root = args.workspace / "runs" / args.run_id
        args.model_only = False
        args.workers = 4
        control.prepare(args)
        if args.keeper_script:
            pidfile = args.keeper_script.parent / "gpu_keeper_v2.pid"
            if pidfile.is_file():
                pid = int(pidfile.read_text())
                tokens = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if b"gpu_keeper.py" not in [Path(token.decode()).name.encode() for token in tokens if token]:
                    raise RuntimeError("Keeper pidfile points to another process")
                subprocess.run(keeper_command(args.keeper_script, "stop"), check=True, timeout=20)
                self.restored_keeper = True
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if active.strip():
            raise RuntimeError(f"GPU compute processes are still present: {active.strip()}")
        gpu_log = (self.logs / f"{args.run_id}-gpu.csv").open("x")
        self.monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
             "--format=csv", "-l", "10"], stdout=gpu_log, stderr=subprocess.STDOUT, start_new_session=True)
        gpu_log.close()
        candidates = []
        for workers in (4, 8):
            candidate = self.output / f"probe-w{workers}"
            make_probe_inputs(args.data_root, candidate)
            worker_command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=6",
                              "-m", "scripts.d1.launch_wp8_p1", "worker", "--workspace", str(args.workspace),
                              "--data-root", str(args.data_root), "--run-id", args.run_id,
                              "--output-dir", str(candidate), "--workers", str(workers)]
            self.run_child(f"probe-w{workers}", worker_command)
            reports = [read_json(candidate / f"rank-{rank}.json") for rank in range(6)]
            aggregate = aggregate_probe(reports, self.identity, workers)
            reloads = read_json(candidate / "run/checkpoint-reload.json")
            if any(reloads[name].get("strict_reload") is not True for name in ("best.pt", "last.pt")):
                raise RuntimeError("Probe checkpoint reload failed")
            aggregate["checkpoint_reload"] = reloads
            control.write_json(candidate / "aggregate.json", aggregate)
            candidates.append(aggregate)
        best = max(candidates, key=lambda item: item["images_per_second"])
        control.write_json(self.output / "benchmark.json", {"status": "passed", "candidates": candidates, "best": best})
        evaluation = self.output / "probe-official-evaluation"
        best_probe = self.output / f"probe-w{best['workers']}"
        self.run_child("probe-evaluate", self.evaluate_command(
            best_probe / "run/weights/last.pt", evaluation, self.output / "inputs/coco2017.yaml"))
        evaluated = read_json(evaluation / "report.json")
        if evaluated["status"] != "passed" or evaluated["seen"] != 5000 or evaluated["identity"] != self.identity:
            raise RuntimeError("Official evaluation gate failed")
        args.workers = best["workers"]
        control.prepare(args)
        gate = {
            "status": "passed", "identity": self.identity, "workers": best["workers"],
            "preparation_sha256": control.sha256_file(self.output / "preparation.json"),
            "six_gpu_worker_benchmark": True, "real_batch_backward": True,
            "strict_checkpoint_reload": True, "evaluation_ready": True,
            "benchmark_sha256": control.sha256_file(self.output / "benchmark.json"),
            "evaluation_sha256": control.sha256_file(evaluation / "report.json"),
        }
        control.write_json(self.output / "runtime-gate.json", gate)
        self.phase("READY_TO_TRAIN", selected_workers=best["workers"],
                   estimated_hours_range=best["estimated_hours_range"],
                   official_evaluation_seconds=evaluated["elapsed_seconds"])
        train_command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=6",
            "-m", "scripts.d1.run_wp8_p1_control", "train", "--output-dir", str(self.output),
            "--run-root", str(args.run_root), "--runtime-gate", str(self.output / "runtime-gate.json"), "--approved",
        ]
        wall = self.run_child("train", train_command, timeout=7 * 24 * 3600)
        for name in ("best", "last"):
            self.run_child(f"evaluate-{name}", self.evaluate_command(
                args.run_root / f"weights/{name}.pt", self.output / f"official-{name}",
                self.output / "inputs/coco2017.yaml"), timeout=3600)
        with (args.run_root / "results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 30 or any(not math.isfinite(float(value)) for row in rows for value in row.values()):
            raise RuntimeError("Final results do not contain 30 finite epochs")
        best_eval = read_json(self.output / "official-best/report.json")
        last_eval = read_json(self.output / "official-last/report.json")
        scratch_ap = best_eval["official"]["AP_all"]
        summary = {
            "status": "passed", "scope": "P1-COCO-30 preliminary single-seed as-run comparison",
            "identity": self.identity, "epochs": 30, "workers": best["workers"],
            "training_wall_seconds": wall, "training_gpu_hours": wall * 6 / 3600,
            "best": best_eval, "epoch30": last_eval,
            "p0_reference_official_ap": 0.11615628398461215,
            "ap_retention_percent": 100 * 0.11615628398461215 / scratch_ap if scratch_ap > 0 else None,
            "p0_reference_has_aux_broadcast_deviation": True,
        }
        control.write_json(self.output / "summary.json", summary)
        self.phase("COMPLETED", training_wall_seconds=wall, summary=str(self.output / "summary.json"))

    def close(self):
        if self.child is not None:
            terminate_process_group(self.child.pid, grace_seconds=5)
            self.child.wait()
        if self.monitor is not None:
            self.monitor.terminate()
            self.monitor.wait(timeout=10)
        if self.restored_keeper:
            subprocess.run(keeper_command(self.args.keeper_script, "start"), check=True, timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "worker", "evaluate"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--copy-receipt", type=Path)
    parser.add_argument("--workers", type=int, choices=(4, 8), default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--keeper-script", type=Path)
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("Invalid run ID")
    args.workspace = args.workspace.resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir = (args.output_dir or args.workspace / "manifests" / args.run_id).resolve()
    if ROOT == args.output_dir or ROOT in args.output_dir.parents:
        raise ValueError("Keep runtime output outside the repository")
    if args.command == "worker":
        result = probe_worker(args)
    elif args.command == "evaluate":
        result = evaluate(args)
    else:
        pipeline = Pipeline(args)
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            pipeline.execute()
            result = {"status": "completed"}
        except BaseException as exc:
            pipeline.phase("STOPPED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                           error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            pipeline.close()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

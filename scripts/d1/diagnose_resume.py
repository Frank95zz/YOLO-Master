"""Locate D1 resume drift with bounded six-rank runs and read-only numerical probes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import torch

from scripts.d1 import check_ema_runtime as acceptance
from scripts.d1 import run_p5_ablation as p5
from scripts.d1.p1p2_runtime import restore_rng, rng_state
from scripts.d1.run_wp8_p1_control import write_json
from ultralytics.nn.foundation.cache import sha256_file
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("input", "before", "forward", "local_gradients", "reduced_gradients", "after")


def cpu_tree(value):
    """Make independent, weights-only-loadable snapshots without retaining the training graph."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported snapshot type: {type(value)}")


def fingerprint(value):
    """Record exact tensor content without duplicating large cached feature batches on disk."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {k: fingerprint(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported fingerprint type: {type(value)}")


def compare(left, right):
    """Report exact differences and floating magnitudes; never turn a tolerance into an exact pass."""
    rows = []

    def visit(a, b, key):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                rows.append({"key": key, "reason": "tensor metadata"})
            elif not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                rows.append({"key": key, "reason": "non-finite tensor"})
            elif not torch.equal(a, b):
                delta = a.double() - b.double()
                rows.append(
                    {
                        "key": key,
                        "reason": "tensor values",
                        "max_abs": float(delta.abs().max()),
                        "different_elements": int((a != b).sum()),
                        "elements": a.numel(),
                        "delta_squared": float(delta.square().sum()),
                        "reference_squared": float(a.double().square().sum()),
                    }
                )
        elif type(a) is not type(b):
            rows.append({"key": key, "reason": "type"})
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                rows.append({"key": key, "reason": "keys"})
            for name in a.keys() & b.keys():
                visit(a[name], b[name], f"{key}.{name}")
        elif isinstance(a, (tuple, list)):
            if len(a) != len(b):
                rows.append({"key": key, "reason": "length"})
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{key}.{i}")
        elif a != b:
            rows.append({"key": key, "reason": "value", "left": a, "right": b})

    visit(left, right, "root")
    rows.sort(key=lambda r: (-r.get("max_abs", 0), r["key"]))
    return {"exact": not rows, "different_entries": len(rows), "differences": rows}


class DiagnosticTrainer(p5.P5FrozenTrainer):
    """Observe the real P5 loop; hooks clone values and never replace outputs or gradients."""

    prime_ddp = False

    def _setup_train(self):
        super()._setup_train()
        self.probe = None
        self.probe_local = {}
        self.probe_handles = []
        self.add_callback("on_train_epoch_start", self._attach_probes)
        self.add_callback("on_train_epoch_end", self._detach_probes)

    def _attach_probes(self, trainer):
        if self.probe_handles:
            raise RuntimeError("Duplicate diagnostic hooks")
        model = unwrap_model(self.model)
        self.probe_handles.append(model.register_forward_pre_hook(self._probe_before))
        self.probe_handles.append(model.register_forward_hook(self._probe_forward))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.probe_handles.append(parameter.register_hook(self._gradient_hook(name)))

    def _detach_probes(self, trainer):
        for handle in self.probe_handles:
            handle.remove()
        self.probe_handles.clear()

    def _prime_ddp(self, batch):
        model = unwrap_model(self.model)
        before = self._state()
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        random_state = rng_state(self.device)
        criterion = model.criterion
        model.criterion = deepcopy(criterion, {id(model): model})
        try:
            for _ in range(min(self.optimizer_steps, 2)):
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, _ = self.model(batch)
                self.scaler.scale(loss.sum() * self.world_size).backward()
                self.optimizer.zero_grad(set_to_none=True)
        finally:
            model.criterion = criterion
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
            restore_rng(random_state, self.device)
            self.optimizer.zero_grad(set_to_none=True)
        report = compare(before, self._state())
        if not report["exact"]:
            raise RuntimeError(f"DDP priming changed train state: {report}")
        write_json(self.e1_output / f"priming-rank-{self.e1_rank}.json", {"state_unchanged": True})

    def _gradient_hook(self, name):
        def capture(gradient):
            if self.probe is not None:
                if name in self.probe_local:
                    raise RuntimeError(f"Repeated local gradient hook: {name}")
                self.probe_local[name] = gradient.detach().clone()

        return capture

    def _state(self):
        model = unwrap_model(self.model)
        criterion = getattr(model, "criterion", None)
        native = getattr(criterion, "native_criterion", criterion)
        return cpu_tree(
            {
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.ema.state_dict(),
                "ema_updates": self.ema.updates,
                "optimizer_steps": self.optimizer_steps,
                "criterion": {k: getattr(native, k, None) for k in ("updates", "o2m", "o2o")},
                "training": {k: m.training for k, m in model.named_modules()},
            }
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.prime_ddp and self.resume and self.epoch == self.start_epoch and self.e1_epoch_batches == 0:
            self._prime_ddp(batch)
        if self.epoch == 1:
            self.probe_local = {}
            self.probe = {"input": fingerprint(batch), "batch": self.e1_epoch_batches}
        return batch

    def _probe_before(self, module, inputs):
        if self.probe is not None:
            self.probe["before"] = self._state()
            self.probe["ddp"] = cpu_tree(self.model._get_ddp_logging_data())

    def _probe_forward(self, module, inputs, output):
        if self.probe is not None:
            self.probe["forward"] = cpu_tree(output)

    def optimizer_step(self):
        if self.probe is not None:
            self.probe["local_gradients"] = cpu_tree(self.probe_local)
            self.probe["reduced_gradients"] = cpu_tree(
                {k: p.grad for k, p in unwrap_model(self.model).named_parameters() if p.grad is not None}
            )
        result = super().optimizer_step()
        if self.probe is not None:
            if self.e1_retries:
                raise RuntimeError("AMP retries invalidate the first-difference probe")
            self.probe["after"] = self._state()
            destination = self.e1_output / "probes"
            destination.mkdir(exist_ok=True)
            torch.save(self.probe, destination / f"rank-{self.e1_rank}-batch-{self.probe['batch']}.pt")
            self.probe = None
            self.probe_local.clear()
        return result


def summarize(output, variant="BASE"):
    """Compare every rank and both the full and short-tail batch in epoch two."""
    reports = output / "reports"
    paths = [reports / f"{variant}-foreach-v1-{mode}" / "probes" for mode in ("continuous", "resumed")]
    comparisons = []
    for rank in range(6):
        for batch in range(2):
            name = f"rank-{rank}-batch-{batch}.pt"
            states = [torch.load(path / name, map_location="cpu", weights_only=True) for path in paths]
            stages = {key: compare(states[0][key], states[1][key]) for key in STAGES}
            comparisons.append(
                {
                    "rank": rank,
                    "batch": batch,
                    "stages": stages,
                    "first_difference": next((key for key in STAGES if not stages[key]["exact"]), None),
                    "ddp": {mode: state["ddp"] for mode, state in zip(("continuous", "resumed"), states)},
                }
            )
    result = {
        "schema": "d1-resume-diagnostics-v1",
        "variant": variant,
        "comparisons": comparisons,
        "source_sha256": sha256_file(Path(__file__)),
        "formal_training": False,
    }
    write_json(output / "diagnostics.json", result)
    return result


def child(args, phase):
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=6",
        "--module",
        "scripts.d1.diagnose_resume",
        "worker",
        "--output",
        str(args.output),
        "--variant",
        args.variant,
        "--phase",
        phase,
    ]
    if args.prime_ddp:
        command.append("--prime-ddp")
    if args.rank_buffers:
        command.append("--rank-buffers")
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH=str(ROOT))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    with (args.output / f"{phase}.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        write_json(args.output / "status.json", {"status": "running", "phase": phase, "pid": process.pid})
        print(f"START {phase} pid={process.pid}", flush=True)
        try:
            code = process.wait(timeout=300)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise RuntimeError(f"Diagnostic {phase} failed: exit={code}")
        print(f"PASS {phase}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-workspace", type=Path)
    parser.add_argument("--variant", choices=tuple(p5.VARIANTS), default="BASE")
    parser.add_argument("--phase", choices=("continuous", "stop", "resume"))
    parser.add_argument("--prime-ddp", action="store_true", help="Test no-update reducer priming only on resume")
    parser.add_argument("--rank-buffers", action="store_true", help="Register per-rank buffer checkpoint restoration")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.implementation = "foreach-v1"
    if args.command == "run":
        if args.source_workspace is None:
            parser.error("run requires --source-workspace")
        acceptance.prepare(args.source_workspace.resolve(), args.output)
        write_json(
            args.output / "probe-source.json",
            {"sha256": sha256_file(Path(__file__)), "prime_ddp": args.prime_ddp, "rank_buffers": args.rank_buffers},
        )
        try:
            for phase in ("continuous", "stop", "resume"):
                child(args, phase)
            summarize(args.output, args.variant)
        except BaseException as error:
            write_json(args.output / "status.json", {"status": "failed", "error": str(error)})
            raise
        write_json(args.output / "status.json", {"status": "completed", "formal_training": False})
    elif args.command == "worker":
        receipt = json.loads((args.output / "probe-source.json").read_text())
        if receipt != {
            "sha256": sha256_file(Path(__file__)),
            "prime_ddp": args.prime_ddp,
            "rank_buffers": args.rank_buffers,
        }:
            raise ValueError("Diagnostic code changed")
        if args.phase is None:
            parser.error("worker requires --phase")
        DiagnosticTrainer.prime_ddp = args.prime_ddp
        original_spec_for = p5.spec_for

        def registered_spec(matrix, variant):
            return original_spec_for({**matrix, "resume_rank_buffers": args.rank_buffers}, variant)

        with patch.object(p5, "P5FrozenTrainer", DiagnosticTrainer), patch.object(p5, "spec_for", registered_spec):
            acceptance.worker(args)
    else:
        summarize(args.output, args.variant)


if __name__ == "__main__":
    main()

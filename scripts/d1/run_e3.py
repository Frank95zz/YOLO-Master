"""Three-seed BN64 VisDrone aux sweep; official MATLAB remains a separate gate."""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

from scripts.d1 import run_e2 as e2
from scripts.d1.e3_mechanism import probe
from scripts.d1.evaluate_visdrone import validate_official_report
from scripts.d1.p1p2_runtime import atomic_torch, clean_child_env, restore_rng, rng_state
from scripts.d1.prepare_visdrone import encoded, file_sha, immutable
from scripts.d1.run_p5_ablation import P5FrozenTrainer
from scripts.d1.run_p5_ablation import model_config as p5_model_config
from scripts.d1.run_wp8_p1_control import LOCKED_TRAIN, write_json
from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/experiments/d1/p1p2/e3-visdrone.yaml"
EXPECTED = {
    "schema_version": "d1-e3-bn64-aux-v1",
    "dataset": "VisDrone2019-DET",
    "architecture": "BN64",
    "nc": 10,
    "imgsz": 640,
    "schedule_epochs": 300,
    "screen_epochs": 60,
    "seeds": [0, 1, 2],
    "world_size": 6,
    "global_batch": 96,
    "workers": 4,
    "prefetch_factor": 1,
    "p3_upsample_mode": "separable_bilinear2x",
    "ema_implementation": "foreach-v1",
    "resume_temperature_policy": "epoch-boundary-v1",
    "resume_rank_buffers": True,
    "balance_candidates": [0.0, 0.01, 0.1],
    "z_candidates": [0.0, 0.001, 0.01],
    "stage1_gain": 0.1,
    "stage2_gains": [0.0, 0.03, 0.1, 0.3],
    "mixture_aux_budget": 3.0,
    "evaluation_period": 5,
    "official_backend": "official-matlab",
    "selection_epoch": 60,
    "max_det": 500,
    "conf": 0.001,
    "minimum_free_gib": 80,
    "maximum_cgroup_fraction": 0.75,
}
POLICIES = ("ema_implementation", "resume_temperature_policy", "resume_rank_buffers")
PARAMETERS = 1340259


def load_contract(path=CONFIG):
    value = YAML.load(path)
    if encoded(value) != encoded(EXPECTED):
        raise ValueError("Unregistered E3 contract")
    return value


def identity():
    result = e2.source_identity()
    result["contract_sha256"] = file_sha(CONFIG)
    return result


def candidates():
    """Default and zero first; no candidate or seed is pruned."""
    pairs = list(itertools.product(EXPECTED["balance_candidates"], EXPECTED["z_candidates"]))
    pairs = [(0.01, 0.001), (0.0, 0.0)] + [p for p in pairs if p not in ((0.01, 0.001), (0.0, 0.0))]
    return [{"balance": b, "z": z, "gain": 0.1, "seed": seed} for b, z in pairs for seed in EXPECTED["seeds"]]


def key(candidate):
    for field, values in (
        ("balance", EXPECTED["balance_candidates"]),
        ("z", EXPECTED["z_candidates"]),
        ("gain", EXPECTED["stage2_gains"]),
        ("seed", EXPECTED["seeds"]),
    ):
        if type(candidate[field]) not in (int, float) or candidate[field] not in values:
            raise ValueError(f"Unregistered candidate {field}")
    if type(candidate["seed"]) is not int:
        raise ValueError("Seed must be an integer")
    return f"b{candidate['balance']:g}-z{candidate['z']:g}-g{candidate['gain']:g}-s{candidate['seed']}"


def model_config(candidate):
    key(candidate)
    config = p5_model_config("BN64", "separable_bilinear2x")
    config["detect"]["nc"] = 10
    config["latent_mixture"].update(balance_loss_coeff=candidate["balance"], router_z_loss_coeff=candidate["z"])
    config["loss"].update(latent_aux_gain=candidate["gain"], mixture_aux_budget=3.0)
    return config


def construct_model(candidate):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(candidate["seed"])
        model = D1FoundationDetectionModel(model_config(candidate), verbose=False)
        initialize_mixture_loss_ema_buffer(model)
    model.detect.max_det = 500
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError("BN64 10-class parameter count changed")
    return model


def prepare(workspace, data, cache):
    load_contract()
    current = identity()
    workspace = workspace.resolve()
    if workspace.is_relative_to(ROOT) or (workspace.exists() and any(workspace.iterdir())):
        raise ValueError("Use a new, empty external workspace")
    reference = e2.prepare(workspace / "data-check", data, cache)
    inputs = workspace / "inputs"
    inputs.mkdir()
    files = {}
    for seed in EXPECTED["seeds"]:
        c = {"balance": 0.01, "z": 0.001, "gain": 0.1, "seed": seed}
        model = construct_model(c)
        tensors = {k: v.detach().clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        path = inputs / f"initial-s{seed}.pt"
        atomic_torch(path, tensors)
        files[path.name] = file_sha(path)
    for b, z, gain in itertools.product(
        EXPECTED["balance_candidates"], EXPECTED["z_candidates"], EXPECTED["stage2_gains"]
    ):
        c = {"balance": b, "z": z, "gain": gain, "seed": 0}
        name = key(c).rsplit("-s", 1)[0] + ".yaml"
        YAML.save(inputs / name, model_config(c))
        files[name] = file_sha(inputs / name)
    result = {
        "identity": current,
        "contract": load_contract(),
        "workspace": str(workspace),
        "data": reference["data"],
        "cache": reference["cache"],
        "parameters": PARAMETERS,
        "reference_sha256": file_sha(workspace / "data-check/matrix.json"),
        "inputs_sha256": files,
        "candidates": candidates(),
        "official_evaluation_deferred": True,
    }
    immutable(workspace / "matrix.json", encoded(result))
    return result


def matrix(workspace):
    workspace = workspace.resolve()
    mat = json.loads((workspace / "matrix.json").read_text())
    if mat["identity"] != identity() or mat["contract"] != load_contract() or mat["candidates"] != candidates():
        raise ValueError("E3 source, contract or candidate matrix changed")
    if mat["workspace"] != str(workspace) or mat["reference_sha256"] != file_sha(workspace / "data-check/matrix.json"):
        raise ValueError("E3 workspace or provenance changed")
    reference = e2.matrix(workspace / "data-check")
    if any(mat[k] != reference[k] for k in ("data", "cache")):
        raise ValueError("E3 data/cache changed")
    for name, digest in mat["inputs_sha256"].items():
        if file_sha(workspace / "inputs" / name) != digest:
            raise ValueError(f"E3 prepared input changed: {name}")
    return mat


def spec_for(mat, candidate, profile="screen"):
    if profile not in ("screen", "smoke", "continuous", "benchmark"):
        raise ValueError("Unknown bounded E3 profile")
    cid = key(candidate)
    workspace = Path(mat["workspace"])
    run_id = f"E3-{profile}-{cid}"
    initial = f"initial-s{candidate['seed']}.pt"
    model = cid.rsplit("-s", 1)[0] + ".yaml"
    policies = {p: EXPECTED[p] for p in POLICIES}
    return {
        "identity": {
            "source": mat["identity"],
            "run_id": run_id,
            "candidate": candidate,
            "reference_sha256": mat["reference_sha256"],
            "initial_sha256": mat["inputs_sha256"][initial],
            "model_sha256": mat["inputs_sha256"][model],
            **policies,
        },
        "workspace": str(workspace),
        "run_id": run_id,
        "profile": "E1" if profile == "screen" else "benchmark",
        "window": {"screen": 60, "smoke": 2, "continuous": 2, "benchmark": 3}[profile],
        "seed": candidate["seed"],
        "parameters": PARAMETERS,
        "variant": "BN64",
        "candidate": candidate,
        "output": str(workspace / "reports" / run_id),
        "initial_state": str(workspace / "inputs" / initial),
        "model_file": str(workspace / "inputs" / model),
        "data_profile": "smoke" if profile in ("smoke", "continuous") else "benchmark",
        **policies,
    }


def overrides(mat, spec):
    return {
        **deepcopy(LOCKED_TRAIN),
        "epochs": 300,
        "batch": 96,
        "nbs": 96,
        "workers": 4,
        "seed": spec["seed"],
        "device": "0,1,2,3,4,5",
        "patience": 300,
        "save_period": 5,
        "conf": 0.001,
        "max_det": 500,
        "latent_aux_gain": spec["candidate"]["gain"],
        "mixture_aux_budget": 3.0,
        "exist_ok": True,
        "verbose": False,
        "data": str(Path(mat["workspace"]) / "data-check/inputs" / f"{spec['data_profile']}.yaml"),
        "model": spec["model_file"],
        "project": str(Path(mat["workspace"]) / "runs"),
        "name": spec["run_id"],
    }


class E3Trainer(P5FrozenTrainer):
    """Use the tested fast trainer with E3 diagnostics and VisDrone export."""

    required_schedule_epochs = 300
    required_global_batch = 96
    required_workers = 4

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.detect.max_det = 500
        return model

    def e1_on_train_start(self, trainer):
        super().e1_on_train_start(trainer)
        model = unwrap_model(self.model)
        c = self.e1["candidate"]
        if model.config_dict() != model_config(c) or model.detect.nc != 10 or model.detect.max_det != 500:
            raise ValueError("Resolved E3 architecture or aux settings changed")
        if self.args.seed != self.e1["seed"] or self.args.latent_aux_gain != c["gain"]:
            raise ValueError("Resolved E3 seed/gain changed")

    def e1_on_train_epoch_start(self, trainer):
        started = time.monotonic()
        saved = rng_state(self.device)
        try:
            dataset = self.train_loader.dataset
            selected = [(2 * self.e1_rank + i) % len(dataset) for i in range(2)]
            batch = dataset.collate_fn([dataset[i] for i in selected])
            batch = super().preprocess_batch(batch)
            report = probe(unwrap_model(self.model), batch)
            write_json(
                self.e1_output / "mechanism" / f"rank-{self.e1_rank}-epoch-{self.epoch + 1:03d}.json",
                {
                    "epoch": self.epoch + 1,
                    "rank": self.e1_rank,
                    "candidate": self.e1["candidate"],
                    "indices": selected,
                    "seconds": time.monotonic() - started,
                    **report,
                },
            )
        finally:
            restore_rng(saved, self.device)
            self.e1_retry_batch = self.e1_retry_buffers = self.e1_retry_rng = None
        self.e1_epoch_started = time.monotonic()
        self.e1_epoch_batches = 0
        self.e1_previous_end = None
        self.e1_wait = self.e1_compute = 0.0
        self.e1_start_updates = self.optimizer_steps
        torch.cuda.reset_peak_memory_stats(self.device)

    def e1_standard_evaluation(self, epoch):
        output = self.e1_output / "official" / f"epoch-{epoch:03d}"
        output.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.d1.run_e3",
            "evaluate",
            "--workspace",
            self.e1["workspace"],
            "--candidate",
            key(self.e1["candidate"]),
            "--epoch",
            str(epoch),
            "--profile",
            "screen",
        ]
        started = time.monotonic()
        with (output / "evaluation.log").open("w") as log:
            subprocess.run(
                command, cwd=ROOT, env=clean_child_env(), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200
            )
        write_json(
            output / "cost.json",
            {
                "seconds": time.monotonic() - started,
                "reserved_gpus": self.world_size,
                "scope": "GPU export while other ranks wait; included in enclosing training attempt",
            },
        )


def resolve_candidate(cid):
    for b, z, gain, seed in itertools.product(
        EXPECTED["balance_candidates"], EXPECTED["z_candidates"], EXPECTED["stage2_gains"], EXPECTED["seeds"]
    ):
        c = {"balance": b, "z": z, "gain": gain, "seed": seed}
        if key(c) == cid:
            return c
    raise ValueError("Unknown E3 candidate")


def train(args):
    if not args.approved or int(os.environ.get("WORLD_SIZE", 0)) != 6:
        raise ValueError("E3 needs explicit approval and six-rank torchrun")
    mat = matrix(args.workspace)
    spec = spec_for(mat, resolve_candidate(args.candidate), args.profile)
    if args.window is not None:
        if args.profile != "smoke" or args.window not in (1, 2):
            raise ValueError("Only isolated smoke may stop at epoch one")
        spec["window"] = args.window
    output = Path(spec["output"])
    last = args.workspace / "runs" / spec["run_id"] / "weights/last.pt"
    common = overrides(mat, spec)
    if args.resume:
        saved = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != spec["identity"] or saved["epoch"] + 1 >= spec["window"] or not last.is_file():
            raise ValueError("Invalid or already complete E3 resume")
        common["resume"] = str(last)
    elif last.exists() or (output / "resume.pt").exists() or (output / "training-result.json").exists():
        raise FileExistsError("Never overwrite an E3 run; request validated resume")
    cache = Path(mat["cache"])
    trainer = E3Trainer(
        overrides=common,
        run_spec=spec,
        feature_caches={s: cache / f"visdrone-{s}" for s in ("train", "val")},
        trusted_feature_cache=True,
        feature_prefetch_factor=1,
        max_open_feature_shards=4,
        amp_init_scale=16,
        amp_growth_interval=1000000,
    )
    trainer.train()


def evaluate(args):
    mat = matrix(args.workspace)
    c = resolve_candidate(args.candidate)
    spec = spec_for(mat, c, args.profile)
    if not 1 <= args.epoch <= spec["window"]:
        raise ValueError("Evaluation epoch outside registered window")
    checkpoint = args.workspace / "runs" / spec["run_id"] / "weights/last.pt"
    out = Path(spec["output"]) / "official" / f"epoch-{args.epoch:03d}"
    receipt = out / "report.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["run_identity"] != spec["identity"] or result["checkpoint_epoch"] != args.epoch:
            raise ValueError("Evaluation identity changed")
        return result
    saved = out / "checkpoint.pt"
    if saved.exists() and file_sha(saved) != file_sha(checkpoint):
        raise ValueError("Partial evaluation belongs to another checkpoint")
    if not saved.exists():
        out.mkdir(parents=True, exist_ok=True)
        temporary = saved.with_suffix(".part")
        shutil.copyfile(checkpoint, temporary)
        os.replace(temporary, saved)
    report = e2.evaluate_registered(mat, spec, construct_model(c), saved, out, args.epoch, overrides(mat, spec))
    report.update(run_identity=spec["identity"], candidate=c, scoring_status="AWAITING_OFFICIAL_MATLAB")
    write_json(receipt, report)
    return report


def competing_job(argv):
    """Recognize compute jobs without confusing a TensorBoard viewer with a learner."""
    if not argv:
        return None
    if any(Path(a).name == "tensorboard" for a in argv[:2]) or "tensorboard_data_server/bin/server" in argv[0]:
        return None
    executable = argv[0].lower()
    arguments = list(argv[1:])
    while arguments and arguments[0] in ("-u", "-B", "-I", "-s", "-S"):
        arguments.pop(0)
    # Match executable/module/script identity, never arbitrary inline inspection text or log paths.
    if arguments and arguments[0] in ("-c", "-lc", "-c\u0020"):
        return None
    entry = arguments[1] if len(arguments) > 1 and arguments[0] == "-m" else (arguments[0] if arguments else "")
    identity_text = (executable + " " + entry).lower()
    if "gpu_keeper" in identity_text:
        return None
    if "fins" in identity_text and any(word in executable for word in ("python", "unity", ".x86_64")):
        return "simulation"
    if any(word in identity_text for word in ("torch.distributed.run", "torchrun", "run_p5_suite")):
        return "training"
    if entry in ("scripts.d1.run_e3", "scripts/d1/run_e3.py") and any(x in arguments for x in ("train", "suite")):
        return "training"
    return None


def check_resources(workspace):
    """Stop on competitors or memory/disk pressure, without killing unrelated jobs."""
    import psutil

    conflicts = []
    own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            argv = process.info["cmdline"] or []
            command = " ".join(argv)
            if process.pid in own or "gpu_keeper" in command or not argv:
                continue
            kind = competing_job(argv)
            if kind:
                conflicts.append({"pid": process.pid, "kind": kind})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if conflicts:
        raise RuntimeError(f"Resource conflict: {conflicts}")
    cgroup = Path("/sys/fs/cgroup")
    used_file, max_file = cgroup / "memory.current", cgroup / "memory.max"
    if not used_file.exists():
        used_file, max_file = cgroup / "memory/memory.usage_in_bytes", cgroup / "memory/memory.limit_in_bytes"
    used, limit = int(used_file.read_text()), int(max_file.read_text())
    if used / limit > EXPECTED["maximum_cgroup_fraction"]:
        raise RuntimeError("Memory pressure; no force pressure-reclaim during a sweep")
    free = shutil.disk_usage(workspace).free
    if free < EXPECTED["minimum_free_gib"] * 1024**3:
        raise RuntimeError("Insufficient free output space")
    gpu = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(gpu) != 6 or any("A40" not in row or float(row.split(",")[-2]) > 1024 for row in gpu):
        raise RuntimeError("Six idle A40 GPUs required; small GPU-keeper allocations allowed")
    return {
        "memory_used": used,
        "memory_limit": limit,
        "output_free_bytes": free,
        "gpus": gpu,
        "conflicting_jobs": conflicts,
        "updated_unix": time.time(),
    }


def completed(workspace, spec):
    root = Path(spec["output"])
    path = root / "training-result.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    if report["status"] != "completed" or report["epochs"] != spec["window"]:
        return False
    checkpoint = workspace / "runs" / spec["run_id"] / "weights/last.pt"
    if report["last_sha256"] != file_sha(checkpoint) or report["resume_sha256"] != file_sha(root / "resume.pt"):
        raise ValueError("Completed checkpoint checksum changed")
    state = torch.load(root / "resume.pt", map_location="cpu", weights_only=False)
    if state["identity"] != spec["identity"]:
        raise ValueError("Completed run identity changed")
    return True


def launch_train(workspace, c, profile, *, resume=False, window=None):
    mat = matrix(workspace)
    spec = spec_for(mat, c, profile)
    resources = check_resources(workspace)
    jobs = workspace / "jobs"
    jobs.mkdir(exist_ok=True)
    attempt = jobs / f"{spec['run_id']}-{time.time_ns()}"
    attempt_json, attempt_log = Path(str(attempt) + ".json"), Path(str(attempt) + ".log")
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=6",
        "--module",
        "scripts.d1.run_e3",
        "train",
        "--workspace",
        str(workspace),
        "--candidate",
        key(c),
        "--profile",
        profile,
        "--approved",
    ]
    if resume:
        command.append("--resume")
    if window is not None:
        command.extend(["--window", str(window)])
    started = time.time()
    write_json(workspace / "status.json", {"status": "TRAINING", "run_id": spec["run_id"], "started_unix": started})
    write_json(
        attempt_json, {"status": "STARTING", "command": command, "identity": spec["identity"], "resources": resources}
    )
    with attempt_log.open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=clean_child_env(), stdout=log, stderr=subprocess.STDOUT)
        write_json(workspace / "active.json", {"pid": child.pid, "run_id": spec["run_id"], "log": str(attempt_log)})
        code = child.wait()
    elapsed = time.time() - started
    result = {
        "status": "RETURNED" if code == 0 else "FAILED",
        "returncode": code,
        "started_unix": started,
        "ended_unix": time.time(),
        "seconds": elapsed,
        "gpu_hours": elapsed * 6 / 3600,
        "identity": spec["identity"],
        "resources_before": resources,
        "command": command,
    }
    write_json(attempt_json, result)
    if code:
        raise RuntimeError(f"Training failed ({code}); inspect {attempt_log}")
    return result


def validate_run(workspace, spec):
    if not completed(workspace, spec):
        raise ValueError("Run interrupted or incomplete; no automatic candidate exclusion")
    expected_batches = 1 if spec["data_profile"] == "smoke" else 68
    expected_seen = 8 if spec["data_profile"] == "smoke" else 548
    root = Path(spec["output"])
    for epoch in range(1, spec["window"] + 1):
        validation = json.loads((root / "validation" / f"epoch-{epoch:03d}.json").read_text())
        if validation["seen"] != expected_seen:
            raise ValueError("Validation coverage changed")
        for rank in range(6):
            row = json.loads((root / "epochs" / f"rank-{rank}-epoch-{epoch:03d}.json").read_text())
            if row["batches"] != expected_batches or row["optimizer_steps"] != epoch * expected_batches:
                raise ValueError("Missing sampler data or optimizer update")
            probe_file = root / "mechanism" / f"rank-{rank}-epoch-{epoch:03d}.json"
            if json.loads(probe_file.read_text())["candidate"] != spec["candidate"]:
                raise ValueError("Wrong per-epoch mechanism candidate")
    return True


def resume_comparison(workspace, candidate):
    mat = matrix(workspace)
    a, b = [
        torch.load(Path(spec_for(mat, candidate, p)["output"]) / "resume.pt", map_location="cpu", weights_only=False)
        for p in ("smoke", "continuous")
    ]
    worst = 0.0

    def compare(x, y, path):
        nonlocal worst
        if isinstance(x, torch.Tensor):
            if not isinstance(y, torch.Tensor) or x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f"Resume tensor metadata differs: {path}")
            if x.is_floating_point():
                delta = float((x - y).abs().max()) if x.numel() else 0.0
                worst = max(worst, delta)
                if not torch.allclose(x, y, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"Resume numerical mismatch: {path}: {delta}")
            elif not torch.equal(x, y):
                raise ValueError(f"Resume nonfloating tensor differs: {path}")
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                raise ValueError(f"Resume keys differ: {path}")
            for k in x:
                compare(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                raise ValueError(f"Resume sequence differs: {path}")
            for i, (xx, yy) in enumerate(zip(x, y)):
                compare(xx, yy, f"{path}/{i}")
        elif x != y:
            raise ValueError(f"Resume scalar differs: {path}")

    for name in ("model", "ema", "optimizer", "scaler", "scheduler", "ema_updates", "optimizer_steps", "criterion"):
        compare(a[name], b[name], name)
    for rank in range(6):
        for name in ("model_buffers", "ema_buffers", "loader_generator"):
            compare(a["ranks"][rank][name], b["ranks"][rank][name], f"rank{rank}/{name}")
    return {"status": "PASSED", "rtol": 1e-5, "atol": 1e-6, "maximum_tensor_abs_error": worst}


def gates(workspace):
    mat = matrix(workspace)
    receipt = workspace / "gates.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["identity"] != mat["identity"] or result["status"] != "PASSED":
            raise ValueError("Invalid existing gate receipt")
        return result
    default = candidates()[0]
    launch_train(workspace, default, "smoke", window=1)
    launch_train(workspace, default, "smoke", resume=True, window=2)
    launch_train(workspace, default, "continuous")
    for c in (default, {**default, "balance": 0.0, "z": 0.0}, {**default, "balance": 0.0}, {**default, "z": 0.0}):
        spec = spec_for(mat, c, "continuous")
        if not completed(workspace, spec):
            launch_train(workspace, c, "continuous")
        validate_run(workspace, spec)
    comparison = resume_comparison(workspace, default)
    attempt = launch_train(workspace, default, "benchmark")
    spec = spec_for(mat, default, "benchmark")
    validate_run(workspace, spec)
    evaluation = evaluate(argparse.Namespace(workspace=workspace, candidate=key(default), profile="benchmark", epoch=3))
    times = [
        json.loads((Path(spec["output"]) / "validation" / f"epoch-{epoch:03d}.json").read_text())["epoch_wall_seconds"]
        for epoch in (2, 3)
    ]
    probes = [
        max(
            json.loads((Path(spec["output"]) / "mechanism" / f"rank-{rank}-epoch-{epoch:03d}.json").read_text())[
                "seconds"
            ]
            for rank in range(6)
        )
        for epoch in (2, 3)
    ]
    epoch_seconds = statistics.median([a + b for a, b in zip(times, probes)])
    overhead = max(0, attempt["seconds"] - 3 * epoch_seconds)
    per_run = 60 * epoch_seconds + overhead + 12 * (evaluation["seconds"] + 20)
    result = {
        "status": "PASSED",
        "identity": mat["identity"],
        "resume_comparison": comparison,
        "median_epoch_seconds_including_probe": epoch_seconds,
        "estimated_run_seconds": per_run,
        "estimated_stage1_hours": per_run * 27 / 3600,
        "estimated_all36_hours": per_run * 36 / 3600,
        "eta_scope": "Training, probe/validation/checkpoint, startup and 12 GPU exports; MATLAB/transfer separate",
        "official_backend": "official-matlab",
        "official_test_reference": "experiments/d1/manifests/e2-official-evaluation.json",
        "resources": check_resources(workspace),
        "updated_unix": time.time(),
    }
    write_json(receipt, result)
    return result


def rank_groups(rows):
    """Require all three seeds for all nine candidates; raw AP drives the ranking."""
    expected = {(c["balance"], c["z"], c["gain"], c["seed"]) for c in candidates()}
    actual = [(r["balance"], r["z"], r["gain"], r["seed"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate stage1 candidates/seeds")
    if any(not math.isfinite(r["AP_all"]) or not 0 <= r["AP_all"] <= 1 for r in rows):
        raise ValueError("Official AP must be finite and in 0..1")
    groups = []
    for b, z in itertools.product(EXPECTED["balance_candidates"], EXPECTED["z_candidates"]):
        selected = sorted((r for r in rows if r["balance"] == b and r["z"] == z), key=lambda r: r["seed"])
        aps = [r["AP_all"] for r in selected]
        groups.append(
            {
                "balance": b,
                "z": z,
                "AP_values": aps,
                "AP_mean": statistics.mean(aps),
                "AP_sample_std": statistics.stdev(aps),
                "gpu_hours_mean": statistics.mean(r["training_gpu_hours"] for r in selected),
            }
        )
    nonzero = [g for g in groups if g["balance"] or g["z"]]
    winner = min(nonzero, key=lambda g: (-g["AP_mean"], g["AP_sample_std"], g["gpu_hours_mean"], g["balance"], g["z"]))
    return groups, winner


def summarize_stage1(workspace):
    mat = matrix(workspace)
    rows = []
    for c in candidates():
        spec = spec_for(mat, c)
        validate_run(workspace, spec)
        out = Path(spec["output"]) / "official/epoch-060"
        exported = json.loads((out / "report.json").read_text())
        official = validate_official_report(json.loads((out / "official-matlab.json").read_text()))
        if exported["run_identity"] != spec["identity"] or official["image_count"] != 548:
            raise ValueError("Official evaluation identity/coverage mismatch")
        if official["export_sha256"] != file_sha(out / "predictions-txt/export.json"):
            raise ValueError("Official scores refer to different predictions")
        if official["toolkit_manifest_sha256"] != file_sha(ROOT / "experiments/d1/manifests/visdrone-toolkit.json"):
            raise ValueError("Official toolkit manifest changed")
        for name, digest in json.loads((out / "predictions-txt/export.json").read_text())["files_sha256"].items():
            if file_sha(out / "predictions-txt" / name) != digest:
                raise ValueError("Prediction TXT changed after scoring")
        if file_sha(out / "checkpoint.pt") != exported["checkpoint_sha256"]:
            raise ValueError("Evaluation checkpoint changed")
        attempts = [json.loads(p.read_text()) for p in (workspace / "jobs").glob(f"{spec['run_id']}-*.json")]
        if not attempts or any("gpu_hours" not in a for a in attempts):
            raise ValueError("Incomplete cost ledger")
        rows.append({**c, **official["metrics"], "training_gpu_hours": sum(a["gpu_hours"] for a in attempts)})
    groups, winner = rank_groups(rows)
    result = {
        "status": "STAGE1_SCORED_AWAITING_STAGE2_APPROVAL",
        "identity": mat["identity"],
        "rows": rows,
        "groups": groups,
        "selected_nonzero": winner,
        "seeds": [0, 1, 2],
        "stage2_new_runs": 9,
        "AUX_STAR": None,
    }
    write_json(workspace / "stage1-summary.json", result)
    return result


def suite(args):
    if not args.approved:
        raise ValueError("Explicit approval required for acceptance and stage1 training")
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (args.workspace / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            mat = matrix(args.workspace)
            gate = gates(args.workspace)
            if args.gates_only:
                write_json(args.workspace / "status.json", {"status": "GATES_PASSED", "gate": gate})
                return gate
            for index, c in enumerate(candidates()):
                spec = spec_for(mat, c)
                if not completed(args.workspace, spec):
                    resume = (Path(spec["output"]) / "resume.pt").exists()
                    launch_train(args.workspace, c, "screen", resume=resume)
                validate_run(args.workspace, spec)
                for epoch in range(5, 61, 5):
                    receipt = Path(spec["output"]) / "official" / f"epoch-{epoch:03d}/report.json"
                    if not receipt.exists() or json.loads(receipt.read_text())["run_identity"] != spec["identity"]:
                        raise ValueError("Missing scheduled prediction export; no silent omission")
                write_json(
                    args.workspace / "status.json",
                    {
                        "status": "RUN_COMPLETE",
                        "completed": index + 1,
                        "total": 27,
                        "run_id": spec["run_id"],
                        "updated_unix": time.time(),
                    },
                )
            result = {
                "status": "STAGE1_TRAINED_AWAITING_OFFICIAL_MATLAB",
                "training_runs": 27,
                "AUX_STAR": None,
                "stage2_started": False,
                "updated_unix": time.time(),
            }
            write_json(args.workspace / "status.json", result)
            return result
        except Exception as exc:
            write_json(
                args.workspace / "status.json",
                {
                    "status": "FAILED_OR_BLOCKED",
                    "error": repr(exc),
                    "updated_unix": time.time(),
                    "automatic_candidate_exclusion": False,
                },
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate", "suite", "status", "summarize"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--candidate", default=key(candidates()[0]))
    parser.add_argument("--profile", choices=("screen", "smoke", "continuous", "benchmark"), default="screen")
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--window", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--gates-only", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        result = prepare(args.workspace, args.data, args.cache)
    elif args.mode == "train":
        result = train(args)
    elif args.mode == "evaluate":
        result = evaluate(args)
    elif args.mode == "suite":
        result = suite(args)
    elif args.mode == "summarize":
        result = summarize_stage1(args.workspace)
    else:
        result = json.loads((args.workspace / "status.json").read_text())
        active = args.workspace / "active.json"
        if active.exists():
            result["latest_child"] = json.loads(active.read_text())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

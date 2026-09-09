"""Bounded D1 EMA acceptance: six-rank short-tail epochs, validation and exact checkpoint resume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch

from scripts.d1 import run_p5_ablation as p5
from scripts.d1.ema import D1ModelEMA
from scripts.d1.run_wp8_p1_control import write_json
from ultralytics.nn.foundation.cache import sha256_file
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
TRAIN_COUNT, VAL_COUNT = 389, 13


def equal(a, b, key="state"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True, msg=key)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), key
        for name in a:
            equal(a[name], b[name], f"{key}.{name}")
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), key
        for index, (left, right) in enumerate(zip(a, b)):
            equal(left, right, f"{key}.{index}")
    else:
        assert a == b, key


def source_hashes():
    return {
        name: sha256_file(ROOT / name)
        for name in (
            "scripts/d1/ema.py",
            "scripts/d1/run_p5_ablation.py",
            "scripts/d1/check_ema_runtime.py",
            "scripts/d1/p1p2_runtime.py",
            "ultralytics/utils/torch_utils.py",
        )
    }


def prepare(source, output):
    if output == ROOT or ROOT in output.parents or output == source or source in output.parents:
        raise ValueError("Acceptance needs a separate external workspace")
    output.mkdir(parents=True, exist_ok=False)
    matrix = json.loads((source / "matrix.json").read_text())
    if matrix["p3_upsample_mode"] != "separable_bilinear2x":
        raise ValueError("Expected the approved P3 implementation")
    for name, digest in matrix["input_hashes"].items():
        if sha256_file(source / "inputs" / name) != digest:
            raise ValueError(f"Changed source input: {name}")
    dataset = YAML.load(matrix["data_files"]["E1"])
    dataset["path"] = str(output / "dataset")
    selection = {}
    for split, count in (("train", TRAIN_COUNT), ("val", VAL_COUNT)):
        paths = sorted((Path(matrix["data_root"]) / "images" / f"{split}2017").glob("*.jpg"))[:count]
        if len(paths) != count:
            raise ValueError(f"Insufficient existing COCO images: {split}")
        # Keep YOLO's subset label-cache writes away from the original full-data cache.
        linked = []
        images = output / "dataset" / "images" / f"{split}2017"
        labels = output / "dataset" / "labels" / f"{split}2017"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for path in paths:
            image = images / path.name
            image.symlink_to(path)
            label = Path(matrix["data_root"]) / "labels" / f"{split}2017" / f"{path.stem}.txt"
            if label.is_file():
                (labels / label.name).symlink_to(label)
            linked.append(image)
        destination = output / f"{split}.txt"
        destination.write_text("".join(f"{p}\n" for p in linked))
        dataset[split] = str(destination)
        selection[split] = {"count": count, "sha256": sha256_file(destination)}
    YAML.save(output / "data.yaml", dataset)
    write_json(
        output / "acceptance.json",
        {
            "schema": "d1-ema-loop-acceptance-v1",
            "source_workspace": str(source),
            "source_matrix_sha256": sha256_file(source / "matrix.json"),
            "source_hashes": source_hashes(),
            "selection": selection,
            "data_sha256": sha256_file(output / "data.yaml"),
            "formal_training": False,
            "global_batch": 384,
            "world_size": 6,
            "workers_per_rank": 4,
            "epochs": 2,
            "scope": "Small-subset complete loop; not full-COCO throughput or precision acceptance",
        },
    )


def worker(args):
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if int(os.environ.get("WORLD_SIZE", "0")) != 6:
        raise ValueError("Acceptance requires six ranks")
    receipt = json.loads((args.output / "acceptance.json").read_text())
    if receipt["source_hashes"] != source_hashes():
        raise ValueError("Source changed during acceptance")
    source = Path(receipt["source_workspace"])
    if receipt["source_matrix_sha256"] != sha256_file(source / "matrix.json"):
        raise ValueError("Source matrix changed")
    if receipt["data_sha256"] != sha256_file(args.output / "data.yaml"):
        raise ValueError("Acceptance dataset changed")
    for split, row in receipt["selection"].items():
        if row["sha256"] != sha256_file(args.output / f"{split}.txt"):
            raise ValueError("Acceptance split changed")
    matrix = json.loads((source / "matrix.json").read_text())
    matrix["ema_implementation"] = args.implementation
    matrix["resume_temperature_policy"] = "epoch-boundary-v1"
    spec = p5.spec_for(matrix, args.variant)
    suffix = "-continuous" if args.phase == "continuous" else "-resumed"
    run_id = f"{args.variant}-{args.implementation}{suffix}"
    spec.update(
        profile="benchmark",
        run_id=run_id,
        workspace=str(args.output),
        window=2,
        output=str(args.output / "reports" / run_id),
    )
    spec["identity"].update(acceptance_source_hashes=source_hashes(), acceptance_selection=receipt["selection"])
    overrides = p5.overrides_for(matrix, spec)
    overrides.update(
        data=str(args.output / "data.yaml"),
        project=str(args.output / "runs"),
        name=run_id,
        plots=False,
        save=True,
        save_period=-1,
        verbose=False,
    )
    last = args.output / "runs" / run_id / "weights" / "last.pt"
    if args.phase == "resume":
        if not last.is_file():
            raise FileNotFoundError("Missing acceptance checkpoint")
        overrides["resume"] = str(last)
    elif last.exists():
        raise FileExistsError("Refusing to overwrite an acceptance run")
    Path(spec["output"]).mkdir(parents=True, exist_ok=True)
    cache = Path(matrix["cache_root"])
    trainer = p5.P5FrozenTrainer(
        overrides=overrides,
        run_spec=spec,
        feature_caches={"train": cache / "train2017", "val": cache / "val2017"},
        trusted_feature_cache=True,
        max_open_feature_shards=4,
        feature_prefetch_factor=1,
        amp_init_scale=16,
        amp_growth_interval=1000000,
    )

    def installed(t):
        if len(t.train_loader.dataset) != TRAIN_COUNT or len(t.train_loader) != 2:
            raise ValueError("Expected two batches per rank including the short tail")
        if t.ema is not None and isinstance(t.ema, D1ModelEMA) != (args.implementation == "foreach-v1"):
            raise ValueError("Requested EMA implementation was not installed")
        if args.phase == "resume" and t.start_epoch != 1:
            raise ValueError("Expected exact epoch-one resume")

    def stop(t):
        if args.phase == "stop":
            t.e1_stop_requested = True

    trainer.add_callback("on_train_start", installed)
    trainer.add_callback("on_train_epoch_end", stop)
    trainer.train()
    if int(os.environ["RANK"]) == 0:
        raw = torch.load(last, map_location="cpu", weights_only=False)
        ema = raw["ema"]
        fresh = p5.construct_model(args.variant, "separable_bilinear2x")
        fresh.load_state_dict(ema.state_dict(), strict=True)
        if any("teacher" in name.lower() for name, _ in fresh.named_parameters()):
            raise ValueError("Teacher leaked into checkpoint")
        write_json(
            Path(spec["output"]) / f"checkpoint-{args.phase}.json",
            {
                "strict_load": True,
                "teacher_absent": True,
                "sha256": sha256_file(last),
                "epoch": trainer.epoch + 1,
                "ema_updates": trainer.ema.updates,
            },
        )


def child(args, variant, implementation, phase):
    label = f"{variant}-{implementation}-{phase}"
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=6",
        "--module",
        "scripts.d1.check_ema_runtime",
        "worker",
        "--output",
        str(args.output),
        "--variant",
        variant,
        "--implementation",
        implementation,
        "--phase",
        phase,
    ]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH=str(ROOT))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    with (args.output / f"{label}.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        write_json(args.output / "status.json", {"status": "running", "job": label, "pid": process.pid})
        print(f"START {label} pid={process.pid}", flush=True)
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
            raise RuntimeError(f"{label} failed, exit={code}; inspect its log")
        print(f"PASS {label}", flush=True)


def run(args):
    prepare(args.source_workspace, args.output)
    results = {}
    for variant in p5.VARIANTS:
        child(args, variant, "scalar-v1", "continuous")
        child(args, variant, "foreach-v1", "continuous")
        child(args, variant, "scalar-v1", "stop")
        child(args, variant, "scalar-v1", "resume")
        child(args, variant, "foreach-v1", "stop")
        child(args, variant, "foreach-v1", "resume")
        paths = [
            args.output / "reports" / f"{variant}-{mode}"
            for mode in ("scalar-v1-continuous", "foreach-v1-continuous", "scalar-v1-resumed", "foreach-v1-resumed")
        ]
        states = [torch.load(path / "resume.pt", map_location="cpu", weights_only=False) for path in paths]
        checked = ("model", "ema", "optimizer", "scaler", "scheduler", "ema_updates", "optimizer_steps", "criterion")
        for key in checked:
            equal(states[0][key], states[1][key], key)
            equal(states[2][key], states[3][key], key)
        assert all(state["epoch"] == 1 for state in states)
        assert states[0]["optimizer_steps"] == states[0]["ema_updates"] == 4
        resume_comparison = {}
        for name, continuous, resumed in (("scalar-v1", states[0], states[2]), ("foreach-v1", states[1], states[3])):
            exact = True
            for key in checked:
                try:
                    equal(continuous[key], resumed[key], key)
                except AssertionError:
                    exact = False
            squared_delta = squared_norm = maximum = 0.0
            for key, value in continuous["model"].items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel():
                    delta = value.double() - resumed["model"][key].double()
                    squared_delta += float(delta.square().sum())
                    squared_norm += float(value.double().square().sum())
                    maximum = max(maximum, float(delta.abs().max()))
            resume_comparison[name] = {
                "bitwise_equal": exact,
                "model_max_abs": maximum,
                "model_relative_l2": (squared_delta / max(squared_norm, 1e-30)) ** 0.5,
            }
        equal(resume_comparison["scalar-v1"], resume_comparison["foreach-v1"], "resume_drift")
        for path in paths:
            val = json.loads((path / "validation/epoch-002.json").read_text())
            assert val["seen"] == VAL_COUNT
            for rank in range(6):
                for epoch in (1, 2):
                    row = json.loads((path / "epochs" / f"rank-{rank}-epoch-{epoch:03d}.json").read_text())
                    assert row["amp_retries"] == 0 and row["batches"] == 2
        results[variant] = {
            "status": "passed",
            "exact_fields": list(checked),
            "epochs": 2,
            "optimizer_steps_per_rank": 4,
            "ema_updates": 4,
            "validation_images": VAL_COUNT,
            "short_tail_batch_per_rank": 1,
            "resume_epoch": 1,
            "amp_retries": 0,
            "ema_continuous_exact": True,
            "ema_resumed_exact": True,
            "continuous_vs_resumed": resume_comparison,
        }
        write_json(args.output / "summary.json", {"status": "running", "results": results})
    write_json(
        args.output / "summary.json",
        {
            "status": "ema_passed_with_resume_limitation"
            if any(not row["continuous_vs_resumed"]["scalar-v1"]["bitwise_equal"] for row in results.values())
            else "passed",
            "results": results,
            "source_hashes": source_hashes(),
            "formal_training": False,
            "scope": "Six-rank small-subset full loops, not full-COCO epoch or AP equivalence",
        },
    )
    write_json(
        args.output / "status.json",
        {
            "status": "completed",
            "ema_gates": "passed",
            "full_resume_bitwise_equal": all(
                row["continuous_vs_resumed"]["scalar-v1"]["bitwise_equal"] for row in results.values()
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--source-workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(p5.VARIANTS))
    parser.add_argument("--implementation", choices=("scalar-v1", "foreach-v1"))
    parser.add_argument("--phase", choices=("continuous", "stop", "resume"))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "run":
        if args.source_workspace is None:
            parser.error("--source-workspace is required")
        args.source_workspace = args.source_workspace.resolve()
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite an existing acceptance workspace")
        try:
            run(args)
        except BaseException as error:
            if args.output.is_dir():
                write_json(args.output / "status.json", {"status": "failed", "error": repr(error)})
            raise
    else:
        worker(args)


if __name__ == "__main__":
    main()

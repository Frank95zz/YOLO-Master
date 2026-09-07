"""Launch one approved frozen-DINOv3 NPY run with owned processes and cleanup."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from scripts.d1.run_wp8_train import code_fingerprint, git_commit, write_json

ROOT = Path(__file__).resolve().parents[2]


def keeper_command(script: Path, command: str) -> list[str]:
    return [
        str(script.parent / ".venv/bin/python3"),
        str(script),
        command,
        "--target",
        "10",
        "--cycle",
        "0.1",
        "--control-every",
        "1",
        "--pidfile",
        str(script.parent / "gpu_keeper_v2.pid"),
        "--log",
        str(script.parent / "gpu_keeper_v2.log"),
        "--status-file",
        str(script.parent / "gpu_keeper_v2_status.json"),
    ]


class Pipeline:
    """Fail closed on preflight errors and restore the keeper after this job exits."""

    def __init__(self, args):
        self.args = args
        if not args.approved:
            raise ValueError("Explicit approval is required before launching training.")
        if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
            raise ValueError("run-id must be a single new directory name.")
        for path in (args.workspace, args.data_root, args.cache_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
            filesystem = subprocess.check_output(["stat", "-f", "-c", "%T", str(path)], text=True).strip()
            if filesystem.startswith("nfs"):
                raise ValueError(f"This launch requires local storage, got {filesystem}: {path}")
        self.output = args.workspace / "manifests" / args.run_id
        self.run_root = args.workspace / "runs" / args.run_id
        if self.run_root.exists():
            raise FileExistsError("Fresh restart must not overwrite an existing experiment.")
        self.output.mkdir(parents=True, exist_ok=False)
        self.child = None
        self.restore_keeper = False
        self.state = {
            "run_id": args.run_id,
            "pid": os.getpid(),
            "approved": True,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_commit": git_commit(ROOT),
            "code_fingerprint": code_fingerprint(ROOT),
            "restart_mode": "fresh",
            "aux_contract": "scalar-once-v1",
            "run_directory": str(self.run_root),
        }
        (self.output / "run.pid").write_text(str(os.getpid()) + "\n")

    def status(self, phase, **fields):
        self.state.update(fields)
        self.state.update(phase=phase, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(self.output / "status.json", self.state)
        print(f"[{self.state['updated_utc']}] {phase}", flush=True)

    def stop_child(self):
        if self.child is None or self.child.poll() is not None:
            return
        os.killpg(self.child.pid, signal.SIGTERM)
        try:
            self.child.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()

    def execute(self, name, command):
        log = self.output / f"{name.lower()}.log"
        env = dict(
            os.environ,
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            PYTHONUNBUFFERED="1",
            MPLBACKEND="Agg",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        with log.open("x") as stream:
            self.child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.status(name, child_pid=self.child.pid, child_log=str(log), command=command)
        code = self.child.wait()
        self.child = None
        if code:
            raise RuntimeError(f"{name} exited with code {code}; see {log}")

    def run(self):
        args = self.args
        base = [
            "--workspace",
            str(args.workspace),
            "--data-root",
            str(args.data_root),
            "--train-cache",
            str(args.cache_root / "train2017"),
            "--val-cache",
            str(args.cache_root / "val2017"),
            "--run-root",
            str(self.run_root),
            "--report-dir",
            str(self.output),
        ]
        self.status("PREPARING")
        # Archive exact runtime sources, including uncommitted fixes, before launch.
        with tarfile.open(self.output / "runtime-sources.tar.gz", "w:gz") as archive:
            for directory in ("ultralytics", "scripts/d1"):
                for path in sorted((ROOT / directory).rglob("*")):
                    if path.is_file() and path.suffix in {".py", ".yaml"}:
                        archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        self.execute("PREFLIGHT", [sys.executable, "-m", "scripts.d1.run_wp8_train", *base, "prepare"])
        if args.keeper_script:
            pidfile = args.keeper_script.parent / "gpu_keeper_v2.pid"
            if pidfile.is_file():
                pid = int(pidfile.read_text())
                cmdline = Path(f"/proc/{pid}/cmdline")
                if cmdline.exists():
                    tokens = cmdline.read_bytes().split(b"\0")
                    if str(args.keeper_script).encode() not in tokens:
                        raise RuntimeError("Keeper pidfile belongs to a different process.")
                    self.restore_keeper = True
                    subprocess.run(keeper_command(args.keeper_script, "stop"), check=True, timeout=30)
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            raise RuntimeError(f"Other GPU processes are still active: {active}")
        training = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=6",
            "-m",
            "scripts.d1.run_wp8_train",
            *base,
            "train",
        ]
        wrapped = [
            sys.executable,
            "-m",
            "scripts.d1.run_with_cache_cleanup",
            "--cache-path",
            str(args.cache_root / "train2017"),
            "--cache-path",
            str(args.cache_root / "val2017"),
            "--suffix",
            ".npy",
            "--wait-seconds",
            "0",
            "--report",
            str(self.output / "cleanup.json"),
            "--",
            *training,
        ]
        self.execute("TRAINING", wrapped)
        self.execute("SUMMARIZING", [sys.executable, "-m", "scripts.d1.run_wp8_train", *base, "summarize"])
        self.status("COMPLETED", exit_code=0)

    def cleanup(self):
        self.stop_child()
        if self.restore_keeper:
            subprocess.run(keeper_command(self.args.keeper_script, "start"), check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--keeper-script", type=Path)
    parser.add_argument("--approved", action="store_true")
    args = parser.parse_args()
    for field in ("workspace", "data_root", "cache_root"):
        setattr(args, field, getattr(args, field).resolve())
    pipeline = Pipeline(args)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.status("STOPPED", exit_code=130)
        return 130
    except Exception as exc:
        pipeline.status("FAILED", error=f"{type(exc).__name__}: {exc}", exit_code=1)
        raise
    finally:
        pipeline.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

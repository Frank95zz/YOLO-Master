#!/usr/bin/env python3
"""Run a command and release selected feature-cache pages after it exits."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "d1-cache-release-v1"
DEFAULT_SUFFIXES = (".safetensors",)
CGROUP_V1_ROOT = Path("/sys/fs/cgroup/memory")
CGROUP_V2_ROOT = Path("/sys/fs/cgroup")


def _read_int(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="ascii").strip()
    return None if value == "max" else int(value)


def _read_memory_stat(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    return {
        name: int(value)
        for line in path.read_text(encoding="ascii").splitlines()
        for name, value in [line.split()]
    }


def cgroup_memory_snapshot(
    v1_root: Path = CGROUP_V1_ROOT,
    v2_root: Path = CGROUP_V2_ROOT,
) -> dict[str, int | str | None]:
    """Read the current container memory counters from cgroup v1 or v2."""
    v1_usage = v1_root / "memory.usage_in_bytes"
    if v1_usage.is_file():
        stats = _read_memory_stat(v1_root / "memory.stat")
        return {
            "cgroup_version": 1,
            "usage_bytes": _read_int(v1_usage),
            "limit_bytes": _read_int(v1_root / "memory.limit_in_bytes"),
            "rss_bytes": stats.get("total_rss", stats.get("rss")),
            "cache_bytes": stats.get("total_cache", stats.get("cache")),
            "kernel_bytes": _read_int(v1_root / "memory.kmem.usage_in_bytes"),
        }
    v2_usage = v2_root / "memory.current"
    if v2_usage.is_file():
        stats = _read_memory_stat(v2_root / "memory.stat")
        return {
            "cgroup_version": 2,
            "usage_bytes": _read_int(v2_usage),
            "limit_bytes": _read_int(v2_root / "memory.max"),
            "rss_bytes": stats.get("anon"),
            "cache_bytes": stats.get("file"),
            "kernel_bytes": stats.get("kernel"),
        }
    return {
        "cgroup_version": "unavailable",
        "usage_bytes": None,
        "limit_bytes": None,
        "rss_bytes": None,
        "cache_bytes": None,
        "kernel_bytes": None,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def normalize_cache_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve, validate, and deduplicate explicitly selected cache paths."""
    if not paths:
        raise ValueError("at least one --cache-path is required")
    result = []
    seen = set()
    for raw_path in paths:
        expanded = raw_path.expanduser()
        if expanded.is_symlink():
            raise ValueError(f"cache root must not be a symbolic link: {expanded}")
        path = expanded.resolve()
        if path == Path(path.anchor):
            raise ValueError(f"refusing to scan a filesystem root: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"cache path must be a regular file or directory: {path}")
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


def normalize_suffixes(suffixes: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize suffix filters used while scanning cache directories."""
    values = suffixes or DEFAULT_SUFFIXES
    result = []
    for value in values:
        if not value or value == ".":
            raise ValueError("cache suffixes must not be empty")
        suffix = value if value.startswith(".") else f".{value}"
        if suffix not in result:
            result.append(suffix)
    return tuple(result)


def _cache_files(roots: Sequence[Path], suffixes: tuple[str, ...]):
    for root in roots:
        if root.is_file():
            if root.name.endswith(suffixes):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            directory = Path(dirpath)
            dirnames[:] = [name for name in dirnames if not (directory / name).is_symlink()]
            for filename in filenames:
                path = directory / filename
                if filename.endswith(suffixes) and not path.is_symlink():
                    yield path


def release_file_cache(
    paths: Sequence[Path],
    *,
    suffixes: Sequence[str] | None = None,
    wait_seconds: float = 10.0,
) -> dict[str, Any]:
    """Advise Linux to discard clean page-cache entries for selected cache files."""
    if wait_seconds < 0:
        raise ValueError("wait_seconds must be non-negative")
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is not available on this platform")
    roots = normalize_cache_paths(paths)
    normalized_suffixes = normalize_suffixes(suffixes)
    before = cgroup_memory_snapshot()
    files_advised = 0
    logical_bytes = 0
    errors = []
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    started = time.monotonic()
    for path in _cache_files(roots, normalized_suffixes):
        try:
            descriptor = os.open(path, flags)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    continue
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                files_advised += 1
                logical_bytes += info.st_size
            finally:
                os.close(descriptor)
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})
    if wait_seconds:
        time.sleep(wait_seconds)
    after = cgroup_memory_snapshot()
    before_usage = before.get("usage_bytes")
    after_usage = after.get("usage_bytes")
    estimated_released = None
    if isinstance(before_usage, int) and isinstance(after_usage, int):
        estimated_released = max(0, before_usage - after_usage)
    status = "passed" if files_advised and not errors else "failed"
    return {
        "status": status,
        "paths": [str(path) for path in roots],
        "suffixes": list(normalized_suffixes),
        "files_advised": files_advised,
        "logical_bytes": logical_bytes,
        "estimated_released_bytes": estimated_released,
        "elapsed_seconds": time.monotonic() - started,
        "memory_before": before,
        "memory_after": after,
        "errors": errors,
    }


def _command_tokens(values: Sequence[str]) -> list[str]:
    result = list(values)
    if result[:1] == ["--"]:
        result.pop(0)
    return result


def run_wrapped_command(command: Sequence[str], *, termination_grace_seconds: float = 30.0) -> dict[str, Any]:
    """Run a child process group and forward termination signals to it."""
    if termination_grace_seconds <= 0:
        raise ValueError("termination_grace_seconds must be positive")
    tokens = _command_tokens(command)
    if not tokens:
        return {"executed": False, "exit_code": 0, "signal": None, "elapsed_seconds": 0.0}
    started = time.monotonic()
    child = subprocess.Popen(tokens, start_new_session=True)
    received_signal: int | None = None
    forwarded_at: float | None = None
    previous_handlers = {}

    def forward(signum, _frame) -> None:
        nonlocal received_signal, forwarded_at
        if received_signal is None:
            received_signal = signum
            forwarded_at = time.monotonic()
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    try:
        while child.poll() is None:
            if forwarded_at is not None and time.monotonic() - forwarded_at >= termination_grace_seconds:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(0.2)
        child_returncode = int(child.returncode)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if received_signal is not None:
        exit_code = 128 + received_signal
    elif child_returncode < 0:
        exit_code = 128 - child_returncode
    else:
        exit_code = child_returncode
    return {
        "executed": True,
        "exit_code": exit_code,
        "child_returncode": child_returncode,
        "signal": received_signal,
        "elapsed_seconds": time.monotonic() - started,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run a command and release selected feature-cache pages after it exits."
    )
    result.add_argument("--cache-path", action="append", type=Path, required=True)
    result.add_argument("--suffix", action="append", help="file suffix to release; defaults to .safetensors")
    result.add_argument("--wait-seconds", type=float, default=10.0)
    result.add_argument("--termination-grace-seconds", type=float, default=30.0)
    result.add_argument("--report", type=Path)
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command_report: dict[str, Any]
    command_error = None
    try:
        command_report = run_wrapped_command(
            args.command,
            termination_grace_seconds=args.termination_grace_seconds,
        )
    except (OSError, ValueError) as exc:
        command_error = f"{type(exc).__name__}: {exc}"
        command_report = {"executed": bool(_command_tokens(args.command)), "exit_code": 127, "error": command_error}
    try:
        cleanup_report = release_file_cache(
            args.cache_path,
            suffixes=args.suffix,
            wait_seconds=args.wait_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cleanup_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    command_exit = int(command_report["exit_code"])
    if command_exit:
        status = "command_failed"
    elif cleanup_report.get("status") != "passed":
        status = "cleanup_failed"
    else:
        status = "passed"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "command": command_report,
        "cleanup": cleanup_report,
    }
    if args.report:
        atomic_write_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return command_exit or (1 if status != "passed" else 0)


if __name__ == "__main__":
    raise SystemExit(main())

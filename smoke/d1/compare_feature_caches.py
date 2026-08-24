#!/usr/bin/env python3
"""Compare two independent D1 feature-cache runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first = load_manifest(args.first / "manifest.jsonl")
    second = load_manifest(args.second / "manifest.jsonl")
    first_pairs = [(item["source_sha256"], item["feature_tensor_sha256"]) for item in first]
    second_pairs = [(item["source_sha256"], item["feature_tensor_sha256"]) for item in second]
    mismatches = [index for index, pair in enumerate(first_pairs) if index >= len(second_pairs) or pair != second_pairs[index]]
    payload = {
        "result": "PASS" if len(first) == len(second) == 100 and not mismatches else "FAIL",
        "first": str(args.first),
        "second": str(args.second),
        "image_count_first": len(first),
        "image_count_second": len(second),
        "matching_tensor_hashes": len(first) - len(mismatches),
        "mismatch_indices": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

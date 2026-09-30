#!/usr/bin/env python3
"""Validate exact placewipe rollout manifests before or after coord training."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

MODE_RE = re.compile(r"mode(-?\d+)")
MANIFEST_KEYS = (
    "twoarm_train_files",
    "singlearm_train_files",
    "twoarm_validation_files",
    "singlearm_validation_files",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stats", required=True)
    parser.add_argument("--output-stats")
    parser.add_argument("--expected-twoarm", type=int, required=True)
    parser.add_argument("--expected-singlearm", type=int, required=True)
    parser.add_argument("--expected-modes", default="0,1,2,3")
    parser.add_argument(
        "--allow-missing-rollouts",
        action="store_true",
        help="Validate names/counts without requiring every rollout file to exist.",
    )
    return parser.parse_args(argv)


def _load_pickle(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected stats dictionary: {path}")
    return value


def exact_manifest_from_stats(stats: dict[str, Any]) -> dict[str, list[str]]:
    embedded = stats.get("dataset_manifest")
    if isinstance(embedded, dict):
        return {key: [str(x) for x in embedded.get(key, [])] for key in MANIFEST_KEYS}
    mapping = {
        "twoarm_train_files": "twoarm_selected_rollout_files",
        "singlearm_train_files": "singlearm_selected_rollout_files",
        "twoarm_validation_files": "twoarm_validation_rollout_files",
        "singlearm_validation_files": "singlearm_validation_rollout_files",
    }
    missing = [source for source in mapping.values() if source not in stats]
    if missing:
        raise KeyError(f"Stats missing exact manifest fields: {missing}")
    return {
        target: [str(x) for x in stats[source]]
        for target, source in mapping.items()
    }


def _modes(files: list[str]) -> Counter[int]:
    modes: Counter[int] = Counter()
    for path in files:
        match = MODE_RE.search(path)
        if match is None:
            raise ValueError(f"Rollout filename has no mode token: {path}")
        modes[int(match.group(1))] += 1
    return modes


def validate_manifest(
    manifest: dict[str, list[str]],
    *,
    expected_twoarm: int,
    expected_singlearm: int,
    expected_modes: set[int],
    require_files: bool,
) -> dict[str, Any]:
    ta = manifest["twoarm_train_files"]
    sa = manifest["singlearm_train_files"]
    if len(ta) != expected_twoarm:
        raise ValueError(f"twoarm train count {len(ta)} != {expected_twoarm}")
    if len(sa) != expected_singlearm:
        raise ValueError(f"singlearm train count {len(sa)} != {expected_singlearm}")
    all_lists = [manifest[key] for key in MANIFEST_KEYS]
    flat = [path for values in all_lists for path in values]
    if len(flat) != len(set(flat)):
        duplicates = sorted(path for path, count in Counter(flat).items() if count > 1)
        raise ValueError(f"Manifest contains duplicate train/validation files: {duplicates}")
    if require_files:
        missing = [path for path in flat if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"Manifest rollout files missing ({len(missing)}): {missing[:5]}")
    ta_modes = _modes(ta)
    if set(ta_modes) != expected_modes:
        raise ValueError(
            f"twoarm modes {sorted(ta_modes)} != expected {sorted(expected_modes)}"
        )
    return {
        "contract": "exact_manifest_match",
        "twoarm_train_count": len(ta),
        "singlearm_train_count": len(sa),
        "twoarm_validation_count": len(manifest["twoarm_validation_files"]),
        "singlearm_validation_count": len(manifest["singlearm_validation_files"]),
        "twoarm_per_mode": dict(sorted(ta_modes.items())),
        "singlearm_per_mode": dict(sorted(_modes(sa).items())),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    source_manifest = exact_manifest_from_stats(_load_pickle(args.source_stats))
    if args.output_stats:
        output_manifest = exact_manifest_from_stats(_load_pickle(args.output_stats))
        if source_manifest != output_manifest:
            raise ValueError("Output stats manifest is not exactly equal to source stats")
    summary = validate_manifest(
        source_manifest,
        expected_twoarm=args.expected_twoarm,
        expected_singlearm=args.expected_singlearm,
        expected_modes={int(x) for x in args.expected_modes.split(",") if x},
        require_files=not args.allow_missing_rollouts,
    )
    summary["source_stats"] = str(Path(args.source_stats).resolve())
    summary["output_stats"] = (
        str(Path(args.output_stats).resolve()) if args.output_stats else None
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

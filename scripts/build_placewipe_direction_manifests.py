#!/usr/bin/env python3
"""Build the immutable 10-demonstration/mode PlaceWipe direction manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path


SCHEMA = "placewipe_direction_exact_manifest.v1"
MODE_RE = re.compile(r"mode(-?\d+)")
DIRECTIONS = {
    "forward_only": {"place_return": (0, 1), "wipe": (0, 1)},
    "two_frame_all_modes": {"place_return": (0, 1, 2, 3), "wipe": (0, 1)},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stats", required=True)
    parser.add_argument("--output-dir", default="manifests/placewipe")
    parser.add_argument("--selection-seed", type=int, default=0)
    return parser.parse_args(argv)


def _mode(path: str) -> int:
    match = MODE_RE.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"Rollout filename has no mode token: {path}")
    return int(match.group(1))


def _by_mode(files, expected_modes):
    result = {str(mode): [] for mode in expected_modes}
    for path in sorted(map(os.path.abspath, files)):
        mode = _mode(path)
        if mode in expected_modes:
            result[str(mode)].append(path)
    bad = {mode: len(paths) for mode, paths in result.items() if len(paths) != 10}
    if bad:
        raise ValueError(f"Expected exactly 10 selected demonstrations/mode, got {bad}")
    return result


def _write_immutable(path: Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"Refusing to overwrite a different manifest: {path}")
        print(f"Unchanged: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    print(f"Wrote: {path}")


def build_manifests(source_stats: str, selection_seed: int) -> dict[str, dict]:
    raw_bytes = Path(source_stats).read_bytes()
    stats = pickle.loads(raw_bytes)
    twoarm = list(stats["twoarm_selected_rollout_files"])
    singlearm = list(stats["singlearm_selected_rollout_files"])
    place_return = [path for path in singlearm if "singlearm_place_return" in path]
    wipe = [path for path in singlearm if "singlearm_wipe" in path]
    if len(twoarm) != 40 or len(place_return) != 40 or len(wipe) != 20:
        raise ValueError(
            "Canonical source must contain TA40 + place-return40 + wipe20; "
            f"got {len(twoarm)} + {len(place_return)} + {len(wipe)}"
        )
    ta_by_mode = _by_mode(twoarm, (0, 1, 2, 3))
    outputs = {}
    for direction, sources in DIRECTIONS.items():
        outputs[direction] = {
            "schema": SCHEMA,
            "direction": direction,
            "selection_seed": int(selection_seed),
            "per_mode_demo_budget": 10,
            "source_stats_path": os.path.abspath(source_stats),
            "source_stats_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "twoarm": {"files_by_mode": ta_by_mode},
            "singlearm": {
                "place_return": {"files_by_mode": _by_mode(place_return, sources["place_return"])},
                "wipe": {"files_by_mode": _by_mode(wipe, sources["wipe"])},
            },
        }
    return outputs


def main(argv=None) -> int:
    args = parse_args(argv)
    outputs = build_manifests(args.source_stats, args.selection_seed)
    output_dir = Path(args.output_dir)
    for direction, payload in outputs.items():
        _write_immutable(output_dir / f"{direction}_10permode.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

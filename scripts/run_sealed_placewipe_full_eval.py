#!/usr/bin/env python3
"""Run the immutable two-arm PlaceWipe 200-episode full-task panel.

``run_eval.py`` owns model execution.  This adapter owns the protocol: it
writes the contract before starting that process, tails its append-only
progress stream into the sealed episode ledger, and emits a final result only
after exact matrix and artifact validation succeeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sealed_evaluation_protocol import (
    append_rollout_record,
    build_contract,
    expected_keys,
    finalize,
    initialize_output,
    read_rollouts,
    validate_rollouts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(1000, 1050))
MODES = tuple(range(4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("coord", "fs"), required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--base-stats-path", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-seeds", default="0")
    parser.add_argument("--sampling-seed-base", type=int, default=20260908)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _record_from_metrics(metrics: dict[str, Any], *, started: float) -> dict[str, Any]:
    mode = int(metrics["mode"])
    seed = int(metrics["seed"])
    success = metrics.get("task_success")
    return {
        "task": "placewipe",
        "mode": mode,
        "seed": seed,
        "repeat": int(metrics.get("repeat", 0)),
        "success": success if isinstance(success, bool) else False,
        "steps": metrics.get("steps", metrics.get("steps_executed")),
        "wall_time_seconds": time.monotonic() - started,
        # The native runner does not expose a two-arm geometry snapshot.  This
        # explicit scene provenance still fixes the deterministic hard scene,
        # task mode, and geometry seed used for the episode.
        "geometry": {
            "env_class": "TwoArmPlaceWipeHard", "env_variant": "hard",
            "mode": mode, "geometry_seed": seed,
        },
        "scene": {"task": "placewipe", "env_variant": "hard"},
        "error": None if isinstance(success, bool) else "missing Boolean task_success metric",
    }


def _drain_progress(root: Path, offsets: dict[Path, int], started: float) -> None:
    """Convert newly flushed generic-progress rows into sealed JSONL rows."""

    existing = {(row["task"], int(row["mode"]), int(row["seed"])) for row in read_rollouts(root)} \
        if (root / "rollouts.jsonl").is_file() else set()
    for path in sorted(root.rglob("eval_progress_*.jsonl")):
        offset = offsets.get(path, 0)
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(offset)
            lines = stream.readlines()
            offsets[path] = stream.tell()
        for line in lines:
            metrics = json.loads(line)
            row = _record_from_metrics(metrics, started=started)
            key = (row["task"], row["mode"], row["seed"])
            if key in existing:
                continue
            append_rollout_record(root, row)
            existing.add(key)


def _command(args: argparse.Namespace) -> list[str]:
    mode = "coord-shoulderonly" if args.method == "coord" else "fromscratch-shoulderonly"
    command = [
        str(args.python), "-u", "scripts/run_eval.py", "--mode", mode,
        "--task", "placewipe", "--env-variant", "hard",
        "--strict-experiment-contract", "--experiment-profile", "headline_hard",
        "--eval-purpose", "sealed_twoarm_full200",
        "--seeds", ",".join(map(str, SEEDS)), "--repeats", "1",
        "--mode-list", ",".join(map(str, MODES)), "--replan-freq", "15",
        "--max-steps", "1800", "--no-video", "--device", args.device,
        "--sealed-sampling-seed-base", str(args.sampling_seed_base),
        "--checkpoint-path", str(args.checkpoint_path),
        "--adapter-stats-path", str(args.stats_path), "--output-dir", str(args.output_root),
    ]
    if args.method == "coord":
        command.extend([
            "--base-model-path", str(args.base_model_path),
            "--base-stats-path", str(args.base_stats_path), "--selector-mode", "coord",
        ])
    return command


def main() -> None:
    args = parse_args()
    contract = build_contract(
        protocol="twoarm_full200", method=args.method,
        checkpoint=args.checkpoint_path, stats=args.stats_path,
        base_model=args.base_model_path, base_stats=args.base_stats_path,
        training_seeds=args.training_seeds, sampling_seed_base=args.sampling_seed_base,
    )
    root = initialize_output(args.output_root, contract, resume=args.resume)
    if args.dry_run:
        if args.resume:
            raise ValueError("--dry-run cannot resume an existing evaluation")
        print(json.dumps(contract, indent=2, sort_keys=True))
        return
    if args.resume:
        final_path = root / "evaluation.json"
        if not final_path.is_file():
            raise ValueError(
                "the generic two-arm evaluator cannot safely resume partial rollouts; "
                "use a fresh output root after investigating the interrupted panel"
            )
        validate_rollouts(contract, read_rollouts(root))
        print(final_path)
        return

    started = time.monotonic()
    process = subprocess.Popen(_command(args), cwd=REPO_ROOT)
    offsets: dict[Path, int] = {}
    while process.poll() is None:
        _drain_progress(root, offsets, started)
        time.sleep(0.25)
    _drain_progress(root, offsets, started)
    if process.returncode != 0:
        raise RuntimeError(f"two-arm full evaluator failed with exit code {process.returncode}")
    expected = expected_keys("twoarm_full200")
    rows = read_rollouts(root)
    if len(rows) != len(expected):
        raise RuntimeError(f"two-arm full evaluator wrote {len(rows)} rows; expected {len(expected)}")
    result = finalize(root)
    print(json.dumps({"status": "pass", "result": str(result), "episodes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one sealed two-arm PlaceWipe single-arm source-retention panel."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

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
    sampling_seed,
    validate_rollouts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLER = REPO_ROOT / "envs/arm/sample/sample_single_arm_placewipe_shoulder.py"
SUCCESS_RE = re.compile(r"task_success=(True|False)")
STEPS_RE = re.compile(r"steps=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("base", "coord", "fs"), required=True)
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
    parser.add_argument("--protocol", choices=("twoarm_source200", "twoarm_source300", "twoarm_native_source200"), default="twoarm_source300")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sampler_command(args: argparse.Namespace, task: str, mode: int, seed: int) -> list[str]:
    if args.method == "coord":
        model, stats = args.base_model_path, args.base_stats_path
    else:
        model, stats = args.checkpoint_path, args.stats_path
    assert model is not None and stats is not None
    command = [
        str(args.python), str(SAMPLER), "--model-path", str(model), "--stats-path", str(stats),
        "--subtask", task, "--seed", str(seed), "--mode", str(mode),
        "--env-variant", "hard", "--max-steps", "1800", "--replan-freq", "15",
        "--sampling-seed", str(sampling_seed(base=args.sampling_seed_base, seed=seed, mode=mode)),
        "--no-warm-start", "--no-video", "--device", args.device,
    ]
    if args.method == "coord":
        command.extend([
            "--selector-mode", "coord", "--coord-head-path", str(args.checkpoint_path),
            "--coord-stats-path", str(args.stats_path),
        ])
    else:
        command.extend(["--selector-mode", "base"])
    return command


def _row(task: str, mode: int, seed: int, output: str, returncode: int, elapsed: float) -> dict:
    match = SUCCESS_RE.findall(output)
    steps = STEPS_RE.findall(output)
    error = None
    if returncode != 0:
        error = f"sampler exited {returncode}"
    elif len(match) != 1:
        error = "sampler did not emit exactly one Boolean task_success"
    return {
        "task": task, "mode": mode, "seed": seed, "repeat": 0,
        "success": match[-1] == "True" if match else False,
        "steps": int(steps[-1]) if steps else None,
        "wall_time_seconds": elapsed,
        "geometry": {
            "env_class": "SingleArmPlaceWipeHard", "env_variant": "hard",
            "subtask": task, "mode": mode, "geometry_seed": seed,
        },
        "scene": {"task": task, "env_variant": "hard", "camera": "shoulder_only"},
        "error": error,
    }


def main() -> None:
    args = parse_args()
    contract = build_contract(
        protocol=args.protocol, method=args.method,
        checkpoint=args.checkpoint_path, stats=args.stats_path,
        base_model=args.base_model_path, base_stats=args.base_stats_path,
        training_seeds=args.training_seeds, sampling_seed_base=args.sampling_seed_base,
    )
    if args.dry_run:
        if args.output_root.exists() or args.resume:
            raise ValueError("--dry-run requires a fresh output root and no --resume")
        print(json.dumps(contract, indent=2, sort_keys=True))
        return
    root = initialize_output(args.output_root, contract, resume=args.resume)
    final_path = root / "evaluation.json"
    if final_path.is_file():
        validate_rollouts(contract, read_rollouts(root))
        print(final_path)
        return
    rows = read_rollouts(root) if (root / "rollouts.jsonl").is_file() else []
    if any(row.get("error") is not None for row in rows):
        raise ValueError("cannot resume a panel containing evaluator-error records; use a fresh root")
    completed = {(str(row["task"]), int(row["mode"]), int(row["seed"])) for row in rows}
    for task, mode, seed in sorted(expected_keys(args.protocol)):
        if (task, mode, seed) in completed:
            continue
        log_path = root / "logs" / f"{task}_mode{mode}_seed{seed}.log"
        started = time.monotonic()
        process = subprocess.run(
            _sampler_command(args, task, mode, seed), cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        elapsed = time.monotonic() - started
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(process.stdout, encoding="utf-8")
        append_rollout_record(root, _row(task, mode, seed, process.stdout, process.returncode, elapsed))
    result = finalize(root)
    print(json.dumps({
        "status": "pass",
        "result": str(result),
        "episodes": len(expected_keys(args.protocol)),
    }, indent=2))


if __name__ == "__main__":
    main()

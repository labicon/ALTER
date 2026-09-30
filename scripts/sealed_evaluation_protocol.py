#!/usr/bin/env python3
"""Shared immutable contracts and coverage checks for final evaluations.

The helpers in this module deliberately operate on small JSON artifacts only.
They are used by evaluators, launchers, and tests so that an episode count is
derived from its task/mode/seed matrix instead of being a comment in a shell
script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CONTRACT_SCHEMA = "sealed_evaluation_contract.v1"
RESULT_SCHEMA = "sealed_evaluation_result.v1"
LEDGER_SCHEMA = "sealed_evaluation_ledger.v1"
SAMPLING_SEED_RULE = "base + 1000003*seed + 10007*mode + 101*repeat"


def _bucket(task: str, modes: Iterable[int], seeds: Iterable[int]) -> dict[str, Any]:
    return {"task": task, "modes": list(modes), "seeds": list(seeds)}


# Each bucket owns exactly fifty training-disjoint geometry seeds.  The
# three-arm source panel intentionally keeps the existing legacy/tray ranges.
PROTOCOLS: dict[str, dict[str, Any]] = {
    "twoarm_full200": {
        "task_family": "two_arm_placewipe",
        "panel_type": "full_task",
        "buckets": [_bucket("placewipe", range(4), range(1000, 1050))],
        "sampler": {
            "env_variant": "hard", "max_steps": 1800, "replan_freq": 15,
            "guidance_w": 1.2, "sample_N": 50, "warm_start": False,
            "video": False,
        },
        "success_criterion": "run_eval per-episode task_success is true",
    },
    "twoarm_source300": {
        "task_family": "two_arm_placewipe",
        "panel_type": "source_retention",
        "buckets": [
            _bucket("place_return", range(4), range(1000, 1050)),
            _bucket("wipe", range(2), range(1000, 1050)),
        ],
        "sampler": {
            "env_variant": "hard", "max_steps": 1800, "replan_freq": 15,
            "guidance_w": 1.2, "sample_N": 50, "warm_start": False,
            "video": False,
        },
        "success_criterion": "SingleArmPlaceWipeMPCPlayer task_success",
    },
    # Legacy compatibility panel. Its place-return modes 2–3 are not part of
    # the source distribution used by the low-data capacity experiments.
    "twoarm_source200": {
        "task_family": "two_arm_placewipe",
        "panel_type": "source_retention",
        "buckets": [_bucket("place_return", range(4), range(1000, 1050))],
        "sampler": {
            "env_variant": "hard", "max_steps": 1800, "replan_freq": 15,
            "guidance_w": 1.2, "sample_N": 50, "warm_start": False,
            "video": False,
        },
        "success_criterion": "SingleArmPlaceWipeMPCPlayer task_success",
    },
    "twoarm_native_source200": {
        "task_family": "two_arm_placewipe",
        "panel_type": "source_retention",
        "buckets": [
            _bucket("place_return", range(2), range(1000, 1050)),
            _bucket("wipe", range(2), range(1000, 1050)),
        ],
        "sampler": {
            "env_variant": "hard", "max_steps": 1800, "replan_freq": 15,
            "guidance_w": 1.2, "sample_N": 50, "warm_start": False,
            "video": False,
        },
        "success_criterion": "SingleArmPlaceWipeMPCPlayer task_success",
    },
    "threearm_full200": {
        "task_family": "three_arm_wipe",
        "panel_type": "full_task",
        "buckets": [_bucket("ThreeArmWipe", range(4), range(3000, 3050))],
        "sampler": {
            "max_steps": 3000, "replan_freq": 15, "guidance_w": 1.2,
            "sample_N": 50, "warm_start": True, "video": False,
        },
        "success_criterion": "ThreeArmWipe task_success",
    },
    "threearm_source400": {
        "task_family": "three_arm_wipe",
        "panel_type": "source_retention",
        "buckets": [
            _bucket("place_return_neg_y", range(2), range(1050, 1100)),
            _bucket("place_return_pos_y", range(2), range(1050, 1100)),
            _bucket("wipe", range(2), range(1050, 1100)),
            _bucket("tray_drag", range(2), range(3050, 3100)),
        ],
        "sampler": {
            "replan_freq": 15, "guidance_w": 1.2, "sample_N": 50,
            "warm_start": True, "video": False,
        },
        "success_criterion": "native source-task acceptance criterion",
    },
}


def expected_keys(protocol: str) -> set[tuple[str, int, int]]:
    """Return the exact, repeat-free ``(task, mode, seed)`` key matrix."""

    try:
        buckets = PROTOCOLS[protocol]["buckets"]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation protocol: {protocol}") from exc
    return {
        (str(bucket["task"]), int(mode), int(seed))
        for bucket in buckets
        for mode in bucket["modes"]
        for seed in bucket["seeds"]
    }


def expected_episode_count(protocol: str) -> int:
    return len(expected_keys(protocol))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_once(path: str | Path, value: dict[str, Any]) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, target)
    return target


def _parse_seed_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(item.strip()) for item in value.replace(" ", ",").split(",") if item.strip()]
    return [int(item) for item in value]


def sampling_seed(*, base: int, seed: int, mode: int, repeat: int = 0) -> int:
    return int(base) + 1000003 * int(seed) + 10007 * int(mode) + 101 * int(repeat)


def _artifact_paths(
    *, method: str, checkpoint: str | Path, stats: str | Path,
    base_model: str | Path | None, base_stats: str | Path | None,
) -> dict[str, str]:
    if method not in {"base", "coord", "fs"}:
        raise ValueError("method must be 'base', 'coord', or 'fs'")
    artifacts = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "stats": str(Path(stats).resolve()),
    }
    if method == "coord":
        if base_model is None or base_stats is None:
            raise ValueError("Coord requires frozen --base-model and --base-stats artifacts")
        artifacts.update({
            "base_model": str(Path(base_model).resolve()),
            "base_stats": str(Path(base_stats).resolve()),
        })
    elif base_model is not None or base_stats is not None:
        raise ValueError("base/from-scratch evaluations must not record frozen-base artifacts separately")
    missing = [path for path in artifacts.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"evaluation artifact(s) missing: {missing}")
    return artifacts


def build_contract(
    *, protocol: str, method: str, checkpoint: str | Path, stats: str | Path,
    base_model: str | Path | None = None, base_stats: str | Path | None = None,
    training_seeds: str | Iterable[int] = (), sampling_seed_base: int = 20260908,
    sampler_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic contract that is safe for byte-for-byte resume."""

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown evaluation protocol: {protocol}")
    spec = PROTOCOLS[protocol]
    artifacts = _artifact_paths(
        method=method, checkpoint=checkpoint, stats=stats,
        base_model=base_model, base_stats=base_stats,
    )
    train_seeds = _parse_seed_list(training_seeds)
    eval_seeds = sorted({seed for _, _, seed in expected_keys(protocol)})
    overlap = sorted(set(train_seeds).intersection(eval_seeds))
    if overlap:
        raise ValueError(f"evaluation seeds overlap declared training seeds: {overlap}")
    sampler = dict(spec["sampler"])
    if sampler_overrides:
        sampler.update(sampler_overrides)
    return {
        "schema": CONTRACT_SCHEMA,
        "protocol": protocol,
        "task_family": spec["task_family"],
        "panel_type": spec["panel_type"],
        "buckets": spec["buckets"],
        "seed_list": eval_seeds,
        "expected_episode_count": expected_episode_count(protocol),
        "repeat_count": 1,
        "method": method,
        "model_route": {
            "base": "selected_frozen_base_checkpoint",
            "coord": "frozen_base_plus_selected_coordination_head",
            "fs": "selected_from_scratch_checkpoint",
        }[method],
        "sampler": sampler,
        "success_criterion": spec["success_criterion"],
        "training_seed_disjointness": {
            "declared_training_seeds": train_seeds,
            "evaluation_seeds": eval_seeds,
            "disjoint": True,
        },
        "sampling_seed": {"base": int(sampling_seed_base), "derivation": SAMPLING_SEED_RULE},
        "artifacts": artifacts,
        "artifact_sha256": {label: sha256_file(path) for label, path in artifacts.items()},
    }


def initialize_output(root: str | Path, contract: dict[str, Any], *, resume: bool = False) -> Path:
    """Seal a new contract or permit only byte-identical interrupted resumes."""

    root_path = Path(root).resolve()
    contract_path = root_path / "evaluation_contract.json"
    desired = _json_bytes(contract)
    if resume:
        if not root_path.is_dir() or not contract_path.is_file():
            raise ValueError("--resume requires an existing sealed evaluation contract")
        if contract_path.read_bytes() != desired:
            raise ValueError("resume contract is not byte-for-byte compatible")
        return root_path
    if root_path.exists():
        raise FileExistsError(f"refusing pre-existing evaluation output root: {root_path}")
    root_path.mkdir(parents=True)
    write_json_once(contract_path, contract)
    return root_path


def _read_contract(root: Path) -> dict[str, Any]:
    path = root / "evaluation_contract.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"invalid sealed contract: {path}")
    return value


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["task"]), int(row["mode"]), int(row["seed"]))


def append_rollout_record(root: str | Path, row: dict[str, Any]) -> None:
    """Append exactly one prevalidated rollout record to an interrupted panel."""

    root_path = Path(root).resolve()
    contract = _read_contract(root_path)
    protocol = str(contract["protocol"])
    expected = expected_keys(protocol)
    record = dict(row)
    key = _key(record)
    if key not in expected:
        raise ValueError(f"rollout key is outside the sealed matrix: {key}")
    if not isinstance(record.get("success"), bool):
        raise ValueError("rollout success must be a Boolean")
    if "error" not in record:
        raise ValueError("rollout record must include an error field")
    if "steps" not in record or "wall_time_seconds" not in record or "geometry" not in record:
        raise ValueError("rollout record lacks steps, timing, or geometry provenance")
    if record.get("geometry") is None:
        raise ValueError("rollout geometry provenance is missing")
    sampling = contract["sampling_seed"]
    record["sampling_seed"] = sampling_seed(
        base=int(sampling["base"]), seed=key[2], mode=key[1], repeat=int(record.get("repeat", 0)),
    )
    path = root_path / "rollouts.jsonl"
    existing: set[tuple[str, int, int]] = set()
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                previous = json.loads(line)
                existing.add(_key(previous))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid existing rollout record at {path}:{line_number}") from exc
    if key in existing:
        raise ValueError(f"duplicate sealed rollout key: {key}")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rollouts(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root).resolve() / "rollouts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"sealed rollout ledger is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON rollout record at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"rollout record is not an object at {path}:{line_number}")
        rows.append(value)
    return rows


def validate_rollouts(contract: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Fail closed on coverage, artifact, type, error, or provenance drift."""

    protocol = str(contract.get("protocol"))
    expected = expected_keys(protocol)
    if int(contract.get("expected_episode_count", -1)) != len(expected):
        raise ValueError("contract expected episode count does not match its matrix")
    actual: set[tuple[str, int, int]] = set()
    for index, row in enumerate(rows):
        try:
            key = _key(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed rollout key at row {index}") from exc
        if key in actual:
            raise ValueError(f"duplicate rollout key: {key}")
        actual.add(key)
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"rollout success is not Boolean: {key}")
        if row.get("error") is not None:
            raise ValueError(f"rollout has evaluator error: {key}")
        if "steps" not in row or "wall_time_seconds" not in row or row.get("geometry") is None:
            raise ValueError(f"rollout lacks required provenance fields: {key}")
        expected_sampling_seed = sampling_seed(
            base=int(contract["sampling_seed"]["base"]), seed=key[2], mode=key[1],
            repeat=int(row.get("repeat", 0)),
        )
        if row.get("sampling_seed") != expected_sampling_seed:
            raise ValueError(f"rollout sampling seed differs from contract: {key}")
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            "sealed coverage mismatch: "
            f"rows={len(rows)} missing={len(expected - actual)} duplicate={len(rows) - len(actual)}"
        )
    artifacts = contract.get("artifacts")
    hashes = contract.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not isinstance(hashes, dict):
        raise ValueError("contract artifacts or hashes are missing")
    actual_hashes = {label: sha256_file(path) for label, path in artifacts.items()}
    if actual_hashes != hashes:
        raise ValueError("artifact hashes no longer match the sealed contract")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), int(row["mode"]))].append(row)
    by_task_mode = {}
    for (task, mode), values in sorted(grouped.items()):
        successes = sum(row["success"] is True for row in values)
        by_task_mode[f"{task}:mode{mode}"] = {
            "success": successes, "total": len(values), "success_rate": successes / len(values),
        }
    total_success = sum(row["success"] is True for row in rows)
    return {
        "overall": {"success": total_success, "total": len(rows), "success_rate": total_success / len(rows)},
        "by_task_mode": by_task_mode,
    }


def finalize(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    contract = _read_contract(root_path)
    rows = read_rollouts(root_path)
    validate_rollouts(contract, rows)
    summary = summarize(rows)
    ledger_path = write_json_once(root_path / "episode_ledger.json", {
        "schema": LEDGER_SCHEMA, "protocol": contract["protocol"], "episodes": rows, "summary": summary,
    })
    result_path = write_json_once(root_path / "evaluation.json", {
        "schema": RESULT_SCHEMA, "complete": True, "protocol": contract["protocol"],
        "task_family": contract["task_family"], "panel_type": contract["panel_type"],
        "method": contract["method"], "expected_rollouts": len(rows), "observed_rollouts": len(rows),
        "missing_rollouts": 0, "duplicate_rollouts": 0,
        "contract_path": str((root_path / "evaluation_contract.json").resolve()),
        "rollouts_path": str((root_path / "rollouts.jsonl").resolve()),
        "ledger_path": str(ledger_path.resolve()), "summary": summary,
    })
    return result_path


def _cli_init(args: argparse.Namespace) -> int:
    contract = build_contract(
        protocol=args.protocol, method=args.method, checkpoint=args.checkpoint, stats=args.stats,
        base_model=args.base_model, base_stats=args.base_stats, training_seeds=args.training_seeds,
        sampling_seed_base=args.sampling_seed_base,
    )
    root = initialize_output(args.output_root, contract, resume=args.resume)
    print(json.dumps({"output_root": str(root), "protocol": args.protocol,
                      "expected_episode_count": contract["expected_episode_count"],
                      "buckets": contract["buckets"]}, indent=2, sort_keys=True))
    return 0


def _cli_finalize(args: argparse.Namespace) -> int:
    result = finalize(args.output_root)
    print(result)
    return 0


def _cli_validate(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    contract = _read_contract(root)
    validate_rollouts(contract, read_rollouts(root))
    result = root / "evaluation.json"
    if not result.is_file():
        raise FileNotFoundError(f"sealed final result is missing: {result}")
    print(result)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    init.add_argument("--method", choices=("coord", "fs"), required=True)
    init.add_argument("--checkpoint", required=True)
    init.add_argument("--stats", required=True)
    init.add_argument("--base-model")
    init.add_argument("--base-stats")
    init.add_argument("--training-seeds", default="0")
    init.add_argument("--sampling-seed-base", type=int, default=20260908)
    init.add_argument("--output-root", required=True)
    init.add_argument("--resume", action="store_true")
    init.set_defaults(func=_cli_init)
    for name, func in (("finalize", _cli_finalize), ("validate", _cli_validate)):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True)
        command.set_defaults(func=func)
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

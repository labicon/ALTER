#!/usr/bin/env python3
"""Attest an already-completed evaluation that used a sealed stats-file copy.

This is append-only: it never rewrites an evaluation result, its episode
ledger, or either original contract. It records the evidence that evaluation
used a byte-identical checkpoint-local copy of the statistics named by a
legacy training contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_contract import sha256_file, validate_eval_contract


SCHEMA = "content_equivalent_stats_relocation_attestation.v1"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _context_from_result(result: dict) -> dict:
    rows = result.get("per_seed")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation result is missing per-episode rows")
    seeds = sorted({int(row["seed"]) for row in rows if isinstance(row, dict) and "seed" in row})
    modes = sorted({int(row["mode"]) for row in rows if isinstance(row, dict) and "mode" in row})
    if len(rows) != int(result.get("num_runs", -1)):
        raise ValueError("evaluation result num_runs disagrees with its episode rows")
    return {
        "task": result.get("task"),
        "env_variant": result.get("env_variant"),
        "env_class": result.get("env_class"),
        "seeds": seeds,
        "repeats": result.get("repeats"),
        "modes": modes,
        "max_steps": result.get("max_steps"),
        "replan_freq": result.get("replan_freq"),
        "selector_mode": result.get("selector_mode"),
        "sponge_scale": result.get("sponge_scale"),
        "video_policy": "disabled",
        "checkpoint_path": result.get("checkpoint_path"),
        "adapter_stats_path": result.get("adapter_stats_path"),
        "base_model_path": result.get("base_model_path"),
        "base_stats_path": result.get("base_stats_path"),
        "frame_offsets": result.get("frame_offsets"),
    }


def _assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from the sealed evaluation contract")


def attest(
    *, result_path: Path, evaluation_contract_path: Path, training_contract_path: Path, output_path: Path
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite attestation: {output_path}")
    result = _read_json(result_path)
    evaluation_contract = _read_json(evaluation_contract_path)
    training_contract = _read_json(training_contract_path)
    artifacts = evaluation_contract.get("artifacts")
    artifact_hashes = evaluation_contract.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not isinstance(artifact_hashes, dict):
        raise ValueError("evaluation contract is missing sealed artifacts or hashes")

    _assert_equal("result path", str(result_path.resolve()), evaluation_contract.get("result_path"))
    _assert_equal("result hash", sha256_file(result_path), evaluation_contract.get("result_sha256"))
    ledger_path = Path(str(evaluation_contract.get("episode_ledger_path", ""))).resolve()
    if not ledger_path.is_file():
        raise FileNotFoundError(f"missing sealed episode ledger: {ledger_path}")
    _assert_equal("episode ledger hash", sha256_file(ledger_path), evaluation_contract.get("episode_ledger_sha256"))

    for result_key, contract_key in (
        ("checkpoint_path", "checkpoint_path"),
        ("adapter_stats_path", "stats_path"),
        ("base_model_path", "base_model_path"),
        ("base_stats_path", "base_stats_path"),
    ):
        _assert_equal(result_key, str(Path(str(result[result_key])).resolve()), artifacts.get(contract_key))
    live_hashes = {
        "checkpoint_sha256": sha256_file(artifacts["checkpoint_path"]),
        "stats_sha256": sha256_file(artifacts["stats_path"]),
        "base_model_sha256": sha256_file(artifacts["base_model_path"]),
        "base_stats_sha256": sha256_file(artifacts["base_stats_path"]),
    }
    _assert_equal("live artifact hashes", live_hashes, artifact_hashes)
    _assert_equal("result artifact hashes", result.get("artifact_sha256"), artifact_hashes)

    original_stats_path = training_contract.get("artifacts", {}).get("stats_path")
    original_stats_hash = training_contract.get("artifacts", {}).get("stats_sha256")
    evaluated_stats_path = artifacts["stats_path"]
    evaluated_stats_hash = artifact_hashes["stats_sha256"]
    if not original_stats_path or not original_stats_hash:
        raise ValueError("training contract does not seal a stats artifact")
    if original_stats_path == evaluated_stats_path:
        raise ValueError("attestation is only for a relocated stats artifact")
    _assert_equal("training-vs-evaluation stats hash", evaluated_stats_hash, original_stats_hash)
    _assert_equal("current original stats hash", sha256_file(original_stats_path), original_stats_hash)

    mismatches = validate_eval_contract(training_contract, _context_from_result(result))
    if mismatches:
        raise ValueError(f"training contract still has substantive mismatches: {mismatches}")

    return {
        "schema": SCHEMA,
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject": {
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "evaluation_contract_path": str(evaluation_contract_path.resolve()),
            "evaluation_contract_sha256": sha256_file(evaluation_contract_path),
            "training_contract_path": str(training_contract_path.resolve()),
            "training_contract_sha256": sha256_file(training_contract_path),
        },
        "relocation": {
            "artifact": "adapter_stats",
            "training_contract_path": str(Path(str(original_stats_path)).resolve()),
            "evaluation_path": str(Path(str(evaluated_stats_path)).resolve()),
            "sha256": original_stats_hash,
            "content_equivalent": True,
            "path_changed": True,
            "reason": "checkpoint-local regular-file copy replaces a mutable shared stats path",
        },
        "validation": {
            "training_contract_mismatches": mismatches,
            "evaluation_contract_artifact_hashes": artifact_hashes,
            "episodes": result["num_runs"],
            "success_by_mode": result.get("success_by_mode"),
            "task_success": result.get("aggregate", {}).get("task_success"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--evaluation-contract-path", required=True, type=Path)
    parser.add_argument("--training-contract-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_path.resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"attestation directory is missing: {output.parent}")
    attestation = attest(
        result_path=args.result_path.resolve(),
        evaluation_contract_path=args.evaluation_contract_path.resolve(),
        training_contract_path=args.training_contract_path.resolve(),
        output_path=output,
    )
    output.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "attestation": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

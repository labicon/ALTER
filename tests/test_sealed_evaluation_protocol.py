from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sealed_evaluation_protocol import (
    build_contract,
    expected_episode_count,
    expected_keys,
    finalize,
    initialize_output,
    sampling_seed,
    validate_rollouts,
)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    paths = tuple(tmp_path / name for name in ("head.pt", "stats.pkl", "base.pt", "base_stats.pkl"))
    for path in paths:
        path.write_bytes(path.name.encode("utf-8"))
    return paths


def _contract(tmp_path: Path, protocol: str = "twoarm_source300") -> dict:
    checkpoint, stats, base_model, base_stats = _artifacts(tmp_path)
    return build_contract(
        protocol=protocol, method="coord", checkpoint=checkpoint, stats=stats,
        base_model=base_model, base_stats=base_stats, training_seeds="0,1,2",
    )


def _rows(contract: dict) -> list[dict]:
    base = int(contract["sampling_seed"]["base"])
    return [
        {
            "task": task, "mode": mode, "seed": seed, "repeat": 0,
            "success": True, "error": None, "steps": 1, "wall_time_seconds": 0.01,
            "geometry": {"seed": seed, "mode": mode},
            "sampling_seed": sampling_seed(base=base, seed=seed, mode=mode),
        }
        for task, mode, seed in sorted(expected_keys(contract["protocol"]))
    ]


def test_fixed_matrices_have_the_authoritative_cardinalities():
    assert expected_episode_count("twoarm_full200") == 200
    assert expected_episode_count("twoarm_source300") == 300
    assert expected_episode_count("twoarm_native_source200") == 200
    assert expected_episode_count("threearm_full200") == 200
    assert expected_episode_count("threearm_source400") == 400


def test_source_matrix_is_four_place_return_plus_two_wipe_buckets():
    keys = expected_keys("twoarm_source300")
    assert len(keys) == 300
    assert sum(task == "place_return" for task, _, _ in keys) == 200
    assert sum(task == "wipe" for task, _, _ in keys) == 100
    assert {mode for task, mode, _ in keys if task == "place_return"} == {0, 1, 2, 3}
    assert {mode for task, mode, _ in keys if task == "wipe"} == {0, 1}


def test_native_source_matrix_matches_the_two_trained_singlearm_tasks():
    keys = expected_keys("twoarm_native_source200")
    assert len(keys) == 200
    assert sum(task == "place_return" for task, _, _ in keys) == 100
    assert sum(task == "wipe" for task, _, _ in keys) == 100
    assert {mode for task, mode, _ in keys if task == "place_return"} == {0, 1}
    assert {mode for task, mode, _ in keys if task == "wipe"} == {0, 1}


def test_coverage_rejects_wrong_mode_missing_duplicate_and_evaluator_error(tmp_path):
    contract = _contract(tmp_path)
    rows = _rows(contract)
    validate_rollouts(contract, rows)

    wrong_mode = [dict(row) for row in rows]
    wrong_mode[0]["mode"] = 9
    wrong_mode[0]["sampling_seed"] = sampling_seed(base=int(contract["sampling_seed"]["base"]), seed=wrong_mode[0]["seed"], mode=9)
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_rollouts(contract, wrong_mode)

    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_rollouts(contract, rows[:-1])

    with pytest.raises(ValueError, match="duplicate rollout key"):
        validate_rollouts(contract, [*rows, dict(rows[0])])

    errored = [dict(row) for row in rows]
    errored[0]["error"] = "renderer failed"
    with pytest.raises(ValueError, match="evaluator error"):
        validate_rollouts(contract, errored)


def test_validation_rejects_stale_artifact_hash(tmp_path):
    contract = _contract(tmp_path)
    rows = _rows(contract)
    Path(contract["artifacts"]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hashes"):
        validate_rollouts(contract, rows)


def test_output_root_and_resume_are_contract_sealed(tmp_path):
    contract = _contract(tmp_path)
    root = tmp_path / "panel"
    initialize_output(root, contract)
    with pytest.raises(FileExistsError, match="pre-existing"):
        initialize_output(root, contract)
    assert initialize_output(root, contract, resume=True) == root.resolve()

    changed = json.loads(json.dumps(contract))
    changed["sampling_seed"]["base"] += 1
    with pytest.raises(ValueError, match="byte-for-byte"):
        initialize_output(root, changed, resume=True)


def test_finalize_writes_separate_overall_and_per_mode_reporting(tmp_path):
    contract = _contract(tmp_path)
    root = initialize_output(tmp_path / "complete", contract)
    rollout_path = root / "rollouts.jsonl"
    rollout_path.write_text("".join(json.dumps(row) + "\n" for row in _rows(contract)), encoding="utf-8")
    result_path = finalize(root)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["expected_rollouts"] == 300
    assert result["summary"]["overall"]["total"] == 300
    assert len(result["summary"]["by_task_mode"]) == 6


def test_from_scratch_contract_has_no_frozen_base_artifacts(tmp_path):
    checkpoint, stats, base_model, base_stats = _artifacts(tmp_path)
    contract = build_contract(
        protocol="twoarm_full200", method="fs", checkpoint=checkpoint, stats=stats,
        training_seeds="0",
    )
    assert set(contract["artifacts"]) == {"checkpoint", "stats"}
    with pytest.raises(ValueError, match="must not record frozen-base"):
        build_contract(
            protocol="twoarm_full200", method="fs", checkpoint=checkpoint, stats=stats,
            base_model=base_model, base_stats=base_stats, training_seeds="0",
        )


def test_base_contract_identifies_frozen_base_route(tmp_path):
    checkpoint, stats, _, _ = _artifacts(tmp_path)
    contract = build_contract(
        protocol="twoarm_native_source200", method="base",
        checkpoint=checkpoint, stats=stats, training_seeds="0",
    )
    assert contract["method"] == "base"
    assert contract["model_route"] == "selected_frozen_base_checkpoint"
    assert set(contract["artifacts"]) == {"checkpoint", "stats"}

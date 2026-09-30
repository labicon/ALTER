import argparse
import json
import pickle
import sys
from pathlib import Path

import pytest

from envs.arm.train import _train_utils
from scripts import queue_twoarm_forwardonly_capacity_ablation as queue
from scripts.experiment_contract import load_capacity_ablation_job_contract


def _rollouts(root: Path, modes: range) -> list[str]:
    root.mkdir()
    paths = []
    for mode in modes:
        for seed in range(5):
            path = root / f"demo_seed{seed}_mode{mode}.pkl"
            path.touch()
            paths.append(str(path.resolve()))
    return paths


def _inputs(tmp_path: Path) -> dict:
    twoarm = tmp_path / "twoarm"
    place = tmp_path / "place"
    wipe = tmp_path / "wipe"
    twoarm_files = _rollouts(twoarm, range(4))
    place_files = _rollouts(place, range(2))
    wipe_files = _rollouts(wipe, range(2))
    manifest = {
        "schema": "placewipe_forwardonly_all_expert_exact_manifest.v1",
        "direction": "forward_only",
        "selection_seed": 0,
        "per_mode_demo_budget": 5,
        "twoarm_per_mode_demo_budget": 5,
        "singlearm_per_mode_demo_budget": 5,
        "twoarm": {"files_by_mode": {
            str(mode): [path for path in twoarm_files if path.endswith(f"mode{mode}.pkl")]
            for mode in range(4)
        }},
        "singlearm": {
            "place_return": {"files_by_mode": {
                str(mode): [path for path in place_files if path.endswith(f"mode{mode}.pkl")]
                for mode in range(2)
            }},
            "wipe": {"files_by_mode": {
                str(mode): [path for path in wipe_files if path.endswith(f"mode{mode}.pkl")]
                for mode in range(2)
            }},
        },
    }
    manifest_path = tmp_path / "exact_5permode.json"
    manifest_path.write_text(json.dumps(manifest))
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"frozen-base")
    base_stats = tmp_path / "base_stats.pkl"
    base_stats.write_bytes(pickle.dumps({
        "backbone": "resnet18",
        "num_cameras": 1,
        "frame_offsets": [0],
        "d_model": 256,
        "n_heads": 4,
        "depth": 3,
        "dim_feedforward": 1024,
        "horizon": 20,
        "sigma_data": 1.0,
    }))
    return {
        "state_root": str(tmp_path / "queue_state"),
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "manifest": str(manifest_path),
        "base_model": str(base_model),
        "base_stats": str(base_stats),
        "twoarm_dir": str(twoarm),
        "singlearm_dirs": [str(place), str(wipe)],
    }


def _counts() -> dict:
    coord = {}
    fs = {}
    for i, label in enumerate(queue.PRESETS):
        target = 10_000 + i
        coord[label] = {
            "frozen_base_F": 9_000,
            "trainable_head_F": 1_000 + i,
            "deployed_F": target,
        }
        fs[label] = {
            "d_model": 64,
            "n_heads": 4,
            "depth": 3,
            "dim_feedforward": 128,
            "f_params": target,
            "target_f_params": target,
            "delta_f_params": 0,
            "relative_delta": 0.0,
        }
    return {"coord": coord, "fs": fs}


def test_presets_and_fs_calibration_are_deterministic_and_tight() -> None:
    assert queue.PRESETS == {
        "L": (256, 4, 1024),
        "M": (192, 3, 768),
        "S": (128, 3, 512),
        "XS": (64, 2, 256),
    }
    first = queue.select_fs_architecture(
        1000,
        lambda d_model, dim_feedforward: 1000 + (d_model - 64) + (dim_feedforward - 128),
        d_models=range(64, 65, 4),
        ffns=range(128, 129, 4),
    )
    second = queue.select_fs_architecture(
        1000,
        lambda d_model, dim_feedforward: 1000 + (d_model - 64) + (dim_feedforward - 128),
        d_models=range(64, 65, 4),
        ffns=range(128, 129, 4),
    )
    assert first == second
    assert first["n_heads"] == 4
    assert first["depth"] == 3
    assert first["relative_delta"] <= 0.001
    with pytest.raises(ValueError, match="0.1"):
        queue.select_fs_architecture(
            1000,
            lambda _d_model, _dim_feedforward: 1002,
            d_models=range(64, 65, 4),
            ffns=range(128, 129, 4),
        )


def test_initialize_seals_unique_twoarm_forwardonly_specs(tmp_path: Path, monkeypatch) -> None:
    config = _inputs(tmp_path)
    monkeypatch.setattr(queue, "capacities", lambda _path: _counts())
    queue.initialize(config)
    root = Path(config["state_root"])
    experiment = queue.read(root / "specs" / "experiment.json")
    assert experiment["schema"] == queue.SCHEMA
    assert experiment["manifest"]["sha256"] == queue.sha256(config["manifest"])
    assert experiment["manifest"]["twoarm_train_file_count"] == 20
    assert experiment["manifest"]["singlearm_train_file_count"] == 20
    assert not (root / "specs" / "pairs").exists()

    specs = [queue.read(path) for path in sorted((root / "specs" / "jobs").glob("*.json"))]
    assert {spec["job_id"] for spec in specs} == {
        f"capacity-{label.lower()}-{method}"
        for label in queue.PRESETS for method in ("coord", "fs")
    }
    assert len({spec["output_dir"] for spec in specs}) == 8
    assert all(spec["command_sha256"] == queue.canonical_hash(spec["training_argv"]) for spec in specs)
    assert all(spec["training_argv"].count("--num-arms") == 1 for spec in specs)
    assert all(
        spec["training_argv"][spec["training_argv"].index("--num-arms") + 1] == "2"
        for spec in specs
    )
    assert all("--checkpoint-every-steps" in spec["training_argv"] for spec in specs)
    assert all(
        spec["training_argv"][spec["training_argv"].index("--checkpoint-every-steps") + 1] == "100000"
        for spec in specs
    )

    xs_coord = next(spec for spec in specs if spec["job_id"] == queue.XS_COORD_JOB)
    assert xs_coord["capacity_preset"] == {
        "d_model": 64,
        "depth": 2,
        "dim_feedforward": 256,
    }
    assert "--use-side-net" in xs_coord["training_argv"]
    assert "--side-net-fusion" in xs_coord["training_argv"]
    assert "cross_attn" in xs_coord["training_argv"]
    assert "--no-augment" not in xs_coord["training_argv"]
    assert all("--no-augment" in spec["training_argv"] for spec in specs if spec["method"] == "fs")
    assert all("--time" + "ing" not in " ".join(spec["training_argv"]) for spec in specs)

    queue_state = queue.read(queue.queue_state_file(root))
    assert set(queue_state) == {"schema", "history"}


def test_manifest_identity_rejects_nonexact_or_nonforwardonly_inputs(tmp_path: Path) -> None:
    config = _inputs(tmp_path)
    raw = json.loads(Path(config["manifest"]).read_text())
    raw["twoarm_per_mode_demo_budget"] = 10
    Path(config["manifest"]).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exact 5/mode"):
        queue.validate_forwardonly_inputs(config)



def test_failed_training_is_retryable_when_sharing_a_gpu(tmp_path: Path, monkeypatch) -> None:
    config = _inputs(tmp_path)
    monkeypatch.setattr(queue, "capacities", lambda _path: _counts())
    queue.initialize(config)
    root = Path(config["state_root"])
    monkeypatch.setattr(queue, "gpu", lambda _index: {"uuid": "GPU-4"})
    monkeypatch.setattr(queue, "processes", lambda _index: "1234\n")
    monkeypatch.setattr(queue, "snapshot", lambda *_args: {"gpu": {"uuid": "GPU-4"}})
    monkeypatch.setattr(queue, "run_process", lambda *_args: 1)

    queue.run_training(root, queue.XS_COORD_JOB, 4, "python")

    state = queue.read(queue.state_file(root, queue.XS_COORD_JOB))
    assert state["status"] == "PENDING_TRAIN"
    assert state["training_gpu_uuid"] == "GPU-4"
    assert state["attempts"] == 1
    assert state["retryable_failure"] == "training_returncode_1"
    assert not (root / "state" / "evaluation_locks" / "GPU-4.lock").exists()


def test_evaluation_lock_and_evaluation_separation(tmp_path: Path) -> None:
    lock = queue.evaluation_gpu_lock(tmp_path, "GPU-A")
    with pytest.raises(RuntimeError, match="already owns"):
        queue.evaluation_gpu_lock(tmp_path, "GPU-A")
    lock.unlink()
    queue.assert_evaluation_gpu("GPU-A", "GPU-B")
    with pytest.raises(RuntimeError, match="differ"):
        queue.assert_evaluation_gpu("GPU-A", "GPU-A")


def test_headline_full_panel_uses_the_established_1800_step_horizon(tmp_path: Path) -> None:
    spec = {
        "method": "fs",
        "output_dir": str(tmp_path / "checkpoints"),
        "training_argv": ["trainer", "--checkpoint-prefix", "capacity"],
    }
    full = queue.evaluation_command(spec, 100_000, tmp_path / "full", True)
    smoke = queue.evaluation_command(spec, 100_000, tmp_path / "smoke", False)

    assert full[full.index("--max-steps") + 1] == "1800"
    assert smoke[smoke.index("--max-steps") + 1] == "3000"


def test_training_contract_seals_command_and_spec_hash(tmp_path: Path, monkeypatch) -> None:
    trainer = tmp_path / "trainer.py"
    trainer.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base")
    base_stats = tmp_path / "base.pkl"
    base_stats.write_bytes(b"stats")
    spec_path = tmp_path / "job.json"
    argv = [str(trainer), "--example", "value"]
    spec = {
        "schema": queue.SCHEMA,
        "job_id": "capacity-xs-fs",
        "method": "fs",
        "output_dir": str(tmp_path / "output"),
        "manifest": {"path": str(manifest), "sha256": queue.sha256(manifest)},
        "frozen_base": {
            "model_path": str(base_model),
            "base_model_sha256": queue.sha256(base_model),
            "stats_path": str(base_stats),
            "base_stats_sha256": queue.sha256(base_stats),
        },
        "model_counts": {"f_params": 100},
        "training_argv": argv,
        "command_sha256": queue.canonical_hash(argv),
    }
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", [str(trainer), "--example", "value"])
    sealed = load_capacity_ablation_job_contract(str(spec_path), method="fs")
    assert sealed is not None
    assert sealed["job_spec_path"] == str(spec_path.resolve())
    assert sealed["job_spec_sha256"] == queue.sha256(spec_path)

def test_no_removed_measurement_cli_or_queue_symbols_remain() -> None:
    parser = argparse.ArgumentParser()
    _train_utils.add_train_loop_args(parser)
    help_text = parser.format_help().lower()
    forbidden = "time" + "ing"
    assert forbidden not in help_text
    assert not hasattr(queue, "calibra" + "tion_command")
    assert not hasattr(queue, "claim_" + forbidden + "_gpu")
    assert not hasattr(queue, "pair_" + forbidden + "_status")

from __future__ import annotations

import json
import pickle
import random
from argparse import Namespace

import numpy as np
import pytest
import torch

from envs.arm.train import _train_utils
from scripts.backfill_experiment_contract import (
    build_backfill_contract,
    parse_args as parse_backfill_args,
    write_backfill_contract,
)
from scripts.experiment_contract import (
    build_training_contract,
    contract_path_for_checkpoint,
    discover_contract,
    validate_eval_contract,
    write_training_contract,
    sha256_file,
)


def _contract(**overrides):
    contract = {
        "contract": {"schema_version": "experiment_contract.v1", "contract_id": "abc123"},
        "artifacts": {
            "checkpoint_path": "/tmp/model.pt",
            "stats_path": "/tmp/stats.pkl",
            "base_model_path": "/tmp/base.pt",
            "base_stats_path": "/tmp/base_stats.pkl",
        },
        "model": {"task_family": "placewipe", "method": "coord", "num_arms": 2},
        "expected_eval": {
            "task": "placewipe",
            "env_variant": "hard",
            "env_class": "TwoArmPlaceWipeHard",
            "modes": [0, 1, 2, 3],
            "seeds": list(range(1000, 1050)),
            "repeats": 1,
            "max_steps": 1800,
            "replan_freq": 15,
            "selector_mode": "coord",
        "video_policy": "disabled",
        },
        "task_extension": {"placewipe": {"env_variant": "hard"}},
    }
    for key, value in overrides.items():
        contract[key] = value
    return contract


def _eval(**overrides):
    ctx = {
        "task": "placewipe",
        "env_variant": "hard",
        "env_class": "TwoArmPlaceWipeHard",
        "modes": [0, 1, 2, 3],
        "seeds": list(range(1000, 1050)),
        "repeats": 1,
        "max_steps": 1800,
        "replan_freq": 15,
        "selector_mode": "coord",
        "video_policy": "disabled",
        "checkpoint_path": "/tmp/model.pt",
        "adapter_stats_path": "/tmp/stats.pkl",
        "base_model_path": "/tmp/base.pt",
        "base_stats_path": "/tmp/base_stats.pkl",
    }
    ctx.update(overrides)
    return ctx


def test_hard_placewipe_contract_hard_eval_passes():
    assert validate_eval_contract(_contract(), _eval()) == []


def test_hard_placewipe_contract_standard_eval_records_mismatch():
    mismatches = validate_eval_contract(_contract(), _eval(env_variant="standard", env_class="TwoArmPlaceWipe"))
    assert "env_variant:hard!=standard" in mismatches
    assert "env_class:TwoArmPlaceWipeHard!=TwoArmPlaceWipe" in mismatches


def test_strict_policy_can_fail_on_known_mismatch():
    mismatches = validate_eval_contract(_contract(), _eval(env_variant="standard"))
    assert mismatches and mismatches != ["missing_contract"]


def test_intentional_cleanup_standard_still_records_mismatch():
    mismatches = validate_eval_contract(_contract(), _eval(env_variant="standard"))
    assert "env_variant:hard!=standard" in mismatches


def test_coord_eval_wrong_base_stats_mismatches():
    mismatches = validate_eval_contract(_contract(), _eval(base_stats_path="/tmp/other_base_stats.pkl"))
    assert "base_stats_path" in mismatches


def test_content_equivalent_relocated_adapter_stats_are_accepted(tmp_path):
    checkpoint = tmp_path / "model.pt"
    original_stats = tmp_path / "shared_stats.pkl"
    sealed_copy = tmp_path / "checkpoint_local_stats.pkl"
    checkpoint.write_bytes(b"checkpoint")
    original_stats.write_bytes(b"immutable stats")
    sealed_copy.write_bytes(original_stats.read_bytes())
    contract = build_training_contract(
        args=_real_training_args(),
        stats={"pipeline": "coord_placewipe", "rollout_dir": "hard_placewipe"},
        checkpoint_path=str(checkpoint),
        stats_path=str(original_stats),
        producer_script="test",
        repo_root=str(tmp_path),
    )

    relocated_context = _eval(
        checkpoint_path=str(checkpoint),
        adapter_stats_path=str(sealed_copy),
        base_model_path=None,
        base_stats_path=None,
    )
    assert validate_eval_contract(contract, relocated_context) == []

    sealed_copy.write_bytes(b"different stats")
    mismatches = validate_eval_contract(contract, relocated_context)
    assert "adapter_stats_path" in mismatches
    assert "stats_sha256" in mismatches


def test_missing_legacy_contract_warns_and_continues_signal():
    assert validate_eval_contract(None, _eval()) == ["missing_contract"]


def test_three_arm_wipe_task_extension_validates_sponge_scale():
    contract = _contract(
        model={"task_family": "wipe", "method": "fromscratch", "num_arms": 3},
        expected_eval={
            "task": "wipe",
            "modes": [0, 1, 2, 3],
            "max_steps": 1800,
            "replan_freq": 15,
            "sponge_scale": 0.85,
        },
        task_extension={"three_arm_wipe": {"num_arms": 3, "sponge_scale": 0.85}},
    )
    ctx = _eval(task="wipe", env_variant=None, env_class="ThreeArmWipe", sponge_scale=0.85)
    assert validate_eval_contract(contract, ctx) == []
    mismatches = validate_eval_contract(contract, {**ctx, "sponge_scale": 1.0})
    assert "sponge_scale:0.85!=1.0" in mismatches


def test_three_arm_exact_manifest_is_labeled_matched_density(tmp_path):
    checkpoint = tmp_path / "model.pt"
    stats_path = tmp_path / "stats.pkl"
    checkpoint.write_bytes(b"checkpoint")
    stats_path.write_bytes(b"stats")
    args = _real_training_args()
    args.num_arms = 3
    args.task_family = "wipe"
    contract = build_training_contract(
        args=args,
        stats={
            "pipeline": "threearm_coordination_head",
            "dataset_manifest": {
                "schema": "threearm_realign_v2_exact_manifest.v1",
                "per_mode_demo_budget": 5,
                "twoarm_train_files": [f"multi_mode{i % 4}_{i}.pkl" for i in range(20)],
                "singlearm_train_files": [f"single_mode{i % 4}_{i}.pkl" for i in range(50)],
                "twoarm_validation_files": [],
                "singlearm_validation_files": [],
            },
        },
        checkpoint_path=str(checkpoint),
        stats_path=str(stats_path),
        producer_script="test",
        repo_root=str(tmp_path),
    )
    assert contract["data"]["data_policy"] == "sa_matched_density"
    assert contract["data"]["twoarm_train_count"] == 20
    assert contract["data"]["singlearm_train_count"] == 50



def _real_training_args():
    return Namespace(
        checkpoint_prefix="coord_placewipe",
        expected_env_variant=None,
        task_family=None,
        task=None,
        num_arms=2,
        max_train_steps=None,
        epochs=None,
        checkpoint_every_steps=None,
        lr=None,
        batch_size=None,
        num_workers=None,
        prefetch_factor=None,
        data_subset_seed=None,
        training_seed=0,
        wandb_run_id=None,
    )


def test_checkpoint_specific_contract_detects_artifact_changes(tmp_path):
    checkpoint = tmp_path / "model.pt"
    stats_path = tmp_path / "stats.pkl"
    checkpoint.write_bytes(b"checkpoint-v1")
    stats_path.write_bytes(b"stats-v1")
    stats = {"pipeline": "coord_placewipe", "rollout_dir": "hard_placewipe"}

    contract_path = write_training_contract(
        args=_real_training_args(),
        stats=stats,
        checkpoint_path=str(checkpoint),
        stats_path=str(stats_path),
        producer_script="test",
        repo_root=str(tmp_path),
    )
    assert contract_path == str(contract_path_for_checkpoint(checkpoint))
    found_path, contract = discover_contract(
        checkpoint_path=str(checkpoint), stats_path=str(stats_path)
    )
    assert found_path == contract_path
    assert contract is not None

    context = _eval(
        checkpoint_path=str(checkpoint),
        adapter_stats_path=str(stats_path),
        base_model_path=None,
        base_stats_path=None,
    )
    assert validate_eval_contract(contract, context) == []

    other_checkpoint = tmp_path / "other.pt"
    other_checkpoint.write_bytes(b"other")
    assert discover_contract(checkpoint_path=str(other_checkpoint)) == (None, None)

    checkpoint.write_bytes(b"checkpoint-v2")
    assert "checkpoint_sha256" in validate_eval_contract(contract, context)


def test_contract_content_tampering_is_detected(tmp_path):
    checkpoint = tmp_path / "model.pt"
    stats_path = tmp_path / "stats.pkl"
    checkpoint.write_bytes(b"checkpoint")
    stats_path.write_bytes(b"stats")
    contract = build_training_contract(
        args=_real_training_args(),
        stats={"pipeline": "coord_placewipe", "rollout_dir": "hard_placewipe"},
        checkpoint_path=str(checkpoint),
        stats_path=str(stats_path),
        producer_script="test",
        repo_root=str(tmp_path),
    )
    contract["model"]["method"] = "tampered"
    assert "contract_content_sha256" in validate_eval_contract(
        contract,
        _eval(
            checkpoint_path=str(checkpoint),
            adapter_stats_path=str(stats_path),
            base_model_path=None,
            base_stats_path=None,
        ),
    )


def test_placewipe_contract_locks_default_rollout_protocol(tmp_path):
    checkpoint = tmp_path / "model.pt"
    stats_path = tmp_path / "stats.pkl"
    checkpoint.write_bytes(b"checkpoint")
    stats_path.write_bytes(b"stats")
    contract = build_training_contract(
        args=_real_training_args(),
        stats={"pipeline": "coord_placewipe", "rollout_dir": "hard_placewipe",
               "selected_rollout_files": ["mode0.pkl", "mode1.pkl", "mode2.pkl", "mode3.pkl"]},
        checkpoint_path=str(checkpoint), stats_path=str(stats_path),
        producer_script="test", repo_root=str(tmp_path),
    )
    expected = contract["expected_eval"]
    assert expected["env_class"] == "TwoArmPlaceWipeHard"
    assert expected["max_steps"] == 1800
    assert expected["replan_freq"] == 15
    assert expected["repeats"] == 1
    assert expected["seeds"] == list(range(1000, 1050))
    assert expected["modes"] == [0, 1, 2, 3]
    assert "max_steps:1800!=900" in validate_eval_contract(contract, _eval(max_steps=900))
    assert "replan_freq:15!=10" in validate_eval_contract(contract, _eval(replan_freq=10))
    assert "seeds:[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049]!=[1000]" in validate_eval_contract(contract, _eval(seeds=[1000]))
    assert "video_policy:disabled!=enabled" in validate_eval_contract(
        contract, _eval(video_policy="enabled")
    )


def test_diagnostic_profiles_allow_only_documented_deviations():
    base_context = _eval(selector_mode="base")
    assert validate_eval_contract(
        _contract(), base_context, profile="diagnostic_base_routing"
    ) == []
    assert "env_variant:hard!=standard" in validate_eval_contract(
        _contract(), {**base_context, "env_variant": "standard"},
        profile="diagnostic_base_routing",
    )

    selector_context = _eval(
        selector_mode="auto", seeds=[1000], max_steps=1400, video_policy="enabled"
    )
    assert validate_eval_contract(
        _contract(), selector_context, profile="diagnostic_selector_routing"
    ) == []
    assert "replan_freq:15!=10" in validate_eval_contract(
        _contract(), {**selector_context, "replan_freq": 10},
        profile="diagnostic_selector_routing",
    )

    video_context = _eval(max_steps=1400, video_policy="enabled", seeds=[0, 1])
    assert validate_eval_contract(
        _contract(), video_context, profile="diagnostic_short_video"
    ) == []
    assert "selector_mode:coord!=base" in validate_eval_contract(
        _contract(), {**video_context, "selector_mode": "base"},
        profile="diagnostic_short_video",
    )


def test_sha256_file_cache_invalidates_after_rewrite(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"first-content")
    first = sha256_file(artifact)
    assert sha256_file(artifact) == first

    artifact.write_bytes(b"second-content")
    second = sha256_file(artifact)
    assert second != first


def test_diagnostic_smoke_allows_only_reduced_seed_mode_and_steps():
    smoke = _eval(seeds=[1000], modes=[0], max_steps=2)
    assert validate_eval_contract(
        _contract(), smoke, profile="diagnostic_smoke"
    ) == []
    for field, value, expected_fragment in (
        ("replan_freq", 10, "replan_freq:15!=10"),
        ("selector_mode", "base", "selector_mode:coord!=base"),
        ("env_variant", "standard", "env_variant:hard!=standard"),
        ("video_policy", "enabled", "video_policy:disabled!=enabled"),
    ):
        mismatches = validate_eval_contract(
            _contract(), {**smoke, field: value}, profile="diagnostic_smoke"
        )
        assert expected_fragment in mismatches


def test_forensic_video_and_rf_profiles_lock_documented_fields():
    forensic = _eval(
        seeds=[1007], modes=[2], max_steps=1800, video_policy="enabled"
    )
    assert validate_eval_contract(
        _contract(), forensic, profile="diagnostic_forensic_video"
    ) == []
    assert "max_steps:1800!=1400" in validate_eval_contract(
        _contract(),
        {**forensic, "max_steps": 1400},
        profile="diagnostic_forensic_video",
    )
    assert "replan_freq:15!=10" in validate_eval_contract(
        _contract(),
        {**forensic, "replan_freq": 10},
        profile="diagnostic_forensic_video",
    )

    assert validate_eval_contract(
        _contract(), _eval(replan_freq=5), profile="diagnostic_rf5"
    ) == []
    assert validate_eval_contract(
        _contract(), _eval(replan_freq=10), profile="diagnostic_rf10"
    ) == []
    assert "replan_freq:5!=10" in validate_eval_contract(
        _contract(), _eval(replan_freq=10), profile="diagnostic_rf5"
    )


def test_training_seed_covers_process_and_dataloader_rngs():
    _train_utils.seed_training(17)
    first = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )
    _train_utils.seed_training(17)
    second = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )
    assert first == second

    generator_a = _train_utils.dataloader_seed_kwargs(17, stream=0)["generator"]
    generator_b = _train_utils.dataloader_seed_kwargs(17, stream=0)["generator"]
    generator_c = _train_utils.dataloader_seed_kwargs(17, stream=1)["generator"]
    assert torch.equal(
        torch.randperm(20, generator=generator_a),
        torch.randperm(20, generator=generator_b),
    )
    assert not torch.equal(
        torch.randperm(20, generator=_train_utils.dataloader_seed_kwargs(17, stream=0)["generator"]),
        torch.randperm(20, generator=generator_c),
    )


def test_contract_records_training_seed(tmp_path):
    checkpoint = tmp_path / "model.pt"
    stats_path = tmp_path / "stats.pkl"
    checkpoint.write_bytes(b"checkpoint")
    stats_path.write_bytes(b"stats")
    args = _real_training_args()
    args.training_seed = 23
    contract = build_training_contract(
        args=args,
        stats={"pipeline": "coord_placewipe", "rollout_dir": "hard_placewipe"},
        checkpoint_path=str(checkpoint),
        stats_path=str(stats_path),
        producer_script="test",
        repo_root=str(tmp_path),
    )
    assert contract["training"]["training_seed"] == 23


def _legacy_backfill_args(tmp_path):
    checkpoint = tmp_path / "coord_final.pt"
    stats_path = tmp_path / "coord_stats.pkl"
    base_model = tmp_path / "base.pt"
    base_stats = tmp_path / "base_stats.pkl"
    manifest_path = tmp_path / "coord_data_manifest.json"
    checkpoint.write_bytes(b"legacy-checkpoint")
    base_model.write_bytes(b"base-checkpoint")
    base_stats.write_bytes(b"base-stats")
    manifest = {
        "twoarm_train_files": [str(tmp_path / f"twoarm_mode{i}.pkl") for i in range(4)],
        "singlearm_train_files": [str(tmp_path / f"singlearm_mode{i}.pkl") for i in range(4)],
        "twoarm_validation_files": [],
        "singlearm_validation_files": [],
    }
    manifest_path.write_text(json.dumps(manifest))
    with stats_path.open("wb") as handle:
        pickle.dump(
            {
                "pipeline": "mixed_coord_placewipe",
                "rollout_dir": str(tmp_path / "hard_placewipe"),
                "base_model_path": str(base_model),
                "base_stats_path": str(base_stats),
                "dataset_manifest": manifest,
                "training_seed": 0,
            },
            handle,
        )
    argv = [
        "--checkpoint-path", str(checkpoint),
        "--stats-path", str(stats_path),
        "--manifest-path", str(manifest_path),
        "--task-family", "placewipe",
        "--method", "coord",
        "--env-variant", "hard",
    ]
    return parse_backfill_args(argv), checkpoint, stats_path, base_model, base_stats, manifest


def test_legacy_backfill_hashes_artifacts_and_preserves_exact_manifest(tmp_path):
    args, checkpoint, stats_path, base_model, base_stats, manifest = _legacy_backfill_args(tmp_path)
    contract = build_backfill_contract(args)
    assert contract["contract"]["provenance"] == "legacy_backfill"
    assert contract["provenance"]["source"] == "legacy_backfill"
    assert contract["data"]["source_manifest"] == manifest
    assert contract["data"]["twoarm_train_files"] == manifest["twoarm_train_files"]
    assert contract["artifacts"]["checkpoint_sha256"] == sha256_file(checkpoint)
    assert contract["artifacts"]["stats_sha256"] == sha256_file(stats_path)
    assert contract["artifacts"]["base_model_sha256"] == sha256_file(base_model)
    assert contract["artifacts"]["base_stats_sha256"] == sha256_file(base_stats)
    assert contract["expected_eval"]["env_variant"] == "hard"
    assert contract["expected_eval"]["replan_freq"] == 15
    assert contract["training"]["training_seed"] == 0

    output = write_backfill_contract(contract, checkpoint_path=str(checkpoint))
    assert output.is_file()
    with pytest.raises(FileExistsError):
        write_backfill_contract(contract, checkpoint_path=str(checkpoint))


def test_legacy_backfill_rejects_manifest_mismatch(tmp_path):
    args, *_ = _legacy_backfill_args(tmp_path)
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps({"twoarm_train_files": ["different.pkl"]}))
    args.manifest_path = str(mismatch)
    with pytest.raises(ValueError, match="differs"):
        build_backfill_contract(args)


def test_step_checkpoint_metrics_are_finite_before_log_cadence(tmp_path):
    args = Namespace(
        max_train_steps=2,
        log_every_steps=50,
        checkpoint_every_steps=1,
        val_every_steps=1,
        val_batches=0,
        val_noise_seed=42,
    )
    calls = iter([(1.25, 3.5), (0.75, 2.5)])

    def train_step(_batch):
        return next(calls)

    history = _train_utils.train_step_loop(
        args,
        dataloader=[object()],
        train_step_fn=train_step,
        save_fn=lambda _path: None,
        checkpoint_dir=str(tmp_path),
        prefix="smoke",
        log_fn=lambda *_args, **_kwargs: None,
        device="cpu",
    )
    checkpoints = [row for row in history if row["event"] == "checkpoint"]
    assert checkpoints[0]["step"] == 1
    assert checkpoints[0]["train_loss"] == pytest.approx(1.25)
    assert checkpoints[0]["train_grad_norm"] == pytest.approx(3.5)
    assert all(np.isfinite(row["train_loss"]) for row in checkpoints)
    assert all(np.isfinite(row["train_grad_norm"]) for row in checkpoints)

import math
import pickle

import numpy as np

from envs.arm.train.hierarchical_sampling import (
    hierarchical_sample_weights,
    make_hierarchical_sampler,
)
from envs.arm.train.train_combined_e2e_shoulder import CombinedE2EDataset
from envs.arm.train.train_singlearm_mixedfront_e2e_shoulder import (
    SingleArmShoulderE2EDataset,
    discover_rollout_files,
)
from envs.arm.train.twoarm_dataset import TwoArmE2EImageDataset
from envs.arm.train._train_utils import split_train_val_files


class _Dataset:
    def __init__(self):
        # Group A has two rollouts. Rollout A0 has two unequal-length agent
        # trajectories; rollout A1 has one. Group B has one rollout.
        self.all_actions = [
            np.zeros((2, 1)),
            np.zeros((4, 1)),
            np.zeros((3, 1)),
            np.zeros((5, 1)),
        ]
        self.trajectory_group_keys = ["A", "A", "A", "B"]
        self.trajectory_rollout_ids = ["A0", "A0", "A1", "B0"]
        self.index_map = [
            (trajectory, timestep)
            for trajectory, actions in enumerate(self.all_actions)
            for timestep in range(len(actions))
        ]

    def __len__(self):
        return len(self.index_map)


def _mass(dataset, weights, predicate):
    return sum(
        float(weight)
        for weight, (trajectory, _) in zip(weights, dataset.index_map)
        if predicate(trajectory)
    )


def test_hierarchical_weights_balance_groups_rollouts_agents_and_timesteps():
    dataset = _Dataset()
    weights, summary = hierarchical_sample_weights(dataset)

    assert math.isclose(float(weights.sum()), 1.0)
    assert math.isclose(_mass(dataset, weights, lambda trajectory: trajectory < 3), 0.5)
    assert math.isclose(_mass(dataset, weights, lambda trajectory: trajectory == 3), 0.5)
    assert math.isclose(_mass(dataset, weights, lambda trajectory: trajectory in (0, 1)), 0.25)
    assert math.isclose(_mass(dataset, weights, lambda trajectory: trajectory == 2), 0.25)
    assert math.isclose(_mass(dataset, weights, lambda trajectory: trajectory == 0), 0.125)
    assert math.isclose(_mass(dataset, weights, lambda trajectory: trajectory == 1), 0.125)
    assert summary["hierarchy"].endswith("timestep")
    assert summary["groups"]["A"]["rollout_count"] == 2
    assert math.isclose(
        summary["groups"]["A"]["rollouts"]["A0"]["target_probability"], 0.25
    )
    assert math.isclose(
        summary["groups"]["A"]["rollouts"]["A0"]["agent_trajectories"][0][
            "target_probability"
        ],
        0.125,
    )


def test_hierarchical_weights_follow_subset_order():
    dataset = _Dataset()
    subset = [0, 1, 6, 7, 8, 9]
    weights, summary = hierarchical_sample_weights(dataset, subset)

    assert len(weights) == len(subset)
    assert math.isclose(float(weights.sum()), 1.0)
    assert summary["sample_count_per_sampler_epoch"] == len(subset)


def test_hierarchical_sampler_is_reproducible():
    dataset = _Dataset()
    first, _ = make_hierarchical_sampler(dataset, seed=7, stream=3)
    second, _ = make_hierarchical_sampler(dataset, seed=7, stream=3)

    assert list(first) == list(second)


def test_zero_validation_holdout_keeps_every_rollout_for_training():
    files = [
        "place_return_neg_y/mode0/demo_seed0_mode0.pkl",
        "place_return_neg_y/mode0/demo_seed1_mode0.pkl",
        "place_return_neg_y/mode1/demo_seed0_mode1.pkl",
    ]

    train_files, validation_files = split_train_val_files(
        files, holdout_fraction=0.0, max_val_files=16, seed=123,
    )

    assert train_files == sorted(files)
    assert validation_files == []


def test_nested_multiarm_rollout_discovery_and_sampling_metadata(tmp_path):
    rollout_path = tmp_path / "mode0" / "demo_seed1_mode0.pkl"
    rollout_path.parent.mkdir()
    length = 3
    rollout = {"actions": np.zeros((length, 21), dtype=np.float32)}
    for agent_idx in range(3):
        rollout[f"camera_obs{agent_idx}"] = np.zeros(
            (length, 4, 4, 3), dtype=np.uint8
        )
        rollout[f"camera_obs_shoulder{agent_idx}"] = np.zeros(
            (length, 4, 4, 3), dtype=np.uint8
        )
    with rollout_path.open("wb") as handle:
        pickle.dump(rollout, handle)

    expected_path = str(rollout_path.resolve())
    assert CombinedE2EDataset._enumerate_pool([str(tmp_path)], modes=[0]) == [
        expected_path
    ]

    dataset = TwoArmE2EImageDataset(
        rollout_dir=str(tmp_path),
        base_action_mean=np.zeros(7, dtype=np.float32),
        base_action_std=np.ones(7, dtype=np.float32),
        horizon=2,
        augment=False,
        modes=[0],
        num_arms=3,
        rollout_files=[expected_path],
    )

    assert dataset.selected_pkl_files == [expected_path]
    assert dataset.trajectory_group_keys == ["multiarm/mode0"] * 3
    assert dataset.trajectory_rollout_ids == [expected_path] * 3
    weights, summary = hierarchical_sample_weights(dataset)
    assert math.isclose(float(weights.sum()), 1.0)
    assert summary["groups"]["multiarm/mode0"]["agent_trajectories_per_rollout_min"] == 3


def test_nested_singlearm_discovery_returns_task_mode_rollouts(tmp_path):
    task_root = tmp_path / "place_return_neg_y"
    rollout_path = task_root / "mode0" / "demo_seed1_mode0.pkl"
    rollout_path.parent.mkdir(parents=True)
    rollout_path.write_bytes(b"test")

    assert discover_rollout_files(str(task_root)) == [str(rollout_path.resolve())]


def test_nested_singlearm_uses_configured_task_root_for_group(tmp_path):
    task_root = tmp_path / "place_return_neg_y"
    rollout_path = task_root / "mode0" / "demo_seed1_mode0.pkl"
    rollout_path.parent.mkdir(parents=True)
    length = 3
    rollout = {
        "actions_single_arm": np.zeros((length, 7), dtype=np.float32),
        "camera_obs": np.zeros((length, 4, 4, 3), dtype=np.uint8),
        "camera_obs_shoulder": np.zeros((length, 4, 4, 3), dtype=np.uint8),
    }
    with rollout_path.open("wb") as handle:
        pickle.dump(rollout, handle)

    dataset = SingleArmShoulderE2EDataset(
        rollout_dirs=[str(task_root)],
        horizon=2,
        augment=False,
        rollout_files=[str(rollout_path.resolve())],
    )

    assert dataset.trajectory_group_keys == ["singlearm/place_return_neg_y/mode0"]
    assert dataset.trajectory_rollout_ids == [str(rollout_path.resolve())]

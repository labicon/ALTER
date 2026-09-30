"""Deterministic domain-local hierarchical sampling for rollout datasets.

Training code is expected to balance domains with separate loaders.  Within
each loader this sampler gives equal probability to every task/mode bucket,
then every saved rollout in that bucket, then every agent trajectory from that
rollout, and finally every timestep in that trajectory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


HIERARCHY = "domain -> task/mode -> rollout -> agent trajectory -> timestep"


def _selected_indices(dataset, sample_indices: Iterable[int] | None) -> list[int]:
    indices = list(range(len(dataset))) if sample_indices is None else list(sample_indices)
    if not indices:
        raise ValueError("Hierarchical sampling requires at least one sample")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError("Hierarchical sampling indices are outside the dataset")
    return indices


def hierarchical_sample_weights(
    dataset,
    sample_indices: Iterable[int] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Return weights ordered like ``sample_indices`` plus an audit summary.

    Datasets must expose ``index_map`` entries of ``(trajectory, timestep)``
    and one group key and rollout id per trajectory.  Multiple agent
    trajectories may share a rollout id; they divide that rollout's mass
    equally.
    """

    for name in ("index_map", "trajectory_group_keys", "trajectory_rollout_ids"):
        if not hasattr(dataset, name):
            raise TypeError(f"Dataset does not expose required sampling metadata: {name}")
    if not (
        len(dataset.trajectory_group_keys)
        == len(dataset.trajectory_rollout_ids)
        == len(dataset.all_actions)
    ):
        raise ValueError("Trajectory sampling metadata is not aligned with loaded actions")

    indices = _selected_indices(dataset, sample_indices)
    trajectory_sample_counts = Counter()
    group_rollouts: dict[str, set[str]] = defaultdict(set)
    rollout_trajectories: dict[tuple[str, str], set[int]] = defaultdict(set)

    for sample_idx in indices:
        trajectory_idx, _ = dataset.index_map[sample_idx]
        group = str(dataset.trajectory_group_keys[trajectory_idx])
        rollout = str(dataset.trajectory_rollout_ids[trajectory_idx])
        trajectory_sample_counts[trajectory_idx] += 1
        group_rollouts[group].add(rollout)
        rollout_trajectories[(group, rollout)].add(trajectory_idx)

    group_count = len(group_rollouts)
    weights = []
    for sample_idx in indices:
        trajectory_idx, _ = dataset.index_map[sample_idx]
        group = str(dataset.trajectory_group_keys[trajectory_idx])
        rollout = str(dataset.trajectory_rollout_ids[trajectory_idx])
        weight = (
            1.0
            / group_count
            / len(group_rollouts[group])
            / len(rollout_trajectories[(group, rollout)])
            / trajectory_sample_counts[trajectory_idx]
        )
        weights.append(weight)

    tensor = torch.as_tensor(weights, dtype=torch.double)
    legacy_counts = Counter(
        str(dataset.trajectory_group_keys[dataset.index_map[sample_idx][0]])
        for sample_idx in indices
    )
    groups = {}
    for group in sorted(group_rollouts):
        rollout_ids = group_rollouts[group]
        trajectory_counts = [
            len(rollout_trajectories[(group, rollout)]) for rollout in sorted(rollout_ids)
        ]
        rollout_rows = {}
        for rollout in sorted(rollout_ids):
            trajectory_ids = sorted(rollout_trajectories[(group, rollout)])
            rollout_target = 1.0 / group_count / len(rollout_ids)
            agent_target = rollout_target / len(trajectory_ids)
            rollout_rows[rollout] = {
                "target_probability": rollout_target,
                "legacy_timestep_probability": (
                    sum(trajectory_sample_counts[trajectory] for trajectory in trajectory_ids)
                    / len(indices)
                ),
                "agent_trajectories": [
                    {
                        "trajectory_index": trajectory,
                        "timestep_count": trajectory_sample_counts[trajectory],
                        "target_probability": agent_target,
                        "weight_per_timestep": (
                            agent_target / trajectory_sample_counts[trajectory]
                        ),
                    }
                    for trajectory in trajectory_ids
                ],
            }
        groups[group] = {
            "target_probability": 1.0 / group_count,
            "legacy_timestep_probability": legacy_counts[group] / len(indices),
            "rollout_count": len(rollout_ids),
            "target_probability_per_rollout": 1.0 / group_count / len(rollout_ids),
            "agent_trajectories_per_rollout_min": min(trajectory_counts),
            "agent_trajectories_per_rollout_max": max(trajectory_counts),
            "rollouts": rollout_rows,
        }

    summary = {
        "policy": "hierarchical_weighted_replacement",
        "hierarchy": HIERARCHY,
        "sample_count_per_sampler_epoch": len(indices),
        "replacement": True,
        "group_count": group_count,
        "rollout_count": sum(len(rollouts) for rollouts in group_rollouts.values()),
        "agent_trajectory_count": len(trajectory_sample_counts),
        "weight_sum": float(tensor.sum().item()),
        "weight_min": float(tensor.min().item()),
        "weight_max": float(tensor.max().item()),
        "groups": groups,
    }
    return tensor, summary


def make_hierarchical_sampler(
    dataset,
    *,
    sample_indices: Sequence[int] | None = None,
    seed: int,
    stream: int,
) -> tuple[WeightedRandomSampler, dict]:
    weights, summary = hierarchical_sample_weights(dataset, sample_indices)
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1_000_003 * int(stream))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return sampler, summary

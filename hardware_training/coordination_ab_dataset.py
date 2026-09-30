"""Current-frame data with the original phase-balanced A sampling weights."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.coordination_history import history_indices, history_tensor
from hardware_training.coordination_robustness import grasp_transition_windows


class CoordinationABDataset(Dataset):
    def __init__(self, manifest_path, domain, mean, std, history=False, horizon=20, grasp_transition_fraction=0.):
        if history:
            raise ValueError("Hardware camera-history variant B is retired")
        if not 0 <= grasp_transition_fraction < 1:
            raise ValueError("Grasp transition fraction must be in [0,1)")
        manifest = json.loads(Path(manifest_path).read_text())
        self.records = [record for record in manifest["records"] if (record["task"] == "twoarm") == (domain == "twoarm")]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.horizon = horizon
        self.seconds = manifest["history_seconds"]
        self.arrays = []
        self.images = []
        self.index_map = []
        self.invalid_history_anchors = []
        # Preserve the original A/B eligibility filter and summary schema:
        # removing it would change the retained variant A's sampling weights.
        # Historical timestamp helpers remain for dataset/audit reproduction.
        for trajectory, record in enumerate(self.records):
            directory = Path(record["directory"])
            with np.load(directory / "trajectory.npz") as source:
                data = {key: source[key] for key in source.files}
            selected = []
            valid = []
            for anchor, raw_index in enumerate(data["raw_indices"]):
                try:
                    selected.append(history_indices(data["timestamps_ns"], raw_index, self.seconds))
                    valid.append(anchor)
                    self.index_map.append((trajectory, anchor))
                except ValueError:
                    selected.append(np.repeat(raw_index, len(self.seconds)))
                    self.invalid_history_anchors.append([record["key"], anchor])
            data["history_indices"] = np.asarray(selected)
            data["valid"] = np.asarray(valid)
            self.arrays.append(data)
            self.images.append(np.load(directory / "images.npy", mmap_mode="r"))
        self.index_map = np.asarray(self.index_map, dtype=np.int64)
        self.weights = np.zeros(len(self.index_map), dtype=np.float64)
        if domain == "twoarm":
            for arm in (0, 1):
                members = np.asarray([self.records[trajectory]["arm"] == arm for trajectory, anchor in self.index_map])
                baseline = np.asarray([self.arrays[trajectory]["weights"][anchor] for trajectory, anchor in self.index_map]) * members
                self.weights += .25 * baseline / baseline.sum()
                phases = np.asarray([self.arrays[trajectory]["phases"][anchor] for trajectory, anchor in self.index_map])
                present = [phase for phase in range(6) if np.any(members & (phases == phase))]
                for phase in present:
                    phase_members = members & (phases == phase)
                    trajectories = np.unique(self.index_map[phase_members, 0])
                    for trajectory in trajectories:
                        mask = phase_members & (self.index_map[:, 0] == trajectory)
                        self.weights[mask] += .25 / len(present) / len(trajectories) / mask.sum()
        else:
            masses = manifest["singlearm_task_mass"]
            for task, mass in masses.items():
                trajectories = [index for index, record in enumerate(self.records) if record["task"] == task]
                for trajectory in trajectories:
                    mask = self.index_map[:, 0] == trajectory
                    if task == "bird":
                        task_size = sum(len(self.arrays[index]["valid"]) for index in trajectories)
                        self.weights[mask] = mass / sum(masses.values()) / task_size
                    else:
                        self.weights[mask] = mass / sum(masses.values()) / len(trajectories) / mask.sum()
        self.transition_summary = None
        if domain == "twoarm" and grasp_transition_fraction:
            trajectories = [index for index, record in enumerate(self.records) if record["arm"] == 1]
            focused = np.zeros_like(self.weights)
            windows_by_key = {}
            for trajectory in trajectories:
                record = self.records[trajectory]
                windows = grasp_transition_windows(self.arrays[trajectory]["actions"], record["events"], horizon)
                windows_by_key[record["key"]] = windows
                for start, end in windows.values():
                    mask = (self.index_map[:, 0] == trajectory) & (self.index_map[:, 1] >= start) & (self.index_map[:, 1] < end)
                    if not mask.any():
                        raise ValueError("No valid anchors in a grasp-transition stage")
                    focused[mask] += .5 / len(trajectories) / len(windows) / mask.sum()
            cardboard = np.isin(self.index_map[:, 0], trajectories)
            self.weights[cardboard] *= 1 - grasp_transition_fraction
            self.weights += grasp_transition_fraction * focused
            self.transition_summary = dict(cardboard_fraction=grasp_transition_fraction,
                                           total_twoarm_focus_mass=.5 * grasp_transition_fraction,
                                           windows=windows_by_key)
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError("Invalid A/B sampler probabilities")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        trajectory, anchor = self.index_map[index]
        data = self.arrays[trajectory]
        actions = data["actions"]
        rows = np.minimum(np.arange(anchor, anchor + self.horizon), len(actions)-1)
        target = torch.from_numpy((actions[rows] - self.mean) / self.std)
        images = history_tensor([self.images[trajectory][data["raw_indices"][anchor]]])
        current = images[-1]
        return current, target, current, int(data["phases"][anchor])

    def sampler(self, seed):
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(torch.as_tensor(self.weights), len(self), replacement=True, generator=generator)

    def summary(self):
        groups = {}
        for index, (trajectory, anchor) in enumerate(self.index_map):
            record = self.records[trajectory]
            key = f"{record['task']}/arm{record['arm']}/phase{self.arrays[trajectory]['phases'][anchor]}"
            groups[key] = groups.get(key, 0) + float(self.weights[index])
        result = {"samples": len(self), "probabilities": groups, "invalid_history_anchors": self.invalid_history_anchors}
        if self.transition_summary is not None:
            result["grasp_transition"] = self.transition_summary
        return result

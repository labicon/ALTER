import json
from pathlib import Path

import numpy as np
import torch
import pytest

from src.coordination_history import history_tensor

from hardware_training.coordination_ab_dataset import CoordinationABDataset


def make_manifest(root):
    records = []
    for index, (task, arm, length) in enumerate([("twoarm", 0, 30), ("twoarm", 1, 36), ("bird", None, 25), ("bird", None, 35), ("fwd", None, 30), ("bwd", None, 30)]):
        directory = root / str(index)
        directory.mkdir()
        np.save(directory / "images.npy", np.arange(length * 8 * 8 * 3, dtype=np.uint8).reshape(length, 8, 8, 3))
        np.savez(directory / "trajectory.npz", actions=np.zeros((length, 7), np.float32),
                 timestamps_ns=np.arange(length, dtype=np.int64)*50_000_000,
                 raw_indices=np.arange(length), weights=np.ones(length), phases=np.arange(length)%6)
        records.append(dict(key=str(index), task=task, arm=arm, directory=str(directory)))
    path = root / "manifest.json"
    path.write_text(json.dumps(dict(records=records, history_seconds=[1, .5, 0], singlearm_task_mass={"bird": 16766, "fwd": 9024, "bwd": 8122})))
    return path


def test_current_frame_dataset_uses_raw_anchor_image(tmp_path):
    path = make_manifest(tmp_path)
    for domain in ["twoarm", "singlearm"]:
        dataset = CoordinationABDataset(path, domain, np.zeros(7), np.ones(7))
        for index in [0, 15, len(dataset)-1]:
            current, target, side, phase = dataset[index]
            trajectory, anchor = dataset.index_map[index]
            raw_index = dataset.arrays[trajectory]["raw_indices"][anchor]
            expected = history_tensor([dataset.images[trajectory][raw_index]])[0]
            assert torch.equal(current, expected)
            assert torch.equal(side, current)
            assert current.ndim == 3
            assert target.shape == (20, 7)


def test_history_dataset_rejected_before_reading_manifest():
    with pytest.raises(ValueError, match="variant B is retired"):
        CoordinationABDataset("nonexistent.json", "twoarm", np.zeros(7), np.ones(7), True)


def test_explicit_arm_and_singlearm_task_masses(tmp_path):
    path = make_manifest(tmp_path)
    twoarm = CoordinationABDataset(path, "twoarm", np.zeros(7), np.ones(7))
    assert np.isclose(twoarm.weights[twoarm.index_map[:, 0] == 0].sum(), .5)
    assert np.isclose(twoarm.weights[twoarm.index_map[:, 0] == 1].sum(), .5)
    singlearm = CoordinationABDataset(path, "singlearm", np.zeros(7), np.ones(7))
    for task, mass in [("bird", 16766), ("fwd", 9024), ("bwd", 8122)]:
        mask = np.array([singlearm.records[trajectory]["task"] == task for trajectory, anchor in singlearm.index_map])
        assert np.isclose(singlearm.weights[mask].sum(), mass / 33912)
    bird_mask = np.array([singlearm.records[trajectory]["task"] == "bird" for trajectory, anchor in singlearm.index_map])
    assert np.ptp(singlearm.weights[bird_mask]) == 0

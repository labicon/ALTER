import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hardware_training.coordination_ab_dataset import CoordinationABDataset
from hardware_training.coordination_robustness import grasp_transition_windows, light_augment
from test_coordination_ab_dataset import make_manifest


def test_light_augmentation_reproducible_bounded_and_nonmutating():
    torch.set_num_threads(2)
    images = torch.linspace(0, 1, 128 * 128).reshape(1, 1, 128, 128).expand(32, 3, -1, -1).clone()
    original = images.clone()
    first = light_augment(images, torch.Generator().manual_seed(11))
    second = light_augment(images, torch.Generator().manual_seed(11))
    assert torch.equal(first, second)
    assert torch.equal(images, original)
    assert torch.isfinite(first).all() and first.min() >= 0 and first.max() <= 1
    changed = (first != images).flatten(1).any(dim=1)
    assert changed.any() and not changed.all()


def test_temporal_frames_share_the_same_transform():
    images = torch.rand(8, 3, 128, 128)
    history = images[:, None].repeat(1, 3, 1, 1, 1)
    augmented = light_augment(history, torch.Generator().manual_seed(5))
    current = light_augment(images, torch.Generator().manual_seed(5))
    assert torch.equal(augmented[:, 0], augmented[:, 1])
    assert torch.equal(augmented[:, 1], augmented[:, 2])
    assert torch.equal(current, augmented[:, -1])


def test_augmentation_does_not_consume_diffusion_rng():
    state = torch.get_rng_state()
    light_augment(torch.ones(2, 3, 128, 128), torch.Generator().manual_seed(0))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("shape", [(3, 128, 128), (2, 3, 64, 64), (2, 1, 128, 128)])
def test_augmentation_rejects_wrong_shape(shape):
    with pytest.raises(ValueError):
        light_augment(torch.zeros(shape), torch.Generator())


def transition_actions():
    actions = np.zeros((120, 7), dtype=np.float32)
    actions[:, 6] = 800
    actions[40:100, 6] = 100
    actions[65:, 2] = 20
    return actions


def test_transition_windows_include_approach_close_and_lift_without_release():
    windows = grasp_transition_windows(transition_actions(), dict(grasp=40, release=100))
    assert windows == dict(approach=(20, 40), close=(40, 50), lift=(50, 81))
    for anchor in range(*windows["lift"]):
        assert (transition_actions()[anchor:anchor+20, 6] < 300).all()
    with pytest.raises(ValueError, match="lift"):
        grasp_transition_windows(np.zeros((120, 7)), dict(grasp=40, release=100))
    actions = transition_actions()
    actions[:, 2] = 0
    actions[65:, 1] = 25
    assert grasp_transition_windows(actions, dict(grasp=40, release=100)) == windows


def test_focus_preserves_targets_bird_mass_and_singlearm_sampling(tmp_path):
    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    record = manifest["records"][1]
    record["events"] = dict(grasp=40, release=100)
    directory = Path(record["directory"])
    np.save(directory / "images.npy", np.zeros((120, 8, 8, 3), dtype=np.uint8))
    np.savez(directory / "trajectory.npz", actions=transition_actions(),
             timestamps_ns=np.arange(120, dtype=np.int64)*50_000_000,
             raw_indices=np.arange(120), weights=np.ones(120), phases=np.arange(120)//20)
    path.write_text(json.dumps(manifest))
    baseline = CoordinationABDataset(path, "twoarm", np.zeros(7), np.ones(7))
    focused = CoordinationABDataset(path, "twoarm", np.zeros(7), np.ones(7), grasp_transition_fraction=.5)
    bird = baseline.index_map[:, 0] == 0
    assert np.array_equal(baseline.index_map, focused.index_map)
    assert np.array_equal(baseline.weights[bird], focused.weights[bird])
    assert np.isclose(focused.weights[~bird].sum(), .5)
    contribution = focused.weights - np.where(bird, baseline.weights, baseline.weights * .5)
    assert np.isclose(contribution.sum(), .25)
    for start, end in grasp_transition_windows(transition_actions(), record["events"]).values():
        mask = ~bird & (focused.index_map[:, 1] >= start) & (focused.index_map[:, 1] < end)
        assert np.isclose(contribution[mask].sum(), .25 / 3)
    for index in [35, 65, 100]:
        assert torch.equal(baseline[index][1], focused[index][1])
    first = CoordinationABDataset(path, "singlearm", np.zeros(7), np.ones(7))
    second = CoordinationABDataset(path, "singlearm", np.zeros(7), np.ones(7), grasp_transition_fraction=.5)
    assert np.array_equal(first.weights, second.weights)
    assert first.summary() == second.summary()


def test_focus_rejects_invalid_fraction(tmp_path):
    with pytest.raises(ValueError, match="fraction"):
        CoordinationABDataset(tmp_path / "missing", "twoarm", np.zeros(7), np.ones(7), grasp_transition_fraction=1.)

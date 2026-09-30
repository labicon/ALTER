import copy
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from envs.arm.train.hierarchical_sampling import hierarchical_sample_weights
from envs.arm.train.train_singlearm_mixedfront_e2e_shoulder import SingleArmShoulderE2EDataset
from hardware_training.train_singlearm_expert_lightaug import (
    HardwareSingleArmDataset, digest, validate_loaded, validate_sampling, validate_split,
)


class SamplingDataset(SimpleNamespace):
    def __len__(self):
        return len(self.index_map)


class ExpertLightAugTests(unittest.TestCase):
    def hardware_fixture(self, root):
        bird = root / 'pickup_bird_aug16may_pruned_uw_192x256_mode0'
        cardboard = root / 'cardboard_fwd24_bwd16_pruned_uw_192x256_mode0'
        files = []
        for directory, lengths in ((bird, (3, 7)), (cardboard, (5,))):
            directory.mkdir(parents=True)
            for index, length in enumerate(lengths):
                path = directory / f'rollout_{index}_mode0.pkl'
                path.write_bytes(pickle.dumps({
                    'actions_single_arm': np.arange(length * 7, dtype=np.float32).reshape(length, 7),
                    'camera_obs': np.zeros((length, 4, 4, 3), dtype=np.uint8),
                    'camera_obs_shoulder': np.ones((length, 4, 4, 3), dtype=np.uint8),
                }))
                files.append(str(path))
        return dict(rollout_dirs=[str(bird), str(cardboard)], horizon=2,
                    augment=False, rollout_files=files, frame_offsets=(0,))

    def test_hardware_preserves_groups_when_shared_loader_uses_simulation_groups(self):
        original_init = SingleArmShoulderE2EDataset.__init__

        def simulation_init(dataset, *args, **kwargs):
            original_init(dataset, *args, **kwargs)
            # Reproduce main's fallback for these hardware source folders.
            dataset.trajectory_group_keys = ['singlearm/tray_drag/mode0'] * len(dataset.modes)

        with tempfile.TemporaryDirectory() as directory:
            options = self.hardware_fixture(Path(directory))
            with patch.object(SingleArmShoulderE2EDataset, '__init__', simulation_init):
                hardware = HardwareSingleArmDataset(**options)
                simulation = SingleArmShoulderE2EDataset(**options)
            self.assertEqual(set(simulation.trajectory_group_keys), {'singlearm/tray_drag/mode0'})
            expected_groups = [f'singlearm/{Path(path).parent.name}/mode0'
                               for path in hardware.trajectory_rollout_ids]
            self.assertEqual(hardware.trajectory_group_keys, expected_groups)
            weights, summary = hierarchical_sample_weights(hardware)
            self.assertEqual(summary['group_count'], 2)
            for group in set(expected_groups):
                mass = sum(float(weight) for (trajectory, _), weight in zip(hardware.index_map, weights)
                           if expected_groups[trajectory] == group)
                self.assertAlmostEqual(mass, .5)
            reference = SamplingDataset(index_map=hardware.index_map,
                                        trajectory_group_keys=expected_groups,
                                        trajectory_rollout_ids=hardware.trajectory_rollout_ids,
                                        all_actions=hardware.all_actions)
            _, reference_summary = hierarchical_sample_weights(reference)
            validate_sampling(summary, reference_summary)
            # Grouping must not change the images, targets, or normalization.
            np.testing.assert_array_equal(hardware.action_mean, simulation.action_mean)
            np.testing.assert_array_equal(hardware.action_std, simulation.action_std)
            for index in range(len(hardware)):
                for actual, expected in zip(hardware[index], simulation[index]):
                    self.assertTrue(torch.equal(actual, expected))

    def test_hardware_dataset_sampling_survives_payload_path_remap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = HardwareSingleArmDataset(**self.hardware_fixture(root / 'original'))
            remapped = HardwareSingleArmDataset(**self.hardware_fixture(root / 'payload' / 'home' / 'icon'))
            original_weights, reference = hierarchical_sample_weights(original)
            remapped_weights, actual = hierarchical_sample_weights(remapped)
            self.assertTrue(torch.equal(original_weights, remapped_weights))
            validate_sampling(actual, reference)

    def test_sealed_split_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory)
            for name in ('train.pkl', 'val.pkl'):
                (payload / name).write_bytes(name.encode())
            reference = dict(selected_rollout_files=['/train.pkl'], validation_rollout_files=['/val.pkl'])
            manifest = dict(reference, sha256={
                '/' + name: digest(payload / name) for name in ('train.pkl', 'val.pkl')})
            result = validate_split(reference, manifest, payload)
            self.assertEqual(result['selected_rollout_files'], [str(payload / 'train.pkl')])
            (payload / 'train.pkl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate_split(reference, manifest, payload)

    def test_rejects_overlap_and_duplicates(self):
        reference = dict(selected_rollout_files=['/same'], validation_rollout_files=['/same'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_split(reference, dict(reference), Path('/tmp'))
        reference['selected_rollout_files'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_split(reference, dict(reference), Path('/tmp'))

    def test_rejects_changed_selection(self):
        reference = dict(selected_rollout_files=['/train'], validation_rollout_files=['/val'])
        manifest = dict(reference, selected_rollout_files=['/other'])
        with self.assertRaisesRegex(ValueError, 'split differs'):
            validate_split(reference, manifest, Path('/tmp'))

    def test_rejects_silent_dataset_skip(self):
        dataset = SimpleNamespace(trajectory_rollout_ids=['/one'], all_actions=[np.zeros((20, 7))])
        with self.assertRaisesRegex(ValueError, 'skipped'):
            validate_loaded(dataset, ['/one', '/two'])
        validate_loaded(dataset, ['/one'])
        dataset.all_actions[0][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            validate_loaded(dataset, ['/one'])

    def test_sampling_preserved_across_path_remap(self):
        dataset = SamplingDataset(index_map=[(0, 0), (0, 1), (1, 0)],
                                  trajectory_group_keys=['bird', 'cardboard'],
                                  trajectory_rollout_ids=['/old/bird.pkl', '/old/cardboard.pkl'],
                                  all_actions=[np.zeros((2, 7)), np.zeros((1, 7))])
        _, reference = hierarchical_sample_weights(dataset, sample_indices=range(3))
        dataset.trajectory_rollout_ids = ['/new/bird.pkl', '/new/cardboard.pkl']
        _, actual = hierarchical_sample_weights(dataset, sample_indices=range(3))
        validate_sampling(actual, reference)
        changed = copy.deepcopy(actual)
        changed['groups']['bird']['rollouts']['/new/bird.pkl']['target_probability'] = .7
        with self.assertRaisesRegex(ValueError, 'Rollout sampling'):
            validate_sampling(changed, reference)


if __name__ == '__main__':
    unittest.main()

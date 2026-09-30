"""Retrain the original expert-only policy with a sealed split and light augmentation."""

import argparse
import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from envs.arm.train import _train_utils
from envs.arm.train.hierarchical_sampling import make_hierarchical_sampler
from envs.arm.train.train_singlearm_mixedfront_e2e_shoulder import SingleArmShoulderE2EDataset
from hardware_training.coordination_robustness import LIGHT_AUGMENTATION, light_augment
from src.image_diffusion import ImageConditional_ODE


PREFIX = 'singlearm_mixedfront_e2e_shoulder'


class HardwareSingleArmDataset(SingleArmShoulderE2EDataset):
    """Keep the original hardware source/mode groups independent of simulation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_group_keys = [
            f'singlearm/{_train_utils.rollout_source_name(path, self.rollout_dirs)}/mode{mode}'
            for path, mode in zip(self.trajectory_rollout_ids, self.modes, strict=True)
        ]


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def validate_split(reference, manifest, payload):
    splits = {}
    for key in ('selected_rollout_files', 'validation_rollout_files'):
        original = reference[key]
        if len(original) != len(set(original)):
            raise ValueError('Duplicate rollout in ' + key)
        if set(original) != set(manifest[key]):
            raise ValueError('Reference split differs from sealed manifest: ' + key)
        splits[key] = [str(payload / path.lstrip('/')) for path in original]
    if set(reference['selected_rollout_files']) & set(reference['validation_rollout_files']):
        raise ValueError('Training/validation overlap')
    for original, expected in manifest['sha256'].items():
        path = payload / original.lstrip('/')
        if digest(path) != expected:
            raise ValueError('Data checksum mismatch: ' + str(path))
    if set(manifest['sha256']) != set(reference['selected_rollout_files'] + reference['validation_rollout_files']):
        raise ValueError('Incomplete checksum coverage')
    return splits


def validate_loaded(dataset, files):
    if sorted(dataset.trajectory_rollout_ids) != sorted(files):
        raise ValueError('Dataset skipped or duplicated an expert trajectory')
    if any(not np.isfinite(actions).all() for actions in dataset.all_actions):
        raise ValueError('Nonfinite expert targets')


def validate_sampling(summary, reference):
    for key in ('sample_count_per_sampler_epoch', 'rollout_count', 'agent_trajectory_count', 'group_count'):
        if summary[key] != reference[key]:
            raise ValueError('Sampling changed: ' + key)
    for group, expected in reference['groups'].items():
        actual = summary['groups'][group]
        for key in ('target_probability', 'rollout_count', 'legacy_timestep_probability'):
            if actual[key] != expected[key]:
                raise ValueError('Group sampling changed: ' + group + '/' + key)
        expected_rollouts = {Path(path).name: value for path, value in expected['rollouts'].items()}
        for path, actual_rollout in actual['rollouts'].items():
            expected_rollout = expected_rollouts[Path(path).name]
            if actual_rollout != expected_rollout:
                raise ValueError('Rollout sampling changed: ' + path)


def run(args):
    if args.steps <= 0 or args.save_every <= 0:
        raise ValueError('Training and save intervals must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / 'STATUS.json'
    write_json(status_path, dict(stage='verifying_data', pid=os.getpid()))
    reference = pickle.loads(args.reference_stats.read_bytes())
    manifest = json.loads(args.manifest.read_text())
    if digest(args.reference_stats) != manifest['reference_stats_sha256']:
        raise ValueError('Reference stats checksum mismatch')
    splits = validate_split(reference, manifest, args.payload)
    if reference['num_cameras'] != 1 or reference['frame_offsets'] != [0]:
        raise ValueError('Expected original single-camera, single-frame base')
    torch.set_num_threads(4)
    _train_utils.seed_training(0)
    device = torch.device('cuda')
    directories = [str(args.payload / path.lstrip('/')) for path in reference['rollout_dirs']]
    datasets = {}
    for key, files in splits.items():
        dataset = HardwareSingleArmDataset(directories, horizon=reference['horizon'],
                                           augment=False, rollout_files=files, frame_offsets=(0,))
        validate_loaded(dataset, files)
        if key == 'selected_rollout_files':
            np.testing.assert_allclose(dataset.action_mean, reference['action_mean'], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(dataset.action_std, reference['action_std'], rtol=1e-6, atol=1e-6)
        dataset.action_mean = reference['action_mean']
        dataset.action_std = reference['action_std']
        datasets[key] = dataset
    train = datasets['selected_rollout_files']
    sampler, summary = make_hierarchical_sampler(train, seed=0, stream=100)
    validate_sampling(summary, reference['effective_sampling_weights'])
    loader_options = dict(batch_size=256, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train, sampler=sampler, drop_last=True, **loader_options)
    val_loader = DataLoader(datasets['validation_rollout_files'], shuffle=False, **loader_options)
    model = ImageConditional_ODE(
        x_dim=7, sigma_data=reference['sigma_data'], d_model=reference['d_model'],
        n_heads=reference['n_heads'], depth=reference['depth'],
        dim_feedforward=reference['dim_feedforward'], horizon=reference['horizon'],
        device=device, N=50, lr=2e-4, cfg_drop_prob=reference['cfg_drop_prob'],
        num_cameras=1, backbone=reference['backbone'], frame_offsets=(0,))
    if sum(parameter.numel() for parameter in model.F.parameters()) != reference['n_params']:
        raise ValueError('Architecture differs from the original base')
    stats = dict(reference)
    stats.update(rollout_dirs=directories, **splits, effective_sampling_weights=summary,
                 initialization='fresh_full_policy_resnet18_weights_none',
                 training_seed=0, augmentation=LIGHT_AUGMENTATION,
                 reference_stats_sha256=manifest['reference_stats_sha256'],
                 expert_manifest=manifest, original_reference_stats=reference,
                 training_config=dict(steps=args.steps, batch_size=256, learning_rate=2e-4,
                                      optimizer='AdamW', weight_decay=1e-4,
                                      grad_clip=10., ema_decay=.999, save_every=args.save_every,
                                      singlearm_only=True, distilled_replay=False))
    (args.output / (PREFIX + '_stats.pkl')).write_bytes(pickle.dumps(stats))
    generator = torch.Generator(device=device).manual_seed(20015)
    sigma_grid = _train_utils.build_sigma_grid(model, 8)

    def validation(batch, noise_generator, sigma):
        _, images, actions = batch
        actions = actions.to(device, non_blocking=True)
        loss = model.validation_loss(actions, None, images.to(device, non_blocking=True),
                                     generator=noise_generator, sigma=sigma)
        return loss, len(actions)

    history = []
    iterator = iter(train_loader)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            _, images, actions = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            _, images, actions = next(iterator)
        images = light_augment(images.to(device, non_blocking=True), generator)
        loss, grad = model.update(actions.to(device, non_blocking=True), None, images)
        if not np.isfinite([loss, grad]).all():
            raise ValueError('Nonfinite optimization at step ' + str(step))
        if step % 50 == 0 or step == 1:
            speed = step / (time.monotonic() - started)
            status = dict(stage='training', pid=os.getpid(), step=step, steps=args.steps,
                          loss=loss, grad_norm=grad, steps_per_second=speed,
                          eta=(datetime.now().astimezone() + timedelta(seconds=(args.steps-step)/speed)).isoformat())
            write_json(status_path, status)
            print(json.dumps(status), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            val_loss = _train_utils.run_validation(val_loader, validation, val_batches=0,
                                                   sigma_grid=sigma_grid, noise_seed=42, device=str(device))
            checkpoint = args.output / f'{PREFIX}_step{step}.pt'
            temporary = checkpoint.with_suffix('.tmp.pt')
            model.save(str(temporary))
            temporary.replace(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            for key in ('model', 'model_ema'):
                if any(not torch.isfinite(value).all() for value in saved[key].values()):
                    raise ValueError('Nonfinite saved checkpoint')
            history.append(dict(step=step, train_loss=loss, val_loss=val_loss,
                                checkpoint=str(checkpoint), sha256=digest(checkpoint)))
            write_json(args.output / 'checkpoint_metrics.json', history)
            print(json.dumps(history[-1]), flush=True)
    write_json(status_path, dict(stage='complete', step=args.steps, pid=os.getpid()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-stats', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--save-every', type=int, default=5000)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / 'ERROR.json', dict(error=repr(error), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()

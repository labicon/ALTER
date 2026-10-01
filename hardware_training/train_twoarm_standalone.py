"""Train a fresh or pretrained full policy with optional single-arm replay."""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from envs.arm.train import _train_utils
from hardware_training.coordination_ab_dataset import CoordinationABDataset
from hardware_training.coordination_robustness import LIGHT_AUGMENTATION, light_augment
from hardware_training.stage_robustness_manifest import array_digest
from src.image_diffusion import ImageConditional_ODE


PREFIX = 'twoarm_standalone_hardware'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare_manifest(source, destination, expected_episodes=39, include_singlearm=False):
    manifest = json.loads(source.read_text())
    twoarm = [record for record in manifest['records'] if record['task'] == 'twoarm']
    if len(twoarm) != 2 * expected_episodes or len({record['key'] for record in twoarm}) != len(twoarm):
        raise ValueError(f'Expected {2 * expected_episodes} distinct arm trajectories')
    arm_ids = [{record.get('source', record['key'].rsplit('_', 1)[-1]) for record in twoarm if record['arm'] == arm} for arm in (0, 1)]
    if len(arm_ids[0]) != expected_episodes or arm_ids[0] != arm_ids[1]:
        raise ValueError('Two-arm episode membership mismatch')
    records = manifest['records'] if include_singlearm else twoarm
    trajectories = []
    for record in records:
        directory = Path(record['directory'])
        images = np.load(directory / 'images.npy', mmap_mode='r')
        with np.load(directory / 'trajectory.npz') as data:
            checks = dict(actions_sha256=array_digest(data['actions']),
                          camera_sha256=array_digest(images),
                          timestamps_sha256=array_digest(data['timestamps_ns']))
            if any(record[key] != value for key, value in checks.items()):
                raise ValueError('Changed training data: ' + record['key'])
            if not np.isfinite(data['actions']).all():
                raise ValueError('Nonfinite targets: ' + record['key'])
            trajectories.append(data['actions'].copy())
    selected = dict(schema='twoarm_standalone_sep13', records=records,
                    history_seconds=manifest['history_seconds'],
                    singlearm_task_mass=manifest['singlearm_task_mass'] if include_singlearm else {},
                    source_manifest=str(source.resolve()),
                    source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write_json(destination, selected)
    actions = np.concatenate(trajectories)
    mean = actions.mean(axis=0).astype(np.float32)
    std = actions.std(axis=0).clip(1e-6).astype(np.float32)
    return mean, std, float(((actions - mean) / std).std())


def make_model(stats, device, learning_rate):
    return ImageConditional_ODE(
        x_dim=7, sigma_data=stats['sigma_data'], d_model=stats['d_model'],
        n_heads=stats['n_heads'], depth=stats['depth'],
        dim_feedforward=stats['dim_feedforward'], horizon=20, device=device,
        N=50, lr=learning_rate, cfg_drop_prob=.2, num_cameras=1, backbone='resnet18')


def initialize_from_base(model, stats, checkpoint, base_stats_path):
    with base_stats_path.open('rb') as stream:
        base_stats = pickle.load(stream)
    for key in ('action_mean', 'action_std', 'sigma_data', 'd_model', 'n_heads',
                'depth', 'dim_feedforward', 'horizon', 'num_cameras', 'backbone', 'frame_offsets'):
        if key not in base_stats or not np.array_equal(stats[key], base_stats[key]):
            raise ValueError('Pretrained base configuration mismatch: ' + key)
    if model.optim.state:
        raise ValueError('Fine-tuning requires a fresh optimizer')
    if not model.load(checkpoint):
        raise FileNotFoundError(checkpoint)
    model.F.load_state_dict(model.F_ema.state_dict(), strict=True)
    for name, tensor in model.F.state_dict().items():
        if not torch.equal(tensor, model.F_ema.state_dict()[name]):
            raise ValueError('Pretrained EMA initialization mismatch: ' + name)
    stats.update(initialization='pretrained_base_ema_for_model_and_ema',
                 encoder_initialization='pretrained_base_ema',
                 initial_checkpoint=str(checkpoint),
                 initial_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 initial_base_stats=str(base_stats_path),
                 initial_base_stats_sha256=hashlib.sha256(base_stats_path.read_bytes()).hexdigest(),
                 optimizer_initialization='fresh', pretrained_source='model_ema')


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.steps, args.batch_size, args.save_every) <= 0 or args.workers < 0:
        raise ValueError('Invalid training budget')
    if bool(args.init_checkpoint) != bool(args.init_stats):
        raise ValueError('--init-checkpoint and --init-stats must be supplied together')
    if args.init_checkpoint and args.reference_stats is None:
        raise ValueError('Fine-tuning requires the matched --reference-stats')
    args.output.mkdir(parents=True)
    status_path = args.output / 'STATUS.json'
    write_json(status_path, dict(stage='preflight', pid=os.getpid(), time=datetime.now().isoformat()))
    torch.set_num_threads(4)
    _train_utils.seed_training(0)
    device = torch.device(args.device)
    mean, std, sigma_data = prepare_manifest(args.manifest, args.output / 'manifest.json',
                                            args.expected_episodes, args.include_singlearm)
    if args.reference_stats is not None:
        with args.reference_stats.open('rb') as stream:
            reference = pickle.load(stream)
        mean, std, sigma_data = reference['action_mean'], reference['action_std'], reference['sigma_data']
    stats = dict(pipeline='twoarm_standalone_sep13', action_mean=mean, action_std=std,
                 sigma_data=sigma_data, d_model=256, n_heads=4, depth=3, dim_feedforward=1024,
                 horizon=20, num_cameras=1, backbone='resnet18', frame_offsets=[0],
                 cfg_drop_prob=.2, initialization='random_all_parameters_no_pretrained_weights',
                 encoder_initialization='usual_resnet18_weights_none',
                 singlearm_training=args.include_singlearm, singlearm_weight=.25 if args.include_singlearm else 0.,
                 singlearm_objective='imitation' if args.include_singlearm else None,
                 frozen_base=False, coordination_head=False,
                 training_seed=0, max_train_steps=args.steps, batch_size=args.batch_size,
                 learning_rate=args.lr, augmentation_config=LIGHT_AUGMENTATION,
                 grasp_transition_fraction=args.grasp_transition_fraction, pilot=args.pilot,
                 dataset_manifest=json.loads((args.output / 'manifest.json').read_text()))
    dataset = CoordinationABDataset(args.output / 'manifest.json', 'twoarm', mean, std,
                                    grasp_transition_fraction=args.grasp_transition_fraction)
    if any(record['task'] != 'twoarm' for record in dataset.records):
        raise ValueError('Single-arm data reached the training dataset')
    domains = [('twoarm', 1.)] + ([('singlearm', .25)] if args.include_singlearm else [])
    datasets = {'twoarm': dataset}
    if args.include_singlearm:
        datasets['singlearm'] = CoordinationABDataset(args.output / 'manifest.json', 'singlearm', mean, std)
    stats['sampling'] = {domain: selected.summary() for domain, selected in datasets.items()}
    stats['data_manifest_sha256'] = hashlib.sha256((args.output / 'manifest.json').read_bytes()).hexdigest()
    write_json(args.output / 'sampling.json', stats['sampling'])
    write_json(args.output / 'argv.json', {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    loaders = {}
    for stream, (domain, selected) in enumerate(datasets.items()):
        options = dict(batch_size=args.batch_size, sampler=selected.sampler(100 + stream), num_workers=args.workers,
                       pin_memory=True, drop_last=True, **_train_utils.dataloader_seed_kwargs(0, stream=stream))
        if args.workers:
            options.update(persistent_workers=True, prefetch_factor=2)
        loaders[domain] = DataLoader(selected, **options)
    model = make_model(stats, device, args.lr)
    if args.init_checkpoint is not None:
        initialize_from_base(model, stats, args.init_checkpoint, args.init_stats)
    if not all(parameter.requires_grad for parameter in model.F.parameters()):
        raise ValueError('Full policy contains frozen trainable-network parameters')
    stats['trainable_parameters'] = sum(parameter.numel() for parameter in model.F.parameters())
    digest = hashlib.sha256()
    for name, tensor in sorted(model.F.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    stats['initial_model_sha256'] = digest.hexdigest()
    with (args.output / f'{PREFIX}_stats.pkl').open('wb') as stream:
        pickle.dump(stats, stream)
    augmentation_generators = {domain: torch.Generator(device=device).manual_seed(200 + stream)
                               for stream, domain in enumerate(loaders)}
    iterators = {domain: iter(loader) for domain, loader in loaders.items()}
    started = time.monotonic()
    evaluation_seconds = 0.
    checks = {name: parameter.detach().clone() for name, parameter in model.F.named_parameters()
              if name.endswith('weight') and (name.endswith('conv1.weight') or name.endswith('out_proj.weight'))}
    print('FULL POLICY', stats['trainable_parameters'], 'trainable parameters;', domains, 'twoarm anchors', len(dataset), flush=True)
    for step in range(1, args.steps + 1):
        model.optim.zero_grad(set_to_none=True)
        domain_losses = {}
        for domain, multiplier in domains:
            try:
                images, targets, _, _ = next(iterators[domain])
            except StopIteration:
                iterators[domain] = iter(loaders[domain])
                images, targets, _, _ = next(iterators[domain])
            images = light_augment(images.to(device, non_blocking=True), augmentation_generators[domain])
            targets = targets.to(device, non_blocking=True)
            loss = model._training_loss(targets, None, images)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite domain loss: ' + domain)
            (multiplier * loss).backward()
            domain_losses[domain + '_loss'] = float(loss.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.F.parameters(), 10.)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError('Nonfinite training loss or gradient')
        model.optim.step()
        model.ema_update()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(gradient_norm),
                       **domain_losses,
                       seconds=time.monotonic()-started, evaluation_seconds=evaluation_seconds,
                       updates_per_second=step/(time.monotonic()-started-evaluation_seconds),
                       peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else 0)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            write_json(status_path, dict(stage='training', pid=os.getpid(), time=datetime.now().isoformat(), **row))
            print(json.dumps(row), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = args.output / f'{PREFIX}_step{step}.pt'
            temporary = checkpoint.with_suffix('.tmp')
            model.save(temporary)
            temporary.replace(checkpoint)
            changed = {name: not torch.equal(before, dict(model.F.named_parameters())[name]) for name, before in checks.items()}
            if not changed or not all(changed.values()):
                raise RuntimeError('Encoder/decoder parameter update verification failed')
            write_json(args.output / f'step{step}.ready', dict(step=step, checked_parameters_changed=changed))
            if args.eval_bundle is not None and (step % 10000 == 0 or step == args.steps):
                evaluation_started = time.monotonic()
                destination = args.output / 'evaluations' / f'step{step}'
                destination.mkdir(parents=True)
                common = ['--base-only', '--base-checkpoint', str(checkpoint), '--base-stats', str(args.output / f'{PREFIX}_stats.pkl')]
                commands = [
                    [sys.executable, '-m', 'hardware_training.eval_coordination_robustness', '--standalone',
                     '--checkpoint', str(checkpoint), '--stats', str(args.output / f'{PREFIX}_stats.pkl'),
                     '--bundle', str(args.eval_bundle), '--samples', '8', '--out', str(destination / 'robustness.json')],
                    [sys.executable, '-m', 'hardware_training.bench_failure_states', *common,
                     '--n-samples', '8', '--seed', '0', '--out', str(destination / 'legacy.json')],
                    [sys.executable, '-m', 'hardware_training.sanity_dualarm_phases', *common,
                     '--n-samples', '8', '--seed', '0'],
                ]
                for index, command in enumerate(commands):
                    write_json(status_path, dict(stage='evaluation', step=step, command=command, pid=os.getpid()))
                    with (destination / f'suite{index}.log').open('w') as stream:
                        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                    write_json(destination / f'suite{index}.status.json', dict(returncode=result.returncode, command=command))
                evaluation_seconds += time.monotonic() - evaluation_started
    write_json(args.output / 'TRAINING_COMPLETE.json', dict(steps=args.steps, seconds=time.monotonic()-started))
    write_json(status_path, dict(stage='complete', steps=args.steps, time=datetime.now().isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--device', default='cuda', help='Training device, e.g. cuda or cpu')
    parser.add_argument('--save-every', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--eval-bundle', type=Path)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--expected-episodes', type=int, default=39)
    parser.add_argument('--include-singlearm', action='store_true')
    parser.add_argument('--grasp-transition-fraction', type=float, default=.5)
    parser.add_argument('--reference-stats', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--init-stats', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing output directory')
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'FAILED.json', dict(error=str(error), time=datetime.now().isoformat()))
        raise


if __name__ == '__main__':
    main()

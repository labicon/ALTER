"""Train the retained current-frame, phase-balanced variant A with a frozen base."""

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from envs.arm.train import _train_utils
from hardware_training.coordination_ab_dataset import CoordinationABDataset
from hardware_training.coordination_robustness import LIGHT_AUGMENTATION, light_augment
from hardware_training.current_frame import require_current_frame_hardware
from src.image_coordination import ImageCoordinationHead
from src.image_diffusion import ImageConditional_ODE


def make_head(stats, device):
    require_current_frame_hardware(stats)
    return ImageCoordinationHead(
        x_dim=7, base_d_model=stats["base_d_model"], d_model=256,
        n_heads=4, depth=4, dim_feedforward=1024, horizon=20,
        sigma_data=stats["sigma_data"], lr=2e-4, num_cameras=1,
        tokens_per_camera=stats["tokens_per_camera"], d_base_drop_prob=.1,
        use_side_net=True, side_net_input_size=128, side_net_fusion="concat",
        side_net_tokens_per_camera=64, decoder_execution="all_blocks_repeat_last",
        decoder_conditioning="cross_attn", frame_offsets=(0,),
    ).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["A"], required=True,
                        help="Current-frame variant A; history variant B is retired.")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--tiny-fit", action="store_true")
    parser.add_argument("--light-augmentation", action="store_true")
    parser.add_argument("--grasp-transition-fraction", type=float, default=0.)
    parser.add_argument("--twoarm-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-stats", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, help="Override the base checkpoint path recorded in reference statistics")
    parser.add_argument("--base-stats", type=Path, help="Override the matching base statistics path")
    args = parser.parse_args()
    if args.batch_size % args.microbatch or min(args.steps, args.microbatch, args.save_every) <= 0:
        raise ValueError("Invalid step/batch configuration")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    _train_utils.seed_training(0)
    device = torch.device(args.device)
    reference = args.reference_stats
    with reference.open("rb") as stream:
        stats = pickle.load(stream)
    for field, override, option in (("base_model_path", args.base_checkpoint, "--base-checkpoint"),
                                    ("base_stats_path", args.base_stats, "--base-stats")):
        selected = override if override is not None else Path(stats[field])
        if not selected.is_file():
            raise FileNotFoundError(f"Missing {field}: {selected}; supply {option} with the matching local artifact")
        stats[field] = str(selected.resolve())
    with Path(stats["base_stats_path"]).open("rb") as stream:
        base_stats = pickle.load(stream)
    if stats.get("frame_offsets", [0]) != [0] or stats["horizon"] != 20:
        raise ValueError("Unexpected reference base/history")
    base = ImageConditional_ODE(
        x_dim=7, sigma_data=stats["sigma_data"], d_model=base_stats["d_model"],
        n_heads=base_stats["n_heads"], depth=base_stats["depth"],
        dim_feedforward=base_stats["dim_feedforward"], horizon=20, device=device,
        num_cameras=1, backbone=base_stats.get("backbone", "resnet18"), N=50,
    )
    if not base.load(stats["base_model_path"]):
        raise FileNotFoundError(stats["base_model_path"])
    for network in (base.F, base.F_ema):
        network.requires_grad_(False).eval()
    torch.manual_seed(0)
    head = make_head(stats, device)
    _train_utils.seed_training(0)
    domains = [("twoarm", 1.0)] if args.twoarm_only else [("twoarm", 1.0), ("singlearm", .25)]
    datasets = {domain: CoordinationABDataset(args.manifest, domain, stats["action_mean"], stats["action_std"], grasp_transition_fraction=args.grasp_transition_fraction) for domain, _ in domains}
    loaders = {}
    summaries = {}
    for stream, (domain, dataset) in enumerate(datasets.items()):
        summaries[domain] = dataset.summary()
        sampler = dataset.sampler(100 + stream)
        loader_dataset = dataset
        if args.tiny_fit:
            indices = []
            for trajectory, record in enumerate(dataset.records):
                if record["key"] not in {"ta_cardboard_36", "ta_bird_36", "sa_fwd_26", "sa_bwd_26", "sa_bird_01"}:
                    continue
                for phase in np.unique(dataset.arrays[trajectory]["phases"]):
                    members = [index for index, (member, anchor) in enumerate(dataset.index_map) if member == trajectory and dataset.arrays[trajectory]["phases"][anchor] == phase]
                    if members:
                        indices.extend([members[int(fraction * (len(members)-1))] for fraction in [.2, .5, .8]])
            if not indices:
                raise ValueError("Empty tiny-fit dataset")
            summaries[domain]["tiny_fit_indices"] = indices
            loader_dataset = Subset(dataset, indices)
            sampler = WeightedRandomSampler(torch.ones(len(indices)), max(args.batch_size, len(indices)), True, generator=torch.Generator().manual_seed(100+stream))
        options = dict(batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
        if args.workers:
            options.update(persistent_workers=True, prefetch_factor=2)
        options.update(_train_utils.dataloader_seed_kwargs(0, stream=stream))
        loaders[domain] = DataLoader(loader_dataset, **options)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    initial_head_digest = hashlib.sha256()
    for name, tensor in sorted(head.F.state_dict().items()):
        initial_head_digest.update(name.encode())
        initial_head_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    stats.update(
        pipeline="coordination_ab_sep12", variant=args.variant, training_seed=0,
        frame_offsets=[0], side_frame_offsets=[0],
        head_history_seconds=[0.0],
        singlearm_weight=0. if args.twoarm_only else .25, singlearm_training=not args.twoarm_only,
        augment=args.light_augmentation,
        augmentation_config=LIGHT_AUGMENTATION if args.light_augmentation else None,
        grasp_transition_fraction=args.grasp_transition_fraction,
        initial_head_sha256=initial_head_digest.hexdigest(),
        sampling_policy="phase_balanced_with_cardboard_grasp_transition" if args.grasp_transition_fraction else "phase_balanced_half_baseline",
        effective_sampling_weights=summaries, per_domain_batch_size=args.batch_size,
        microbatch=args.microbatch, max_train_steps=args.steps, checkpoint_every_steps=args.save_every,
        data_manifest_path=str(args.manifest.resolve()), data_manifest_sha256=manifest_hash,
        dataset_manifest=json.loads(args.manifest.read_text()), pilot=args.pilot, tiny_fit=args.tiny_fit,
    )
    prefix = "mixed_coord_head_placewipe_hardware"
    with (args.output / f"{prefix}_stats.pkl").open("wb") as stream:
        pickle.dump(stats, stream)
    (args.output / "sampling.json").write_text(json.dumps(summaries, indent=2))
    (args.output / "argv.json").write_text(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2))
    head.checkpoint_metadata = {"pipeline": stats["pipeline"], "variant": args.variant, "data_manifest_sha256": manifest_hash,
                                "singlearm_weight": stats["singlearm_weight"], "head_history_seconds": stats["head_history_seconds"], "training_seed": 0,
                                "augmentation_config": stats["augmentation_config"], "grasp_transition_fraction": args.grasp_transition_fraction}
    iterators = {domain: iter(loader) for domain, loader in loaders.items()}
    augmentation_generators = {domain: torch.Generator(device=device).manual_seed(200 + index) for index, domain in enumerate(loaders)}
    start_time = time.monotonic()
    initial_base = [parameter.detach().clone() for parameter in base.F_ema.parameters()]
    metrics_file = args.output / "metrics.jsonl"
    print("TRAINING", args.variant, "steps", args.steps, "batch", args.batch_size, "microbatch", args.microbatch, flush=True)
    for step in range(1, args.steps + 1):
        head.optimizer.zero_grad(set_to_none=True)
        summary = {}
        for domain, multiplier in domains:
            try:
                batch = next(iterators[domain])
            except StopIteration:
                iterators[domain] = iter(loaders[domain])
                batch = next(iterators[domain])
            current, targets, side, phases = batch
            domain_loss = 0.0
            residual_sum = 0.0
            details = []
            for offset in range(0, len(targets), args.microbatch):
                end = offset + args.microbatch
                diagnostics = {} if step == 1 or step % 250 == 0 else None
                current_images = current[offset:end].to(device, non_blocking=True)
                side_images = side[offset:end].to(device, non_blocking=True)
                if args.light_augmentation:
                    side_images = light_augment(side_images, augmentation_generators[domain])
                    current_images = side_images
                loss, delta = head._training_domain_loss(
                    targets[offset:end].to(device, non_blocking=True), None,
                    current_images, base,
                    zero_residual_target=domain == "singlearm",
                    diagnostics=diagnostics,
                )
                fraction = len(targets[offset:end]) / len(targets)
                (loss * multiplier * fraction).backward()
                domain_loss += float(loss.detach()) * fraction
                residual_sum += float(delta.detach().square().mean()) * fraction
                if diagnostics is not None:
                    details.append({key: value.cpu().numpy() for key, value in diagnostics.items()})
            summary[domain + "_loss"] = domain_loss
            summary[domain + "_residual_rms"] = residual_sum ** .5
            if details:
                combined = {key: np.concatenate([detail[key] for detail in details]) for key in details[0]}
                phase_values = phases.numpy()
                for phase in np.unique(phase_values):
                    mask = phase_values == phase
                    summary[f"{domain}/phase{phase}/loss"] = float(combined["loss"][mask].mean())
                for lower, upper in [(0, .1), (.1, 1), (1, 10), (10, float("inf"))]:
                    mask = (combined["sigma"] >= lower) & (combined["sigma"] < upper)
                    if mask.any():
                        for key in ["loss", "base_error", "residual_rms"]:
                            summary[f"{domain}/sigma{lower}-{upper}/{key}"] = float(combined[key][mask].mean())
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
        if not np.isfinite(list(summary.values())).all() or not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite training loss/gradient")
        head.optimizer.step()
        head.ema_update()
        head.last_train_metrics = summary
        if step == 1 or step % 100 == 0 or step % 250 == 0 or step == args.steps:
            summary.update(step=step, seconds=time.monotonic()-start_time, steps_per_second=step/(time.monotonic()-start_time),
                           gradient_norm=float(norm), peak_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.type == "cuda" else 0)
            with metrics_file.open("a") as stream:
                stream.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            if any(not torch.equal(before, after) for before, after in zip(initial_base, base.F_ema.parameters())):
                raise RuntimeError("Frozen base changed")
            target = args.output / f"{prefix}_step{step}.pt"
            temporary = target.with_suffix(".tmp")
            head.save(temporary, metadata={"step": step})
            os.replace(temporary, target)
            (args.output / f"step{step}.ready").write_text(manifest_hash)
    (args.output / "TRAINING_COMPLETE.json").write_text(json.dumps({"steps": args.steps, "seconds": time.monotonic()-start_time, "variant": args.variant, "data_manifest_sha256": manifest_hash}))


if __name__ == "__main__":
    main()

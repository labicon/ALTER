"""Train the legacy coordination head on paired two-arm and single-arm batches.

The frozen base policy defines normalization for both domains. Two-arm data
uses the existing expert denoising objective; single-arm data applies a
Karras-weighted zero-residual objective. Every optimizer step consumes one
batch from each domain.
"""

import argparse
import hashlib
import json
import os
import pickle as pkl
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.experiment_contract import load_capacity_ablation_job_contract, write_training_contract

from envs.arm.train import _train_utils
from envs.arm.train.exact_threearm_manifest import load_exact_manifest
from envs.arm.train.lowvar_combined_manifest import (
    HIGHVAR_V3_SCHEMA,
    SCHEMA as LOWVAR_COMBINED_SCHEMA,
    load_lowvar_combined_manifest,
)
from envs.arm.train.hierarchical_sampling import HIERARCHY, make_hierarchical_sampler
from torch.utils.data import WeightedRandomSampler
from envs.arm.train.train_singlearm_mixedfront_e2e_shoulder import (
    SingleArmShoulderE2EDataset,
)
from envs.arm.train.twoarm_dataset import TwoArmE2EImageDataset
from src.image_coordination import (
    ALL_BLOCKS_DECODER_EXECUTION,
    DECODER_CONDITIONING_CROSS_ATTN,
    DECODER_CONDITIONING_POOLED,
    ImageCoordinationHead,
    LEGACY_DECODER_EXECUTION,
    PLAN_MEMORY_ADDITIVE,
    VALID_PLAN_MEMORY_MODES,
)
from src.image_diffusion import ImageConditional_ODE
from src.temporal import frame_offsets_from_stats, normalize_frame_offsets
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twoarm-dir", required=True)
    parser.add_argument("--singlearm-dirs", nargs="+", required=True)
    parser.add_argument(
        "--subset-stats-path",
        help=("Stats pkl from the matching matched-low from-scratch run. Its "
              "selected and validation file lists are reused exactly."),
    )
    parser.add_argument(
        "--manifest-path",
        help="Exact three-arm JSON manifest; bypasses legacy subset-stats selection.",
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--base-stats-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-prefix", default="mixed_coord_head_placewipe")
    parser.add_argument("--capacity-job-contract", help="Immutable capacity-ablation job spec to seal into checkpoint contracts.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--frame-offsets", type=int, nargs="+",
        help="Assert the frozen base temporal setting (e.g. 15 0). Defaults to base stats.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--other-arm-cutout", type=str, nargs="*", default=None, metavar="AGENT:x0,y0,x1,y1",
        help="Two-arm augmentation: blank this region (width/height fractions) of AGENT's own "
             "camera with probability --other-arm-cutout-prob, e.g. 0:0.515,0,1,1 hides the "
             "cardboard arm in the bird camera. Stops the head keying on the other arm's posture.",
    )
    parser.add_argument("--other-arm-cutout-prob", type=float, default=0.0)
    parser.add_argument("--singlearm-weight", type=float, default=1.0)
    parser.add_argument(
        "--sampling-policy", choices=("timestep", "hierarchical", "weighted"), default="timestep",
        help="Within-domain sampling policy; hierarchical balances task/mode, rollout, agent, then "
             "timestep. weighted draws two-arm samples in proportion to the per-frame "
             "``frame_weights`` stored in the two-arm pkls (importance sampling of e.g. release "
             "transitions); single-arm stays timestep.",
    )
    parser.add_argument("--singlearm-data-fraction", type=float, default=1.0,
                        help="Fraction of manifest single-arm train files to use; 0 disables the single-arm objective.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2,
        help="AdamW weight decay for the trainable Coord head (default preserves legacy runs).",
    )
    parser.add_argument("--head-d-model", type=int, default=128)
    parser.add_argument("--head-n-heads", type=int, default=4)
    parser.add_argument("--head-depth", type=int, default=3)
    parser.add_argument("--head-dim-feedforward", type=int, default=512)
    parser.add_argument("--d-base-drop-prob", type=float, default=0.1)
    parser.add_argument(
        "--use-side-net", action=argparse.BooleanOptionalAction, default=True,
        help="Use the legacy side CNN (enabled by default).",
    )
    parser.add_argument("--side-net-input-size", type=int, default=128,
                        help="Input resolution for the side CNN; default preserves legacy 128px behavior.")
    parser.add_argument("--side-net-fusion", choices=["add", "concat"], default="add",
                        help="How side CNN tokens condition the head. 'add' is legacy behavior.")
    parser.add_argument("--side-net-tokens-per-camera", type=int, default=0,
                        help="Side CNN tokens per camera. 0 = base tokens_per_camera; use 64 for an 8x8 side stream.")
    parser.add_argument("--decoder-execution",
                        choices=[LEGACY_DECODER_EXECUTION, ALL_BLOCKS_DECODER_EXECUTION],
                        default=LEGACY_DECODER_EXECUTION)
    parser.add_argument("--decoder-conditioning",
                        choices=[DECODER_CONDITIONING_POOLED, DECODER_CONDITIONING_CROSS_ATTN],
                        default=DECODER_CONDITIONING_POOLED)
    parser.add_argument(
        "--plan-memory-mode", choices=sorted(VALID_PLAN_MEMORY_MODES),
        default=PLAN_MEMORY_ADDITIVE,
        help=("How D_base enters the coordination head: legacy additive, "
              "positional plan-memory replacement, or the dual path."),
    )
    parser.add_argument("--num-arms", type=int, default=2, help="Number of agents per multi-arm rollout.")
    parser.add_argument("--twoarm-modes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _train_utils.add_train_loop_args(parser, default_epochs=500)
    parser.set_defaults(max_train_steps=300000, checkpoint_every_steps=10000)
    add_wandb_args(parser)
    return parser.parse_args()


def _parse_cutout(specs):
    out = {}
    for spec in specs or []:
        agent, box = spec.split(":")
        out[int(agent)] = tuple(float(v) for v in box.split(","))
        assert len(out[int(agent)]) == 4, spec
    return out


def _loader(dataset, args, *, shuffle, drop_last, stream, sampler=None):
    kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = not args.no_persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    kwargs.update(_train_utils.dataloader_seed_kwargs(args.training_seed, stream=stream))
    return DataLoader(**kwargs)


def _under(path, root):
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)


def _load_exact_manifest(args):
    if args.manifest_path:
        if args.subset_stats_path:
            raise ValueError("Use only one of --manifest-path and --subset-stats-path")
        if args.singlearm_data_fraction != 1.0:
            raise ValueError("Direct exact manifests require --singlearm-data-fraction 1.0")
        with open(args.manifest_path, "r", encoding="utf-8") as stream:
            schema = json.load(stream).get("schema")
        loader = load_lowvar_combined_manifest if schema in (LOWVAR_COMBINED_SCHEMA, HIGHVAR_V3_SCHEMA) else load_exact_manifest
        exact = loader(
            args.manifest_path, multiarm_root=args.twoarm_dir, singlearm_roots=args.singlearm_dirs
        )
        manifest = {
            **exact,
            "singlearm_data_fraction": 1.0,
            "twoarm_train_files": exact["multiarm_train_files"],
            "twoarm_validation_files": [],
        }
        return manifest, list(exact["multiarm_train_files"]), []
    if not args.subset_stats_path:
        raise ValueError("One of --manifest-path or --subset-stats-path is required")
    with open(args.subset_stats_path, "rb") as f:
        source = pkl.load(f)
    required = (
        "twoarm_selected_rollout_files",
        "singlearm_selected_rollout_files",
        "twoarm_validation_rollout_files",
        "singlearm_validation_rollout_files",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"Subset stats missing exact manifest fields: {missing}")

    ta_train_abs = list(source["twoarm_selected_rollout_files"])
    ta_val_abs = list(source["twoarm_validation_rollout_files"])
    sa_train = list(source["singlearm_selected_rollout_files"])
    sa_val = list(source["singlearm_validation_rollout_files"])
    if any(not _under(path, args.twoarm_dir) for path in ta_train_abs + ta_val_abs):
        raise ValueError("Two-arm manifest contains files outside --twoarm-dir")
    if any(not any(_under(path, root) for root in args.singlearm_dirs) for path in sa_train + sa_val):
        raise ValueError("Single-arm manifest contains files outside --singlearm-dirs")

    if not (0.0 <= args.singlearm_data_fraction <= 1.0):
        raise ValueError(
            f"--singlearm-data-fraction must be in [0, 1], got {args.singlearm_data_fraction}"
        )
    if args.singlearm_data_fraction < 1.0:
        if args.singlearm_data_fraction == 0.0:
            sa_train = []
        else:
            sa_train = _train_utils.subsample_rollout_files(
                sorted(sa_train), args.singlearm_data_fraction, int(source.get("data_subset_seed", 0))
            )

    manifest = {
        "source_stats_path": os.path.abspath(args.subset_stats_path),
        "data_subset_seed": int(source.get("data_subset_seed", 0)),
        "singlearm_data_fraction": float(args.singlearm_data_fraction),
        "twoarm_train_files": ta_train_abs,
        "singlearm_train_files": sa_train,
        "twoarm_validation_files": ta_val_abs,
        "singlearm_validation_files": sa_val,
    }
    return manifest, ta_train_abs, ta_val_abs


def _to_device(batch, device, num_cameras):
    img_eih, img_shoulder, actions = batch
    eih = None if num_cameras == 1 else img_eih.to(device, non_blocking=True)
    return (
        eih,
        img_shoulder.to(device, non_blocking=True),
        actions.to(device, non_blocking=True),
    )


def _module_state_sha256(module: torch.nn.Module) -> str:
    """Hash parameters/buffers without serializing optimizer or device state."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    capacity_ablation = load_capacity_ablation_job_contract(args.capacity_job_contract, method="coord")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be nonnegative")
    _train_utils.seed_training(args.training_seed)
    if args.max_train_steps <= 0:
        raise ValueError("Mixed training is step-based; set --max-train-steps > 0")
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(args.base_stats_path, "rb") as f:
        base_stats = pkl.load(f)
    frame_offsets = frame_offsets_from_stats(base_stats)
    if args.frame_offsets is not None:
        requested_offsets = normalize_frame_offsets(args.frame_offsets)
        if requested_offsets != frame_offsets:
            raise ValueError(
                "--frame-offsets does not match frozen base stats: "
                f"requested={list(requested_offsets)}, base={list(frame_offsets)}"
            )
    manifest, ta_train_names, ta_val_names = _load_exact_manifest(args)

    horizon = int(base_stats.get("horizon", 20))
    num_cameras = int(base_stats.get("num_cameras", 2))
    sigma_data = float(base_stats.get("sigma_data", 1.0))
    base_d_model = int(base_stats.get("d_model", 256))
    base_n_heads = int(base_stats.get("n_heads", 4))
    base_depth = int(base_stats.get("depth", 3))
    base_dim_ff = int(base_stats.get("dim_feedforward", 1024))
    backbone = str(base_stats.get("backbone", "resnet18"))

    ta_dataset = TwoArmE2EImageDataset(
        rollout_dir=args.twoarm_dir,
        base_action_mean=base_stats["action_mean"],
        base_action_std=base_stats["action_std"],
        horizon=horizon,
        augment=not args.no_augment,
        other_arm_cutout=_parse_cutout(args.other_arm_cutout),
        other_arm_cutout_prob=args.other_arm_cutout_prob,
        modes=args.twoarm_modes,
        num_arms=args.num_arms,
        rollout_files=ta_train_names,
        frame_offsets=frame_offsets,
    )
    sa_dataset = None
    sa_loader = None
    if manifest["singlearm_train_files"] and args.singlearm_weight > 0.0:
        sa_dataset = SingleArmShoulderE2EDataset(
            rollout_dirs=args.singlearm_dirs,
            horizon=horizon,
            augment=not args.no_augment,
            rollout_files=manifest["singlearm_train_files"],
            frame_offsets=frame_offsets,
        )
        # Both objectives must operate in the frozen base policy's action space.
        sa_dataset.action_mean = base_stats["action_mean"]
        sa_dataset.action_std = base_stats["action_std"]
    sampling_summaries = {}
    ta_sampler = None
    sa_sampler = None
    if args.sampling_policy == "hierarchical":
        ta_sampler, sampling_summaries["multiarm"] = make_hierarchical_sampler(
            ta_dataset, seed=args.training_seed, stream=100
        )
        if sa_dataset is not None:
            sa_sampler, sampling_summaries["singlearm"] = make_hierarchical_sampler(
                sa_dataset, seed=args.training_seed, stream=101
            )
    elif args.sampling_policy == "weighted":
        w = torch.as_tensor(ta_dataset.sample_weights, dtype=torch.double)
        gen = torch.Generator()
        gen.manual_seed(int(args.training_seed) + 1_000_003 * 100)
        ta_sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=gen)
        n_up = int((w > 1).sum())
        sampling_summaries["multiarm"] = {
            "policy": "weighted", "samples": int(len(w)), "upweighted_samples": n_up,
            "weight_max": float(w.max()), "effective_upweighted_share": float(w[w > 1].sum() / w.sum()),
        }
        print(f"Weighted two-arm sampling: {n_up}/{len(w)} samples upweighted (max w={float(w.max()):.0f}), "
              f"effective share {float(w[w > 1].sum() / w.sum()):.1%}")
    ta_loader = _loader(
        ta_dataset, args, shuffle=True, drop_last=True, stream=0, sampler=ta_sampler
    )
    sa_loader = None if sa_dataset is None else _loader(
        sa_dataset, args, shuffle=True, drop_last=True, stream=1, sampler=sa_sampler
    )

    ta_val_loader = None
    if ta_val_names:
        ta_val_dataset = TwoArmE2EImageDataset(
            rollout_dir=args.twoarm_dir,
            base_action_mean=base_stats["action_mean"],
            base_action_std=base_stats["action_std"],
            horizon=horizon,
            augment=False,
            modes=args.twoarm_modes,
            num_arms=args.num_arms,
            rollout_files=ta_val_names,
            frame_offsets=frame_offsets,
        )
        ta_val_loader = _loader(ta_val_dataset, args, shuffle=False, drop_last=False, stream=2)

    base_model = ImageConditional_ODE(
        x_dim=7, sigma_data=sigma_data, d_model=base_d_model,
        n_heads=base_n_heads, depth=base_depth,
        dim_feedforward=base_dim_ff, horizon=horizon, device=device,
        num_cameras=num_cameras, backbone=backbone,
        frame_offsets=frame_offsets,
    )
    if not base_model.load(args.base_model_path):
        raise FileNotFoundError(args.base_model_path)
    for frozen_module in (base_model.F, base_model.F_ema):
        for parameter in frozen_module.parameters():
            parameter.requires_grad = False
    base_model.F.eval()
    base_model.F_ema.eval()
    frozen_base_hashes = {
        "base_model_file_sha256": _sha256_file(args.base_model_path),
        "base_stats_file_sha256": _sha256_file(args.base_stats_path),
        "base_F_state_sha256": _module_state_sha256(base_model.F),
        "base_F_ema_state_sha256": _module_state_sha256(base_model.F_ema),
    }

    def verify_frozen_base() -> None:
        if any(
            parameter.requires_grad
            for frozen_module in (base_model.F, base_model.F_ema)
            for parameter in frozen_module.parameters()
        ):
            raise RuntimeError("frozen base unexpectedly has a trainable parameter")
        observed = {
            "base_model_file_sha256": _sha256_file(args.base_model_path),
            "base_stats_file_sha256": _sha256_file(args.base_stats_path),
            "base_F_state_sha256": _module_state_sha256(base_model.F),
            "base_F_ema_state_sha256": _module_state_sha256(base_model.F_ema),
        }
        if observed != frozen_base_hashes:
            raise RuntimeError(
                "frozen base/stats changed during coordination-head-only training"
            )

    side_net_tokens_per_camera = (
        base_model.F.tokens_per_camera if args.side_net_tokens_per_camera <= 0
        else int(args.side_net_tokens_per_camera)
    )
    coord_head = ImageCoordinationHead(
        x_dim=7, base_d_model=base_d_model,
        d_model=args.head_d_model, n_heads=args.head_n_heads,
        depth=args.head_depth, dim_feedforward=args.head_dim_feedforward,
        horizon=horizon, sigma_data=sigma_data, lr=args.lr,
        weight_decay=args.weight_decay,
        num_cameras=num_cameras,
        tokens_per_camera=base_model.F.tokens_per_camera * len(frame_offsets),
        d_base_drop_prob=args.d_base_drop_prob,
        use_side_net=args.use_side_net,
        side_net_input_size=args.side_net_input_size,
        side_net_fusion=args.side_net_fusion,
        side_net_tokens_per_camera=side_net_tokens_per_camera,
        decoder_execution=args.decoder_execution,
        decoder_conditioning=args.decoder_conditioning,
        plan_memory_mode=args.plan_memory_mode,
        frame_offsets=frame_offsets,
    ).to(device)

    frozen_base_f_params = sum(parameter.numel() for parameter in base_model.F.parameters())
    coord_head_f_params = sum(parameter.numel() for parameter in coord_head.F.parameters())
    capacity_ablation_payload = None
    if capacity_ablation is not None:
        expected = capacity_ablation["model_counts"]
        observed = {"frozen_base_F": frozen_base_f_params, "trainable_head_F": coord_head_f_params, "deployed_F": frozen_base_f_params + coord_head_f_params}
        if expected != observed:
            raise ValueError(f"capacity-ablation parameter mismatch: expected={expected}, observed={observed}")
        capacity_ablation_payload = dict(capacity_ablation)
        capacity_ablation_payload["observed_model_counts"] = observed

    stats = {
        "training_seed": int(args.training_seed),
        "pipeline": "mixed_data_legacy_coordination_head",
        "base_model_path": os.path.abspath(args.base_model_path),
        "base_stats_path": os.path.abspath(args.base_stats_path),
        "action_mean": base_stats["action_mean"],
        "action_std": base_stats["action_std"],
        "sigma_data": sigma_data,
        "horizon": horizon,
        "num_cameras": num_cameras,
        "frame_offsets": list(frame_offsets),
        "backbone": backbone,
        "base_d_model": base_d_model,
        "head_d_model": args.head_d_model,
        "head_n_heads": args.head_n_heads,
        "head_depth": args.head_depth,
        "head_dim_feedforward": args.head_dim_feedforward,
        "tokens_per_camera": base_model.F.tokens_per_camera * len(frame_offsets),
        "spatial_tokens_per_camera": base_model.F.tokens_per_camera,
        "use_side_net": args.use_side_net,
        "side_net_input_size": int(args.side_net_input_size),
        "side_net_fusion": args.side_net_fusion,
        "side_net_tokens_per_camera": int(side_net_tokens_per_camera),
        "d_base_drop_prob": args.d_base_drop_prob,
        "decoder_execution": args.decoder_execution,
        "decoder_conditioning": args.decoder_conditioning,
        "plan_memory_mode": args.plan_memory_mode,
        "frozen_base_hashes": frozen_base_hashes,
        "singlearm_weight": args.singlearm_weight,
        "singlearm_data_fraction": args.singlearm_data_fraction,
        "lr": args.lr,
        "weight_decay": float(args.weight_decay),
        "ema_decay": 0.999,
        "training_metric_ema_decay": 0.99,
        "augment": not args.no_augment,
        "other_arm_cutout": args.other_arm_cutout,
        "other_arm_cutout_prob": args.other_arm_cutout_prob,
        "max_train_steps": args.max_train_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "num_arms": int(args.num_arms),
        "data_policy": (
            "sa_matched_density"
            if manifest.get("schema") in {
                "threearm_realign_v2_exact_manifest.v1",
                "placewipe_direction_exact_manifest.v1",
                "placewipe_hardware_exact_manifest.v1",
            }
            else None
        ),
        "per_domain_batch_size": int(args.batch_size),
        "domain_batching": "one_multiarm_plus_one_singlearm_batch_per_optimizer_step",
        "sampling_policy": args.sampling_policy,
        "sampling_hierarchy": (
            HIERARCHY if args.sampling_policy == "hierarchical"
            else ("frame_weights importance sampling (two-arm)" if args.sampling_policy == "weighted"
                  else "timestep-indexed shuffle")
        ),
        "effective_sampling_weights": sampling_summaries,
        "multiarm_loss_weight": 1.0,
        "dataset_manifest": manifest,
        "n_params": int(frozen_base_f_params + coord_head_f_params),
        "n_params_trainable": int(coord_head_f_params),
        "capacity_ablation": capacity_ablation_payload,
    }
    stats_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_stats.pkl")
    with open(stats_path, "wb") as f:
        pkl.dump(stats, f)
    manifest_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_data_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    coord_head.checkpoint_metadata = {
        "pipeline": stats["pipeline"],
        "decoder_execution": args.decoder_execution,
        "decoder_conditioning": args.decoder_conditioning,
        "plan_memory_mode": args.plan_memory_mode,
        "frozen_base_hashes": frozen_base_hashes,
        "dataset_manifest": manifest,
        "singlearm_weight": args.singlearm_weight,
        "weight_decay": float(args.weight_decay),
        "sampling_policy": args.sampling_policy,
        "sampling_hierarchy": stats["sampling_hierarchy"],
        "ema_decay": 0.999,
        "training_metric_ema_decay": 0.99,
        "frame_offsets": list(frame_offsets),
    }

    init_wandb(args, extra_config={
        "pipeline": stats["pipeline"],
        "decoder_execution": args.decoder_execution,
        "decoder_conditioning": args.decoder_conditioning,
        "plan_memory_mode": args.plan_memory_mode,
        "twoarm_rollouts": len(manifest["twoarm_train_files"]),
        "singlearm_rollouts": len(manifest["singlearm_train_files"]),
        "singlearm_weight": args.singlearm_weight,
        "weight_decay": float(args.weight_decay),
        "sampling_policy": args.sampling_policy,
        "sampling_hierarchy": stats["sampling_hierarchy"],
        "singlearm_data_fraction": args.singlearm_data_fraction,
        "augment": not args.no_augment,
        "frame_offsets": list(frame_offsets),
    })

    sa_iter = None if sa_loader is None else iter(sa_loader)

    def train_step_fn(twoarm_batch):
        nonlocal sa_iter
        twoarm_device_batch = _to_device(twoarm_batch, device, num_cameras)
        if sa_loader is None:
            loss, grad_norm = coord_head.update(
                twoarm_device_batch[2], twoarm_device_batch[0], twoarm_device_batch[1], base_model
            )
            return loss, grad_norm
        try:
            singlearm_batch = next(sa_iter)
        except StopIteration:
            sa_iter = iter(sa_loader)
            singlearm_batch = next(sa_iter)
        return coord_head.update_mixed(
            twoarm_device_batch,
            _to_device(singlearm_batch, device, num_cameras),
            base_model,
            singlearm_weight=args.singlearm_weight,
        )

    def val_step_fn(batch, generator, sigma):
        eih, shoulder, actions = _to_device(batch, device, num_cameras)
        loss = coord_head.validation_loss(
            actions, eih, shoulder, base_model,
            generator=generator, sigma=sigma,
        )
        return loss, actions.shape[0]

    def save_fn(path):
        verify_frozen_base()
        coord_head.save(path)
        contract_path = write_training_contract(
            args=args, stats=stats, checkpoint_path=path, stats_path=stats_path,
            producer_script=__file__, repo_root=REPO_ROOT,
        )
        print(f"Saved experiment contract: {contract_path}")

    history = _train_utils.train_step_loop(
        args, ta_loader, train_step_fn, save_fn,
        checkpoint_dir=args.checkpoint_dir,
        prefix=args.checkpoint_prefix,
        val_loader=ta_val_loader,
        val_step_fn=val_step_fn,
        sigma_grid=_train_utils.build_sigma_grid(base_model, args.val_sigma_grid_size),
        log_fn=log_metrics,
        device=str(device),
    )
    final_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_final.pt")
    save_fn(final_path)
    print(f"Completed {len(history)} logged training events")
    finish_wandb()


if __name__ == "__main__":
    main()

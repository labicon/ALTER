"""Train a fresh policy from scratch on two-arm task data.

Unlike the fine-tuning scripts (coordination head, LoRA, OFT) which adapt a
frozen single-arm base model, this script trains a brand-new ImageConditional_ODE
directly on the two-arm dataset.  Action stats (mean/std) are computed from
the two-arm data itself.

Pipeline:
    1. Data generation script                               — two-arm rollouts
    2. THIS SCRIPT                                          — train from scratch on two-arm data
"""

import argparse
import json
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.experiment_contract import write_training_contract

from envs.arm.train import _train_utils
from envs.arm.train.twoarm_dataset import TwoArmE2EImageDataset
from src.image_diffusion import ImageConditional_ODE
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train fresh policy from scratch on two-arm task data."
    )
    parser.add_argument(
        "--rollout-dir", type=str,
        default="envs/arm/data_gen/rollouts/horizontal_cube_twoarm_redfront_directgrasp_shoulder",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default="checkpoints/arm/twoarm_fromscratch_e2e_shoulder",
    )
    parser.add_argument(
        "--checkpoint-prefix", type=str, default="twoarm_fromscratch",
    )
    parser.add_argument(
        "--stats-path", type=str,
        default="stats/arm/twoarm_fromscratch_e2e_shoulder_stats.pkl",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--modes", type=int, nargs="+", default=[2, 3, 4, 5],
        help="Rollout modes to include in training (e.g. --modes 2 3 4 5).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--train-rollouts-per-mode",
        type=int,
        default=0,
        help="If >0, use up to this many training rollout files per mode after validation holdout.",
    )
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "dinov2"],
                        help="Image encoder backbone.")
    parser.add_argument("--shoulder-only", action="store_true",
                        help="Train with shoulder camera only (no eye-in-hand).")
    parser.add_argument("--num-arms", type=int, default=2,
                        help="Number of arms in each rollout (action dim = 7 * num_arms).")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    _train_utils.add_train_loop_args(parser, default_epochs=500)
    add_wandb_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _train_utils.seed_training(args.training_seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    stats_dir = os.path.dirname(args.stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset with dummy stats, then compute real stats
    # ------------------------------------------------------------------
    dummy_mean = np.zeros(7, dtype=np.float32)
    dummy_std = np.ones(7, dtype=np.float32)

    available_files = _train_utils.enumerate_pkl_files(args.rollout_dir, modes=args.modes)
    train_pool, val_files = _train_utils.split_train_val_files(
        available_files,
        holdout_fraction=args.val_holdout_fraction,
        max_val_files=args.val_max_rollout_files,
        seed=args.val_subset_seed,
    )
    if args.data_fraction < 1.0:
        train_pool = _train_utils.subsample_rollout_files(
            train_pool, args.data_fraction, args.data_subset_seed
        )
    if args.train_rollouts_per_mode > 0:
        train_pool = _train_utils.select_rollout_files_per_mode(
            train_pool, args.train_rollouts_per_mode, args.data_subset_seed
        )

    dataset = TwoArmE2EImageDataset(
        rollout_dir=args.rollout_dir,
        base_action_mean=dummy_mean,
        base_action_std=dummy_std,
        horizon=args.horizon,
        augment=not args.no_augment,
        modes=args.modes,
        num_arms=args.num_arms,
        rollout_files=train_pool,
    )

    # Compute action stats from the two-arm data
    all_ac = np.concatenate(dataset.all_actions, axis=0)
    action_mean = np.mean(all_ac, axis=0)
    action_std = np.std(all_ac, axis=0)
    action_std[action_std < 1e-6] = 1.0

    # Update dataset with real stats
    dataset.action_mean = action_mean
    dataset.action_std = action_std

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        **_train_utils.dataloader_seed_kwargs(args.training_seed, stream=0),
    )

    val_dataset = None
    val_loader = None
    if val_files:
        val_dataset = TwoArmE2EImageDataset(
            rollout_dir=args.rollout_dir,
            base_action_mean=action_mean,
            base_action_std=action_std,
            horizon=args.horizon,
            augment=False,
            modes=args.modes,
            num_arms=args.num_arms,
            rollout_files=val_files,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            **_train_utils.dataloader_seed_kwargs(args.training_seed, stream=1),
        )
        print(f"Validation rollout files ({len(val_files)}): {val_files}")
    else:
        print("No held-out validation rollout files available.")

    # Compute sigma_data from normalized actions
    all_ac_norm = (all_ac - action_mean) / action_std
    sigma_data = float(all_ac_norm.std())
    print(f"sigma_data={sigma_data:.4f}, dataset size={len(dataset)}")

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    num_cameras = 1 if args.shoulder_only else 2
    camera_views = ["shoulder"] if args.shoulder_only else ["eye_in_hand", "shoulder"]

    model = ImageConditional_ODE(
        x_dim=7,
        sigma_data=sigma_data,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dim_feedforward=args.dim_feedforward,
        horizon=args.horizon,
        device=device,
        N=args.n_steps,
        lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob,
        num_cameras=num_cameras,
        backbone=args.backbone,
    )

    n_params_total = sum(p.numel() for p in model.F.parameters())
    n_params_trainable = sum(p.numel() for p in model.F.parameters() if p.requires_grad)
    print(f"Model params: {n_params_total/1e6:.1f}M total, {n_params_trainable/1e6:.1f}M trainable")

    # ------------------------------------------------------------------
    # Save stats
    # ------------------------------------------------------------------
    prefix = args.checkpoint_prefix
    stats = {
        "training_seed": int(args.training_seed),
        "action_mean": action_mean,
        "action_std": action_std,
        "horizon": int(args.horizon),
        "sigma_data": sigma_data,
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "depth": int(args.depth),
        "dim_feedforward": int(args.dim_feedforward),
        "num_cameras": num_cameras,
        "pipeline": "twoarm_fromscratch_e2e_shoulder",
        "n_params": int(n_params_total),
        "n_params_trainable": int(n_params_trainable),
        "cfg_drop_prob": float(args.cfg_drop_prob),
        "rollout_dir": args.rollout_dir,
        "camera_views": camera_views,
        "backbone": args.backbone,
        "augment": not args.no_augment,
        "data_fraction": args.data_fraction,
        "data_subset_seed": args.data_subset_seed,
        "train_rollouts_per_mode": int(args.train_rollouts_per_mode),
        "max_train_steps": int(args.max_train_steps),
        "checkpoint_every_steps": int(args.checkpoint_every_steps),
        "val_max_rollout_files": int(args.val_max_rollout_files),
        "val_batches": int(args.val_batches),
        "val_subset_seed": int(args.val_subset_seed),
        "available_rollout_files": dataset.available_pkl_files,
        "selected_rollout_files": dataset.selected_pkl_files,
        "validation_rollout_files": val_files,
    }
    with open(args.stats_path, "wb") as f:
        pkl.dump(stats, f)
    print(f"Saved stats: {args.stats_path}")

    subset_manifest_path = os.path.join(args.checkpoint_dir, f"{prefix}_data_subset.json")
    with open(subset_manifest_path, "w") as f:
        json.dump({
            "rollout_dir": args.rollout_dir,
            "augment": not args.no_augment,
            "data_fraction": args.data_fraction,
            "data_subset_seed": args.data_subset_seed,
            "train_rollouts_per_mode": int(args.train_rollouts_per_mode),
            "available_rollout_files": dataset.available_pkl_files,
            "selected_rollout_files": dataset.selected_pkl_files,
            "validation_rollout_files": val_files,
        }, f, indent=2)
    print(f"Saved data subset manifest: {subset_manifest_path}")

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    init_wandb(args, extra_config={
        "sigma_data": sigma_data,
        "n_params": n_params_total,
        "n_params_trainable": n_params_trainable,
        "dataset_size": len(dataset),
        "pipeline": "twoarm_fromscratch_e2e_shoulder",
        "camera_views": camera_views,
        "num_cameras": num_cameras,
        "backbone": args.backbone,
        "augment": not args.no_augment,
        "data_fraction": args.data_fraction,
        "data_subset_seed": args.data_subset_seed,
        "train_rollouts_per_mode": int(args.train_rollouts_per_mode),
        "available_rollout_files": len(dataset.available_pkl_files),
        "selected_rollout_files": len(dataset.selected_pkl_files),
        "max_train_steps": args.max_train_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "validation_rollout_files": len(val_files),
    })

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("Training from scratch on two-arm data...")

    def train_step_fn(batch):
        img_eih, img_shoulder, actions = batch
        img_shoulder = img_shoulder.to(device)
        actions = actions.to(device)
        eih = None if args.shoulder_only else img_eih.to(device)
        return model.update(actions, eih, img_shoulder)

    def val_step_fn(batch, generator, sigma):
        img_eih, img_shoulder, actions = batch
        img_shoulder = img_shoulder.to(device)
        actions = actions.to(device)
        eih = None if args.shoulder_only else img_eih.to(device)
        loss = model.validation_loss(
            actions, eih, img_shoulder, generator=generator, sigma=sigma
        )
        return loss, actions.shape[0]

    def save_fn(path):
        model.save(path)
        contract_path = write_training_contract(
            args=args, stats=stats, checkpoint_path=path, stats_path=args.stats_path,
            producer_script=__file__, repo_root=REPO_ROOT,
        )
        print(f"Saved experiment contract: {contract_path}")

    sigma_grid = _train_utils.build_sigma_grid(model, args.val_sigma_grid_size)
    loop_fn = (
        _train_utils.train_step_loop if args.max_train_steps > 0
        else _train_utils.train_epoch_loop
    )
    history = loop_fn(
        args,
        dataloader,
        train_step_fn,
        save_fn,
        checkpoint_dir=args.checkpoint_dir,
        prefix=prefix,
        val_loader=val_loader,
        val_step_fn=val_step_fn,
        sigma_grid=sigma_grid,
        log_fn=log_metrics,
        device=str(device),
    )

    final_ckpt = os.path.join(args.checkpoint_dir, f"{prefix}_final.pt")
    save_fn(final_ckpt)
    print(f"Saved final model: {final_ckpt}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if history:
        train_xs, train_losses, train_grads, val_xs, val_losses = (
            _train_utils.history_to_curves(history)
        )
        xlabel = "Gradient Step" if args.max_train_steps > 0 else "Epoch"
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_xs, train_losses, label="Training Loss", alpha=0.8)
        if val_xs:
            plt.plot(val_xs, val_losses, label="Validation Loss",
                     marker="o", linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.title("From-Scratch Two-Arm Training vs Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_xs, train_grads, color="orange")
        plt.xlabel(xlabel)
        plt.ylabel("Gradient Norm")
        plt.title("Average Gradient Norm")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
        plt.savefig(plot_path)
        print(f"Saved training plot: {plot_path}")

    finish_wandb()


if __name__ == "__main__":
    main()

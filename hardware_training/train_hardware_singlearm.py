#!/usr/bin/env python3
"""Train a Co-Diff single-arm shoulder-only diffusion policy on real xArm hardware data.

Sibling of Co-Diff/envs/arm/train/train_singlearm_mixedfront_e2e_shoulder.py.
Reuses ImageConditional_ODE from Co-Diff but supplies its own dataset class
that loads only the shoulder camera (no eye-in-hand).
"""

import argparse
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CO_DIFF_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ".."
))
if CO_DIFF_ROOT not in sys.path:
    sys.path.insert(0, CO_DIFF_ROOT)

from envs.arm.train import _train_utils  # noqa: E402
from src.image_diffusion import ImageConditional_ODE  # noqa: E402
from utils.wandb_utils import (  # noqa: E402
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)


class HardwareSingleArmShoulderDataset(Dataset):
    """Load real-arm rollouts as .pkl files. Shoulder camera only.

    Each .pkl must contain:
        actions_single_arm  (T, 7) float
        camera_obs_shoulder (T, H, W, 3) uint8

    Rollout IDs are absolute paths (since multiple directories may be passed).
    Pass `rollout_files` (a list of absolute paths) to load an explicit subset,
    e.g. a held-out train/val split carved by `_train_utils.split_train_val_files`.
    """

    def __init__(
        self,
        rollout_dirs,
        horizon: int = 20,
        augment: bool = True,
        data_fraction: float = 1.0,
        data_subset_seed: int = 0,
        rollout_files=None,
    ):
        if isinstance(rollout_dirs, str):
            rollout_dirs = [rollout_dirs]
        self.horizon = horizon
        self.augment = augment
        self.data_fraction = float(data_fraction)
        self.data_subset_seed = int(data_subset_seed)
        self.train_transform = transforms.RandomResizedCrop(
            128, scale=(0.8, 1.0), antialias=True
        )

        all_paths = []
        for d in rollout_dirs:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Rollout dir not found: {d}")
            files = sorted(f for f in os.listdir(d) if f.endswith(".pkl"))
            if not files:
                raise FileNotFoundError(f"No .pkl files in {d}")
            all_paths.extend(os.path.join(d, f) for f in files)
        all_paths = sorted(all_paths)
        self.available_pkl_files = list(all_paths)

        if rollout_files is not None:
            requested = set(rollout_files)
            missing = sorted(requested.difference(self.available_pkl_files))
            if missing:
                raise FileNotFoundError(
                    f"Requested rollout files not available: {missing[:5]}..."
                )
            rollout_paths = sorted(p for p in self.available_pkl_files if p in requested)
            if not rollout_paths:
                raise FileNotFoundError("No rollout files left after rollout_files filter.")
        else:
            if not (0.0 < self.data_fraction <= 1.0):
                raise ValueError(f"data_fraction must be in (0, 1], got {self.data_fraction}")
            if self.data_fraction < 1.0:
                rollout_paths = _train_utils.subsample_rollout_files(
                    all_paths, self.data_fraction, self.data_subset_seed
                )
            else:
                rollout_paths = list(all_paths)
        self.selected_pkl_files = list(rollout_paths)

        self.all_imgs_shoulder = []
        self.all_actions = []
        self.index_map = []

        for path in rollout_paths:
            try:
                with open(path, "rb") as f:
                    rollout = pkl.load(f)
            except Exception as exc:
                print(f"Skipping {path}: load error ({exc})")
                continue

            try:
                actions = np.asarray(rollout["actions_single_arm"], dtype=np.float32)
                imgs_shoulder = np.asarray(rollout["camera_obs_shoulder"], dtype=np.uint8)
            except KeyError as exc:
                print(f"Skipping {path}: missing key {exc}")
                continue

            if actions.ndim != 2 or actions.shape[1] < 7:
                print(f"Skipping {path}: actions shape {actions.shape}")
                continue
            actions = actions[:, :7]

            T = min(len(actions), len(imgs_shoulder))
            if T < horizon:
                print(f"Skipping {path}: T={T} < horizon={horizon}")
                continue

            traj_idx = len(self.all_actions)
            self.all_actions.append(actions[:T])
            self.all_imgs_shoulder.append(imgs_shoulder[:T])
            for t in range(T):
                self.index_map.append((traj_idx, t))

        if not self.all_actions:
            raise RuntimeError("No valid trajectories loaded.")

        all_ac = np.concatenate(self.all_actions, axis=0)
        self.action_mean = np.mean(all_ac, axis=0)
        self.action_std = np.std(all_ac, axis=0)
        # Floor at 1e-2 so dims with ~zero variance (e.g. roll/yaw in our data) don't
        # blow up into std=1.0, which amplifies denoising noise 100x at inference.
        # For dims with real variance (xyz ~70-160, gripper ~137), the floor is a no-op.
        self.action_std = np.maximum(self.action_std, 1e-2)
        print(f"action_std (after flooring at 1e-2): {self.action_std.round(4)}")

        print(
            f"Loaded {len(self.all_actions)} trajectories, "
            f"{len(self.index_map)} samples, horizon={horizon}"
        )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        traj_idx, t = self.index_map[idx]
        actions = self.all_actions[traj_idx]
        T = len(actions)
        if t + self.horizon > T:
            chunk = actions[t:]
            pad_len = self.horizon - len(chunk)
            chunk = np.concatenate([chunk, np.tile(actions[-1], (pad_len, 1))], axis=0)
        else:
            chunk = actions[t : t + self.horizon]
        chunk = (chunk - self.action_mean) / self.action_std

        img = self.all_imgs_shoulder[traj_idx][t]
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        if self.augment:
            img_t = self.train_transform(img_t)
        else:
            img_t = transforms.functional.resize(img_t, [128, 128], antialias=True)
        return img_t, torch.FloatTensor(chunk)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dirs", type=str, nargs="+",
                   required=True)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=50,
                   help="Diffusion sampling steps (ODE integration), not training steps.")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dim-feedforward", type=int, default=1024)
    p.add_argument("--cfg-drop-prob", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "dinov2"])
    p.add_argument("--checkpoint-prefix", default="singlearm_hardware_shoulder")
    p.add_argument("--checkpoint-dir",
                   required=True)
    p.add_argument("--stats-path",
                   required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Step/epoch loop + held-out validation. Default to step mode (the hardware
    # convention) by setting --max-train-steps nonzero; pass --max-train-steps 0
    # --epochs N to fall back to epoch mode.
    _train_utils.add_train_loop_args(
        p, default_save_every=50, default_val_max=16, default_val_batches=20,
    )
    p.set_defaults(max_train_steps=200000, checkpoint_every_steps=10000,
                   log_every_steps=200)
    add_wandb_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    stats_dir = os.path.dirname(args.stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Held-out train/val split (carved BEFORE --data-fraction). Hardware pkls
    # have no mode<k> tags, so this is a plain deterministic file holdout.
    # ------------------------------------------------------------------
    available_paths = []
    for d in args.rollout_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Rollout dir not found: {d}")
        available_paths.extend(
            os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".pkl")
        )
    available_paths = sorted(available_paths)

    train_pool, val_files = _train_utils.split_train_val_files(
        available_paths,
        holdout_fraction=args.val_holdout_fraction,
        max_val_files=args.val_max_rollout_files,
        seed=args.val_subset_seed,
    )
    if args.data_fraction < 1.0:
        train_pool = _train_utils.subsample_rollout_files(
            train_pool, args.data_fraction, args.data_subset_seed
        )

    dataset = HardwareSingleArmShoulderDataset(
        rollout_dirs=args.rollout_dirs,
        horizon=args.horizon,
        augment=not args.no_augment,
        rollout_files=train_pool,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_files:
        val_dataset = HardwareSingleArmShoulderDataset(
            rollout_dirs=args.rollout_dirs,
            horizon=args.horizon,
            augment=False,
            rollout_files=val_files,
        )
        # Use training stats so val operates in the same normalized space.
        val_dataset.action_mean = dataset.action_mean
        val_dataset.action_std = dataset.action_std
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        print(f"Validation rollout files ({len(val_files)}).")
    else:
        print("No held-out validation rollout files available.")

    all_ac = np.concatenate(dataset.all_actions, axis=0)
    all_ac_norm = (all_ac - dataset.action_mean) / dataset.action_std
    sigma_data = float(all_ac_norm.std())
    print(f"sigma_data={sigma_data:.4f}, dataset size={len(dataset)}")

    model = ImageConditional_ODE(
        x_dim=7, sigma_data=sigma_data,
        d_model=args.d_model, n_heads=args.n_heads, depth=args.depth,
        dim_feedforward=args.dim_feedforward, horizon=args.horizon,
        device=device, N=args.n_steps, lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob, num_cameras=1, backbone=args.backbone,
    )

    n_params_total = sum(p.numel() for p in model.F.parameters())
    n_params_trainable = sum(p.numel() for p in model.F.parameters() if p.requires_grad)
    print(f"Model params: {n_params_total/1e6:.1f}M total, {n_params_trainable/1e6:.1f}M trainable")

    stats = {
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
        "horizon": int(args.horizon),
        "sigma_data": sigma_data,
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "depth": int(args.depth),
        "dim_feedforward": int(args.dim_feedforward),
        "num_cameras": 1,
        "pipeline": "e2e_image_singlearm_hardware_shoulder",
        "n_params": int(n_params_total),
        "n_params_trainable": int(n_params_trainable),
        "cfg_drop_prob": float(args.cfg_drop_prob),
        "rollout_dirs": list(args.rollout_dirs),
        "camera_views": ["shoulder"],
        "backbone": args.backbone,
        "max_train_steps": int(args.max_train_steps),
        "checkpoint_every_steps": int(args.checkpoint_every_steps),
        "val_holdout_fraction": float(args.val_holdout_fraction),
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

    init_wandb(args, extra_config={
        "sigma_data": sigma_data, "n_params": n_params_total,
        "n_params_trainable": n_params_trainable, "dataset_size": len(dataset),
        "pipeline": "e2e_image_singlearm_hardware_shoulder",
        "camera_views": ["shoulder"], "num_cameras": 1, "backbone": args.backbone,
        "validation_rollout_files": len(val_files),
    })

    prefix = args.checkpoint_prefix

    def train_step_fn(batch):
        img_shoulder, actions = batch
        img_shoulder = img_shoulder.to(device)
        actions = actions.to(device)
        return model.update(actions, None, img_shoulder)

    def val_step_fn(batch, generator, sigma):
        img_shoulder, actions = batch
        img_shoulder = img_shoulder.to(device)
        actions = actions.to(device)
        loss = model.validation_loss(
            actions, None, img_shoulder, generator=generator, sigma=sigma
        )
        return loss, actions.shape[0]

    def save_fn(path):
        model.save(path)

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
    model.save(final_ckpt)
    print(f"Saved final model: {final_ckpt}")

    if history:
        train_xs, train_losses, train_grads, val_xs, val_losses = (
            _train_utils.history_to_curves(history)
        )
        xlabel = "Gradient Step" if args.max_train_steps > 0 else "Epoch"
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_xs, train_losses, label="Training Loss", alpha=0.8)
        if val_xs:
            plt.plot(val_xs, val_losses, label="Validation Loss", marker="o", linewidth=2)
        plt.xlabel(xlabel); plt.ylabel("Loss")
        plt.title("Training vs Validation Loss"); plt.grid(True); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_xs, train_grads, color="orange")
        plt.xlabel(xlabel); plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm"); plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")

    finish_wandb()


if __name__ == "__main__":
    main()

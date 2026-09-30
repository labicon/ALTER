"""Train single-arm mixed-front (greenfront + redfront) e2e image-conditioned diffusion policy.

This trainer combines rollouts from both greenfront and redfront single-arm
environments to train one generalizable policy. It uses raw rollout images and
trains the ResNet + DiT jointly (no precomputed latent inputs).
"""

import argparse
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from envs.arm.train import _train_utils
from envs.arm.train.singlearm_frozen_base_manifest import load_frozen_base_manifest
from envs.arm.train.hierarchical_sampling import HIERARCHY, make_hierarchical_sampler
from envs.arm.train.temporal_images import prepare_temporal_views
from src.image_diffusion import ImageConditional_ODE
from src.temporal import normalize_frame_offsets
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_side(side):
    if side is None:
        return None
    side_s = str(side).lower()
    if side_s in ("left", "right"):
        return side_s
    return None


def _unique_values(values):
    unique = []
    for v in values:
        if not any(v == u for u in unique):
            unique.append(v)
    return unique


def build_task_profile(task_configs):
    cleaned = [_to_builtin(cfg) for cfg in task_configs if isinstance(cfg, dict)]
    if len(cleaned) == 0:
        return None, {}

    profile = {
        "source": "rollout_task_config",
        "num_with_task_config": int(len(cleaned)),
    }
    keys = [
        "obstacle_layout",
        "obstacle_y_offset",
        "mode_conditioned_single_obstacle",
        "waypoint_profile",
        "orientation_profile",
    ]
    for key in keys:
        vals = [cfg[key] for cfg in cleaned if key in cfg]
        if len(vals) == 0:
            continue
        uniq = _unique_values(vals)
        if len(uniq) == 1:
            profile[key] = uniq[0]
        else:
            profile[f"{key}_variants"] = uniq

    mode_to_side_map = {}
    for cfg in cleaned:
        side = _normalize_side(cfg.get("active_obstacle_side"))
        if side is None:
            continue
        try:
            mode_i = int(cfg.get("mode"))
        except Exception:
            continue
        mode_to_side_map[mode_i] = side

    if len(mode_to_side_map) > 0:
        profile["mode_to_side_map"] = dict(sorted(mode_to_side_map.items()))

    return profile, dict(sorted(mode_to_side_map.items()))


def active_robot_idx_for_mode(mode):
    if mode in (4, 5):
        return 1
    return 0


def extract_actions_single_arm(rollout, mode):
    if "actions_single_arm" in rollout:
        actions = np.asarray(rollout["actions_single_arm"], dtype=np.float32)
        if actions.ndim == 2 and actions.shape[1] >= 7:
            return actions[:, :7]

    if "actions" in rollout:
        actions = np.asarray(rollout["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise KeyError("actions has invalid shape")
        if actions.shape[1] >= 14:
            if mode in (4, 5):
                return actions[:, 7:14]
            return actions[:, 0:7]
        return actions[:, :7]

    raise KeyError("missing actions_single_arm/actions")


def pick_first_present(rollout, candidates):
    for key in candidates:
        if key in rollout:
            return np.asarray(rollout[key], dtype=np.uint8)
    raise KeyError(f"Missing keys. Tried: {candidates}")


def normalize_rollout_dirs(rollout_dirs):
    if isinstance(rollout_dirs, str):
        dirs = [rollout_dirs]
    else:
        dirs = [str(d) for d in rollout_dirs]

    normalized = []
    for rollout_dir in dirs:
        if not os.path.isdir(rollout_dir):
            raise FileNotFoundError(f"Rollout directory not found: {rollout_dir}")
        normalized.append(rollout_dir)

    if len(normalized) == 0:
        raise ValueError("At least one rollout directory is required.")
    return normalized


def discover_rollout_files(rollout_dir):
    """Return all rollout files below a configured task root."""
    return sorted(
        os.path.abspath(os.path.join(root, filename))
        for root, _, filenames in os.walk(rollout_dir)
        for filename in filenames
        if filename.endswith(".pkl")
    )


def rollout_task_mode_group(path, rollout_dirs, mode):
    """Stable task/mode bucket for balanced frozen-base sampling."""
    absolute = os.path.abspath(path)
    roots = [os.path.abspath(root) for root in rollout_dirs]
    matching = [root for root in roots if os.path.commonpath([absolute, root]) == root]
    if not matching:
        raise ValueError(f"rollout path lies outside configured roots: {path}")
    root = max(matching, key=len)
    relative_parts = os.path.relpath(absolute, root).split(os.sep)
    # Legacy examples are root/<task>/modeN/file; the tray auxiliary is
    # root/modeN/file.  Keep its own task bucket rather than allowing its
    # duration to decide how frequently it appears in training.
    if len(relative_parts) >= 3:
        task = relative_parts[0]
    else:
        root_name = os.path.basename(root)
        task = (
            root_name if root_name in {"place_return_neg_y", "place_return_pos_y", "wipe"}
            else "tray_drag"
        )
    return f"singlearm/{task}/mode{int(mode)}"


class SingleArmShoulderE2EDataset(Dataset):
    """Load single-arm rollouts with EIH + shoulder images and horizon chunks.

    Rollout IDs are absolute paths (since multiple directories may be passed).
    Use `available_pkl_files` for the full mode-filtered pool and
    `selected_pkl_files` for the actually-loaded subset.
    """

    @staticmethod
    def _mode_key(filename):
        return _train_utils.mode_key(filename)

    def __init__(
        self,
        rollout_dirs,
        horizon: int = 20,
        augment: bool = True,
        modes=None,
        data_fraction: float = 1.0,
        data_subset_seed: int = 0,
        rollout_files=None,
        frame_offsets=(0,),
    ):
        self.horizon = horizon
        self.augment = augment
        self.frame_offsets = normalize_frame_offsets(frame_offsets)
        self.rollout_dirs = normalize_rollout_dirs(rollout_dirs)
        self.data_fraction = float(data_fraction)
        self.data_subset_seed = int(data_subset_seed)

        all_paths = []
        counts_by_dir = {}
        for rollout_dir in self.rollout_dirs:
            files = discover_rollout_files(rollout_dir)
            if not files:
                raise FileNotFoundError(f"No pkl files found in {rollout_dir}")
            counts_by_dir[rollout_dir] = len(files)
            all_paths.extend(files)

        if modes is not None:
            mode_strs = {f"mode{m}" for m in modes}
            all_paths = [p for p in all_paths if any(ms in os.path.basename(p) for ms in mode_strs)]
            if not all_paths:
                raise FileNotFoundError(f"No pkl files match modes={modes} in {self.rollout_dirs}")

        all_paths = sorted(all_paths)
        self.available_pkl_files = list(all_paths)

        if rollout_files is not None:
            requested = set(rollout_files)
            missing = sorted(requested.difference(self.available_pkl_files))
            if missing:
                raise FileNotFoundError(
                    f"Requested rollout files not available: {missing[:5]}..."
                )
            rollout_paths = sorted([p for p in self.available_pkl_files if p in requested])
            if not rollout_paths:
                raise FileNotFoundError("No rollout files left after explicit rollout_files filter.")
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

        print(
            f"Found {len(self.available_pkl_files)} rollout files across "
            f"{len(self.rollout_dirs)} dirs; using {len(self.selected_pkl_files)} "
            f"(fraction={self.data_fraction}, subset_seed={self.data_subset_seed})"
        )

        self.all_imgs_eih = []
        self.all_imgs_shoulder = []
        self.all_actions = []
        self.index_map = []
        self.task_configs = []
        self.modes = []
        self.trajectory_group_keys = []
        self.trajectory_rollout_ids = []

        for path in rollout_paths:
            fname = os.path.basename(path)
            mode_key = _train_utils.mode_key(fname)
            try:
                mode = int(mode_key.replace("mode", "")) if mode_key != "mode_unknown" else 2
            except ValueError:
                mode = 2
            active_idx = active_robot_idx_for_mode(mode)

            try:
                with open(path, "rb") as f:
                    rollout = pkl.load(f)
            except Exception as exc:
                print(f"Skipping {path}: load error ({exc})")
                continue

            try:
                actions = extract_actions_single_arm(rollout, mode=mode)
                imgs_eih = pick_first_present(
                    rollout,
                    ["camera_obs", f"camera_obs{active_idx}", "camera_obs0", "camera_obs1"],
                )
                imgs_shoulder = pick_first_present(
                    rollout,
                    [
                        "camera_obs_shoulder",
                        f"camera_obs_shoulder{active_idx}",
                        "camera_obs_shoulder0",
                        "camera_obs_shoulder1",
                    ],
                )
            except Exception as exc:
                print(f"Skipping {path}: {exc}")
                continue

            T = min(len(actions), len(imgs_eih), len(imgs_shoulder))
            if T < horizon:
                continue

            traj_idx = len(self.all_actions)
            self.all_actions.append(np.asarray(actions[:T], dtype=np.float32))
            self.all_imgs_eih.append(np.asarray(imgs_eih[:T], dtype=np.uint8))
            self.all_imgs_shoulder.append(np.asarray(imgs_shoulder[:T], dtype=np.uint8))
            self.task_configs.append(_to_builtin(rollout.get("task_config")))
            self.modes.append(int(mode))
            self.trajectory_group_keys.append(
                rollout_task_mode_group(path, self.rollout_dirs, mode)
            )
            self.trajectory_rollout_ids.append(os.path.abspath(path))

            for t in range(T):
                self.index_map.append((traj_idx, t))

        if len(self.all_actions) == 0:
            raise RuntimeError("No valid trajectories loaded.")

        all_ac = np.concatenate(self.all_actions, axis=0)
        self.action_mean = np.mean(all_ac, axis=0)
        self.action_std = np.std(all_ac, axis=0)
        self.action_std[self.action_std < 1e-6] = 1.0

        print(
            f"Loaded {len(self.all_actions)} trajectories, {len(self.index_map)} samples, "
            f"horizon={horizon}, rollout_dirs={counts_by_dir}"
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

        img_eih, img_shoulder = prepare_temporal_views(
            self.all_imgs_eih[traj_idx],
            self.all_imgs_shoulder[traj_idx],
            t,
            self.frame_offsets,
            augment=self.augment,
        )

        return img_eih, img_shoulder, torch.FloatTensor(chunk)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train single-arm mixed-front (greenfront + redfront) e2e shoulder diffusion policy."
    )
    parser.add_argument(
        "--rollout-dirs",
        type=str,
        nargs="+",
        default=[
            "envs/arm/data_gen/rollouts/horizontal_cube_singlearm_greenfront_directgrasp_shoulder",
            "envs/arm/data_gen/rollouts/horizontal_cube_singlearm_redfront_directgrasp_shoulder",
        ],
        help="List of rollout directories to combine for training.",
    )
    parser.add_argument(
        "--manifest-path",
        help="Immutable frozen-base manifest; bypasses source splitting and requires all 800 approved files.",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true", help="Disable image augmentation.")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "dinov2"],
                        help="Image encoder backbone: 'resnet18' (trainable) or 'dinov2' (frozen pretrained).")
    parser.add_argument("--shoulder-only", action="store_true", help="Train with shoulder camera only (no eye-in-hand).")
    parser.add_argument(
        "--sampling-policy", choices=("timestep", "hierarchical"), default="timestep",
        help=("Within-loader sampling. 'timestep' draws uniformly over frames, so a "
              "source with more or longer rollouts dominates. 'hierarchical' gives "
              "equal mass to each task/mode, then rollout, then timestep."),
    )
    parser.add_argument(
        "--balanced-task-mode-sampling", action="store_true",
        help="deprecated compatibility alias for --sampling-policy hierarchical",
    )
    parser.add_argument(
        "--frame-offsets", type=int, nargs="+", default=[0],
        help="Non-negative image lookbacks, e.g. 15 0 conditions on [t-15, t].",
    )
    parser.add_argument(
        "--modes", type=int, nargs="+", default=None,
        help="Mode IDs to filter rollout pkls by. Pass nothing to use all modes.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/arm/singlearm_mixedfront_e2e_shoulder",
    )
    parser.add_argument(
        "--checkpoint-prefix", type=str, default="singlearm_mixedfront_e2e_shoulder",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="stats/arm/singlearm_mixedfront_e2e_shoulder_stats.pkl",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    _train_utils.add_train_loop_args(parser, default_epochs=400)
    add_wandb_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    # --training-seed was accepted but never applied here, unlike the combined
    # and coordination trainers. Without this the flag is silently a no-op.
    _train_utils.seed_training(args.training_seed)
    args.frame_offsets = list(normalize_frame_offsets(args.frame_offsets))
    device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    stats_dir = os.path.dirname(args.stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    # Use the first rollout directory as the reference for validation and
    # per-mode training density, then apply that density to later directories.
    rollout_dirs = normalize_rollout_dirs(args.rollout_dirs)
    if args.manifest_path:
        if args.modes is not None or args.data_fraction != 1.0:
            raise ValueError("--manifest-path cannot be combined with --modes or --data-fraction")
        if args.val_holdout_fraction != 0.0 or args.val_max_rollout_files != 0:
            raise ValueError("Frozen-base manifest training uses all 800 experts; set validation holdout to zero")
        exact = load_frozen_base_manifest(args.manifest_path, rollout_dirs=rollout_dirs)
        train_pool, val_files = exact["train_files"], []
        split_info = {
            "selection": "direct_frozen_base_manifest",
            "reference_source": exact["source_manifest_path"],
            "reference_train_files": len(train_pool),
            "reference_modes": 8,
            "target_per_mode": 100.0,
            "train_files": train_pool, "val_files": val_files,
            "exact_manifest": exact,
        }
    else:
        source_specs = []
        for rollout_dir in rollout_dirs:
            paths = discover_rollout_files(rollout_dir)
            if args.modes is not None:
                mode_strs = {f"mode{m}" for m in args.modes}
                paths = [
                    p for p in paths if any(ms in os.path.basename(p) for ms in mode_strs)
                ]
            source_specs.append((rollout_dir, sorted(paths)))
        split_info = _train_utils.split_sources_by_reference_mode_count(
            source_specs,
            holdout_fraction=args.val_holdout_fraction,
            max_val_files=args.val_max_rollout_files,
            val_seed=args.val_subset_seed,
            data_fraction=args.data_fraction,
            data_seed=args.data_subset_seed,
        )
        train_pool = split_info["train_files"]
        val_files = split_info["val_files"]
    print(
        "Reference-mode split: "
        f"reference={split_info['reference_source']}, "
        f"target_per_mode={split_info['target_per_mode']:.2f}; "
        f"train={len(train_pool)}, val={len(val_files)}"
    )

    dataset = SingleArmShoulderE2EDataset(
        rollout_dirs=args.rollout_dirs,
        horizon=args.horizon,
        augment=not args.no_augment,
        modes=args.modes,
        rollout_files=train_pool,
        frame_offsets=args.frame_offsets,
    )
    if args.balanced_task_mode_sampling and args.sampling_policy != "timestep":
        raise ValueError(
            "--balanced-task-mode-sampling cannot be combined with an explicit --sampling-policy"
        )
    effective_sampling_policy = (
        "hierarchical" if args.balanced_task_mode_sampling else args.sampling_policy
    )
    sampling_summary = None
    train_sampler = None
    if effective_sampling_policy == "hierarchical":
        train_sampler, sampling_summary = make_hierarchical_sampler(
            dataset, seed=args.training_seed, stream=100
        )
        shares = {
            group: info["legacy_timestep_probability"]
            for group, info in sampling_summary["groups"].items()
        }
        print(f"Hierarchical sampling: {sampling_summary['group_count']} groups, "
              f"each targeted at {1.0 / sampling_summary['group_count']:.3f}")
        for group, legacy in sorted(shares.items()):
            print(f"  {group}: uniform-timestep share was {legacy:.3f}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_files:
        val_dataset = SingleArmShoulderE2EDataset(
            rollout_dirs=args.rollout_dirs,
            horizon=args.horizon,
            augment=False,
            modes=args.modes,
            rollout_files=val_files,
            frame_offsets=args.frame_offsets,
        )
        # Use training stats so val operates in the same normalized space.
        val_dataset.action_mean = dataset.action_mean
        val_dataset.action_std = dataset.action_std
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Validation rollout files ({len(val_files)}).")
    else:
        print("No held-out validation rollout files available.")

    all_ac = np.concatenate(dataset.all_actions, axis=0)
    all_ac_norm = (all_ac - dataset.action_mean) / dataset.action_std
    sigma_data = float(all_ac_norm.std())
    print(f"sigma_data={sigma_data:.4f}, dataset size={len(dataset)}")

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
        frame_offsets=args.frame_offsets,
    )

    n_params_total = sum(p.numel() for p in model.F.parameters())
    n_params_trainable = sum(p.numel() for p in model.F.parameters() if p.requires_grad)
    task_profile, mode_to_side_map = build_task_profile(dataset.task_configs)

    print(f"Model params: {n_params_total/1e6:.1f}M total, {n_params_trainable/1e6:.1f}M trainable")

    prefix = args.checkpoint_prefix
    stats = {
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
        "horizon": int(args.horizon),
        "sigma_data": sigma_data,
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "depth": int(args.depth),
        "dim_feedforward": int(args.dim_feedforward),
        "num_cameras": num_cameras,
        "frame_offsets": list(args.frame_offsets),
        "pipeline": "e2e_image_singlearm_mixedfront_shoulder",
        "n_params": int(n_params_total),
        "n_params_trainable": int(n_params_trainable),
        "cfg_drop_prob": float(args.cfg_drop_prob),
        "rollout_dirs": list(args.rollout_dirs),
        "camera_views": camera_views,
        "backbone": args.backbone,
        "balanced_task_mode_sampling": bool(args.balanced_task_mode_sampling),
        "sampling_summary": sampling_summary,
        "task_profile": task_profile,
        "mode_to_side_map": mode_to_side_map,
        "data_fraction": float(args.data_fraction),
        "data_subset_seed": int(args.data_subset_seed),
        "training_seed": int(args.training_seed),
        "sampling_policy": effective_sampling_policy,
        "sampling_hierarchy": (
            HIERARCHY if effective_sampling_policy == "hierarchical" else "timestep-indexed shuffle"
        ),
        "effective_sampling_weights": sampling_summary,
        "reference_mode_split": split_info,
        "available_rollout_files": dataset.available_pkl_files,
        "selected_rollout_files": dataset.selected_pkl_files,
        "validation_rollout_files": val_files,
    }
    with open(args.stats_path, "wb") as f:
        pkl.dump(stats, f)
    print(f"Saved stats: {args.stats_path}")

    init_wandb(args, extra_config={
        "sigma_data": sigma_data,
        "n_params": n_params_total,
        "n_params_trainable": n_params_trainable,
        "dataset_size": len(dataset),
        "pipeline": "e2e_image_singlearm_mixedfront_shoulder",
        "camera_views": camera_views,
        "num_cameras": num_cameras,
        "frame_offsets": list(args.frame_offsets),
        "backbone": args.backbone,
        "balanced_task_mode_sampling": bool(args.balanced_task_mode_sampling),
        "sampling_group_count": None if sampling_summary is None else sampling_summary["group_count"],
        "task_profile": task_profile,
        "mode_to_side_map": mode_to_side_map,
        "available_rollout_files": len(dataset.available_pkl_files),
        "selected_rollout_files": len(dataset.selected_pkl_files),
        "validation_rollout_files": len(val_files),
        "reference_mode_split": True,
        "reference_source": split_info["reference_source"],
        "reference_train_files": split_info["reference_train_files"],
        "reference_modes": split_info["reference_modes"],
        "target_per_mode": split_info["target_per_mode"],
    })

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
            plt.plot(val_xs, val_losses, label="Validation Loss",
                     marker="o", linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_xs, train_grads, color="orange")
        plt.xlabel(xlabel)
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(args.checkpoint_dir, "training_curves.png")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")

    finish_wandb()


if __name__ == "__main__":
    main()

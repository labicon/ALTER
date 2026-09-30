"""Train a single policy on combined single-arm + two-arm data.

Combines rollouts from:
  - Single-arm greenfront/redfront (and optionally flipped) directories
  - Two-arm rollout directories (both agents extracted per rollout)

All data is loaded into a unified dataset with shared action normalization,
producing a single ImageConditional_ODE that can handle both single-arm
and two-arm scenarios.

Pipeline:
    1. Data generation scripts                              — single-arm + two-arm rollouts
    2. THIS SCRIPT                                          — train combined policy
"""

import argparse
import json
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

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


def active_robot_idx_for_mode(mode):
    if mode in (4, 5):
        return 1
    return 0


def pick_first_present(rollout, candidates):
    for key in candidates:
        if key in rollout:
            return np.asarray(rollout[key], dtype=np.uint8)
    raise KeyError(f"Missing keys. Tried: {candidates}")


class CombinedE2EDataset(Dataset):
    """Combined dataset loading both single-arm and two-arm rollouts.

    Single-arm rollouts: one trajectory per file (7-dim actions).
    Two-arm rollouts: two trajectories per file (both agents, 7-dim each).

    All trajectories share the same action normalization.

    Rollout IDs are absolute paths. Files under any `singlearm_dirs` directory
    are treated as single-arm; files under any `twoarm_dirs` directory are
    treated as two-arm.
    """

    @staticmethod
    def _mode_key(filename):
        return _train_utils.mode_key(filename)

    @staticmethod
    def _enumerate_pool(rollout_dirs, modes=None):
        paths = []
        for rollout_dir in rollout_dirs:
            if not os.path.isdir(rollout_dir):
                continue
            for root, _, files in os.walk(rollout_dir):
                for fname in sorted(files):
                    if not fname.endswith(".pkl"):
                        continue
                    if modes is not None:
                        mode_strs = {f"mode{m}" for m in modes}
                        if not any(ms in fname for ms in mode_strs):
                            continue
                    paths.append(os.path.abspath(os.path.join(root, fname)))
        return sorted(paths)

    def __init__(
        self,
        singlearm_dirs: list,
        twoarm_dirs: list,
        horizon: int = 20,
        augment: bool = True,
        singlearm_modes=None,
        twoarm_modes: list = None,
        data_fraction: float = 1.0,
        data_subset_seed: int = 0,
        rollout_files=None,
        num_arms: int = 2,
        frame_offsets=(0,),
    ):
        self.horizon = horizon
        self.augment = augment
        self.frame_offsets = normalize_frame_offsets(frame_offsets)
        self.data_fraction = float(data_fraction)
        self.data_subset_seed = int(data_subset_seed)
        self.num_arms = int(num_arms)

        # Build the available pool from both sources.
        sa_pool = self._enumerate_pool(singlearm_dirs, modes=singlearm_modes)
        ta_pool = self._enumerate_pool(twoarm_dirs, modes=twoarm_modes)
        self.available_pkl_files = sorted(sa_pool + ta_pool)
        sa_set = set(sa_pool)

        if rollout_files is not None:
            requested = set(rollout_files)
            missing = sorted(requested.difference(self.available_pkl_files))
            if missing:
                raise FileNotFoundError(
                    f"Requested rollout files not available: {missing[:5]}..."
                )
            chosen = sorted([p for p in self.available_pkl_files if p in requested])
        else:
            if not (0.0 < self.data_fraction <= 1.0):
                raise ValueError(f"data_fraction must be in (0, 1], got {self.data_fraction}")
            if self.data_fraction < 1.0:
                chosen = _train_utils.subsample_rollout_files(
                    self.available_pkl_files, self.data_fraction, self.data_subset_seed
                )
            else:
                chosen = list(self.available_pkl_files)
        self.selected_pkl_files = list(chosen)

        sa_paths = [p for p in chosen if p in sa_set]
        ta_paths = [p for p in chosen if p not in sa_set]

        self.all_imgs_eih = []
        self.all_imgs_shoulder = []
        self.all_actions = []
        self.index_map = []
        self.sample_weights = []   # per index_map entry; 1.0 unless frame_weights
        self.singlearm_sample_indices = []
        self.multiarm_sample_indices = []
        self.trajectory_group_keys = []
        self.trajectory_rollout_ids = []

        sa_count = 0
        ta_count = 0

        # --- Load single-arm rollouts ---
        for path in sa_paths:
            fname = os.path.basename(path)
            mk = _train_utils.mode_key(fname)
            try:
                mode = int(mk.replace("mode", "")) if mk != "mode_unknown" else 2
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
                if "actions_single_arm" in rollout:
                    actions = np.asarray(rollout["actions_single_arm"], dtype=np.float32)[:, :7]
                elif "actions" in rollout:
                    actions = np.asarray(rollout["actions"], dtype=np.float32)
                    if actions.shape[1] >= 14:
                        actions = actions[:, 7:14] if mode in (4, 5) else actions[:, 0:7]
                    else:
                        actions = actions[:, :7]
                else:
                    raise KeyError("no actions")

                imgs_eih = pick_first_present(
                    rollout,
                    ["camera_obs", f"camera_obs{active_idx}", "camera_obs0", "camera_obs1"],
                )
                imgs_shoulder = pick_first_present(
                    rollout,
                    ["camera_obs_shoulder", f"camera_obs_shoulder{active_idx}",
                     "camera_obs_shoulder0", "camera_obs_shoulder1"],
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
            task = _train_utils.rollout_source_name(path, singlearm_dirs)
            self.trajectory_group_keys.append(f"singlearm/{task}/mode{mode}")
            self.trajectory_rollout_ids.append(os.path.abspath(path))
            for t in range(T):
                self.singlearm_sample_indices.append(len(self.index_map))
                self.index_map.append((traj_idx, t))
                self.sample_weights.append(1.0)
            sa_count += 1

        # --- Load multi-arm rollouts (one trajectory per agent per file) ---
        for path in ta_paths:
            mode_key = _train_utils.mode_key(os.path.basename(path))
            group_key = f"multiarm/{mode_key}"
            rollout_id = os.path.abspath(path)
            try:
                with open(path, "rb") as f:
                    rollout = pkl.load(f)
            except Exception as exc:
                print(f"Skipping {path}: load error ({exc})")
                continue

            actions_full = np.asarray(rollout.get("actions", []), dtype=np.float32)
            if actions_full.ndim != 2 or actions_full.shape[1] < 7 * self.num_arms:
                continue

            for agent_idx in range(self.num_arms):
                eih_key = f"camera_obs{agent_idx}"
                shoulder_key = f"camera_obs_shoulder{agent_idx}"
                action_slice = slice(agent_idx * 7, (agent_idx + 1) * 7)

                if eih_key not in rollout or shoulder_key not in rollout:
                    continue

                imgs_eih = np.asarray(rollout[eih_key], dtype=np.uint8)
                imgs_shoulder = np.asarray(rollout[shoulder_key], dtype=np.uint8)
                actions_7 = actions_full[:, action_slice]

                T = min(len(actions_7), len(imgs_eih), len(imgs_shoulder))
                if T < horizon:
                    continue

                traj_idx = len(self.all_actions)
                self.all_actions.append(np.asarray(actions_7[:T], dtype=np.float32))
                self.all_imgs_eih.append(imgs_eih[:T])
                self.all_imgs_shoulder.append(imgs_shoulder[:T])
                self.trajectory_group_keys.append(group_key)
                self.trajectory_rollout_ids.append(rollout_id)
                # Optional per-frame importance weights (add_frame_weights_twoarm.py),
                # shape (T, num_arms); consumed by --sampling-policy weighted exactly
                # as TwoArmE2EImageDataset does for the coordination trainer.
                fw = rollout.get("frame_weights")
                if fw is not None:
                    fw = np.asarray(fw, dtype=np.float32)
                    w_arm = fw[:T, agent_idx] if fw.ndim == 2 else fw[:T]
                else:
                    w_arm = np.ones(T, dtype=np.float32)
                for t in range(T):
                    self.multiarm_sample_indices.append(len(self.index_map))
                    self.index_map.append((traj_idx, t))
                    self.sample_weights.append(float(w_arm[t]))
                ta_count += 1

        if len(self.all_actions) == 0:
            raise RuntimeError("No valid trajectories loaded.")

        # Compute unified action stats
        all_ac = np.concatenate(self.all_actions, axis=0)
        self.action_mean = np.mean(all_ac, axis=0)
        self.action_std = np.std(all_ac, axis=0)
        self.action_std[self.action_std < 1e-6] = 1.0

        print(
            f"Combined dataset: {sa_count} single-arm + {ta_count} multi-arm agent trajectories, "
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
            chunk = actions[t: t + self.horizon]
        chunk = (chunk - self.action_mean) / self.action_std

        img_eih, img_shoulder = prepare_temporal_views(
            self.all_imgs_eih[traj_idx],
            self.all_imgs_shoulder[traj_idx],
            t,
            self.frame_offsets,
            augment=self.augment,
        )

        return img_eih, img_shoulder, torch.FloatTensor(chunk)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train combined policy on single-arm + two-arm data."
    )
    parser.add_argument(
        "--singlearm-dirs", type=str, nargs="+",
        default=[
            "envs/arm/data_gen/rollouts/horizontal_cube_singlearm_greenfront_directgrasp_shoulder",
            "envs/arm/data_gen/rollouts/horizontal_cube_singlearm_redfront_directgrasp_shoulder",
        ],
    )
    parser.add_argument(
        "--twoarm-dirs", type=str, nargs="+",
        default=[
            "envs/arm/data_gen/rollouts/horizontal_cube_twoarm_redfront_directgrasp_shoulder",
        ],
    )
    parser.add_argument(
        "--twoarm-modes", type=int, nargs="+", default=None,
        help="Filter two-arm rollouts by mode (e.g. --twoarm-modes 2 4). All modes if omitted.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default="checkpoints/arm/combined_e2e_shoulder",
    )
    parser.add_argument(
        "--checkpoint-prefix", type=str, default="combined",
    )
    parser.add_argument("--capacity-job-contract", help="Immutable capacity-ablation job spec to seal into checkpoint contracts.")
    parser.add_argument(
        "--init-model-path",
        help="Initialize weights and EMA from a compatible checkpoint; optimizer state is intentionally fresh.",
    )
    parser.add_argument(
        "--stats-path", type=str,
        default="stats/arm/combined_e2e_shoulder_stats.pkl",
    )
    parser.add_argument(
        "--singlearm-modes", type=int, nargs="+", default=None,
        help="Filter single-arm rollouts by mode. All modes if omitted.",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--frame-offsets", type=int, nargs="+", default=[0],
        help="Descending image lookbacks ending in 0; e.g. 15 0 conditions on [t-15, t].",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--domain-balanced-batches", action="store_true",
        help="Use one full multi-arm batch and one full single-arm batch per optimizer step.",
    )
    parser.add_argument(
        "--sampling-policy", choices=("timestep", "hierarchical", "weighted"), default="timestep",
        help="Within-domain sampling policy; hierarchical balances task/mode, rollout, agent, then "
             "timestep. weighted draws two-arm samples in proportion to the per-frame "
             "frame_weights baked into the pkls (single-arm stays uniform), matching the "
             "coordination trainer's weighted policy.",
    )
    parser.add_argument(
        "--expected-model-params", type=int, default=0,
        help="If positive, fail unless the instantiated model is within the configured tolerance.",
    )
    parser.add_argument(
        "--model-param-tolerance", type=int, default=0,
        help="Allowed absolute parameter-count difference from --expected-model-params.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "dinov2"])
    parser.add_argument("--shoulder-only", action="store_true",
                        help="Train with shoulder camera only (no eye-in-hand).")
    parser.add_argument("--num-arms", type=int, default=2,
                        help="Number of agents to extract from each multi-arm rollout.")
    parser.add_argument(
        "--manifest-path",
        help="Exact JSON manifest; bypasses all legacy splitting and holdout logic.",
    )
    parser.add_argument("--train-rollouts-per-mode", type=int, default=0,
                        help="If >0, use up to this many training rollout files per mode for each source after validation holdout.")
    parser.add_argument("--singlearm-full-pool", action="store_true",
                        help="Skip --train-rollouts-per-mode and reference-density rebalancing for single-arm sources, keeping their full post-holdout training pool (still gated by --data-fraction).")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    _train_utils.add_train_loop_args(parser, default_epochs=500)
    add_wandb_args(parser)
    return parser.parse_args()


def _resolve_training_split(args, source_specs, source_types):
    if not args.manifest_path:
        return _train_utils.split_sources_by_reference_mode_count(
            source_specs,
            holdout_fraction=args.val_holdout_fraction,
            max_val_files=args.val_max_rollout_files,
            val_seed=args.val_subset_seed,
            data_fraction=args.data_fraction,
            data_seed=args.data_subset_seed,
            train_rollouts_per_mode=args.train_rollouts_per_mode,
            source_types=source_types,
            singlearm_full_pool=args.singlearm_full_pool,
        )
    if len(args.twoarm_dirs) != 1:
        raise ValueError("Exact manifests require exactly one multi-arm directory")
    if args.data_fraction != 1.0 or args.singlearm_full_pool:
        raise ValueError("Exact manifests cannot be combined with data resampling options")
    with open(args.manifest_path, "r", encoding="utf-8") as stream:
        schema = json.load(stream).get("schema")
    loader = load_lowvar_combined_manifest if schema in (LOWVAR_COMBINED_SCHEMA, HIGHVAR_V3_SCHEMA) else load_exact_manifest
    exact = loader(
        args.manifest_path, multiarm_root=args.twoarm_dirs[0], singlearm_roots=args.singlearm_dirs
    )
    selected = set(exact["singlearm_train_files"] + exact["multiarm_train_files"])
    sources = []
    for name, available in source_specs:
        chosen = [path for path in available if path in selected]
        sources.append({
            "name": name, "available_files": chosen,
            "selected_train_files": chosen, "validation_files": [],
        })
    train_files = exact["singlearm_train_files"] + exact["multiarm_train_files"]
    represented = {path for source in sources for path in source["selected_train_files"]}
    if represented != set(train_files):
        raise ValueError("Exact manifest files do not map exactly onto configured rollout directories")
    return {
        "selection": "direct_exact_manifest",
        "reference_source": exact["source_manifest_path"],
        "reference_train_files": len(exact["multiarm_train_files"]),
        "reference_modes": 4,
        "target_per_mode": float(exact["per_mode_demo_budget"]),
        "train_files": train_files, "val_files": [], "sources": sources,
        "exact_manifest": exact,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    capacity_ablation = load_capacity_ablation_job_contract(args.capacity_job_contract, method="fs")
    args.frame_offsets = normalize_frame_offsets(args.frame_offsets)
    if args.expected_model_params < 0:
        raise ValueError("--expected-model-params must be nonnegative")
    if args.model_param_tolerance < 0:
        raise ValueError("--model-param-tolerance must be nonnegative")
    if not args.expected_model_params and args.model_param_tolerance:
        raise ValueError("--model-param-tolerance requires --expected-model-params")
    if args.sampling_policy == "hierarchical" and not args.domain_balanced_batches:
        raise ValueError(
            "--sampling-policy hierarchical requires --domain-balanced-batches "
            "so domain is the first balanced level"
        )
    if args.sampling_policy == "weighted" and not args.domain_balanced_batches:
        raise ValueError(
            "--sampling-policy weighted requires --domain-balanced-batches "
            "(the frame weights apply to the two-arm stream only)"
        )
    _train_utils.seed_training(args.training_seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    stats_dir = os.path.dirname(args.stats_path)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load combined dataset
    # ------------------------------------------------------------------
    source_specs = []
    source_types = {}
    for rollout_dir in args.twoarm_dirs:
        files = CombinedE2EDataset._enumerate_pool([rollout_dir], modes=args.twoarm_modes)
        source_specs.append((rollout_dir, files))
        source_types[rollout_dir] = "twoarm"
    for rollout_dir in args.singlearm_dirs:
        files = CombinedE2EDataset._enumerate_pool([rollout_dir], modes=args.singlearm_modes)
        source_specs.append((rollout_dir, files))
        source_types[rollout_dir] = "singlearm"

    split_info = _resolve_training_split(args, source_specs, source_types)
    train_pool = split_info["train_files"]
    val_files = split_info["val_files"]
    sa_pool = []
    ta_pool = []
    sa_train_pool = []
    ta_train_pool = []
    sa_val_files = []
    ta_val_files = []
    for source in split_info["sources"]:
        source_type = source_types[source["name"]]
        if source_type == "singlearm":
            sa_pool.extend(source["available_files"])
            sa_train_pool.extend(source["selected_train_files"])
            sa_val_files.extend(source["validation_files"])
        else:
            ta_pool.extend(source["available_files"])
            ta_train_pool.extend(source["selected_train_files"])
            ta_val_files.extend(source["validation_files"])

    print(
        "Reference-mode split: "
        f"reference={split_info['reference_source']}, "
        f"target_per_mode={split_info['target_per_mode']:.2f}; "
        f"singlearm train={len(sa_train_pool)}, val={len(sa_val_files)}; "
        f"twoarm train={len(ta_train_pool)}, val={len(ta_val_files)}"
    )

    dataset = CombinedE2EDataset(
        singlearm_dirs=args.singlearm_dirs,
        twoarm_dirs=args.twoarm_dirs,
        horizon=args.horizon,
        augment=not args.no_augment,
        singlearm_modes=args.singlearm_modes,
        twoarm_modes=args.twoarm_modes,
        rollout_files=train_pool,
        num_arms=args.num_arms,
        frame_offsets=args.frame_offsets,
    )
    def make_loader(subset, *, shuffle, drop_last, stream, sampler=None):
        return DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            **_train_utils.dataloader_seed_kwargs(args.training_seed, stream=stream),
        )

    singlearm_loader = None
    sampling_summaries = {}
    if args.domain_balanced_batches:
        if (
            len(dataset.singlearm_sample_indices) < args.batch_size
            or len(dataset.multiarm_sample_indices) < args.batch_size
        ):
            raise ValueError(
                f"Domain-balanced batching requires at least {args.batch_size} samples/domain; "
                f"got multi-arm={len(dataset.multiarm_sample_indices)}, "
                f"single-arm={len(dataset.singlearm_sample_indices)}"
            )
        multiarm_sampler = None
        singlearm_sampler = None
        if args.sampling_policy == "weighted":
            # Two-arm stream: draw in proportion to the pkls' frame_weights,
            # exactly as the coordination trainer's weighted policy. The
            # sampler indexes INTO the Subset, so weights are gathered in
            # subset order. Single-arm stream stays uniform.
            w = torch.as_tensor(
                [dataset.sample_weights[i] for i in dataset.multiarm_sample_indices],
                dtype=torch.double,
            )
            gen = torch.Generator()
            gen.manual_seed(args.training_seed * 1000 + 100)
            multiarm_sampler = torch.utils.data.WeightedRandomSampler(
                w, num_samples=len(w), replacement=True, generator=gen,
            )
            n_up = int((w > 1).sum())
            sampling_summaries["multiarm"] = {
                "policy": "weighted", "samples": int(len(w)), "upweighted_samples": n_up,
                "weight_max": float(w.max()),
                "effective_upweighted_share": float(w[w > 1].sum() / w.sum()),
            }
            print(
                f"Weighted two-arm sampling: {n_up}/{len(w)} samples upweighted "
                f"(max w={float(w.max()):.0f}), effective share "
                f"{float(w[w > 1].sum() / w.sum()):.1%}"
            )
        if args.sampling_policy == "hierarchical":
            multiarm_sampler, sampling_summaries["multiarm"] = make_hierarchical_sampler(
                dataset,
                sample_indices=dataset.multiarm_sample_indices,
                seed=args.training_seed,
                stream=100,
            )
            singlearm_sampler, sampling_summaries["singlearm"] = make_hierarchical_sampler(
                dataset,
                sample_indices=dataset.singlearm_sample_indices,
                seed=args.training_seed,
                stream=101,
            )
        dataloader = make_loader(
            Subset(dataset, dataset.multiarm_sample_indices),
            shuffle=True, drop_last=True, stream=0, sampler=multiarm_sampler,
        )
        singlearm_loader = make_loader(
            Subset(dataset, dataset.singlearm_sample_indices),
            shuffle=True, drop_last=True, stream=1, sampler=singlearm_sampler,
        )
        print(
            f"Domain-balanced batching: {args.batch_size} multi-arm + "
            f"{args.batch_size} single-arm samples per optimizer step"
        )
    else:
        dataloader = make_loader(
            dataset, shuffle=True, drop_last=True, stream=0,
        )

    val_loader = None
    if val_files:
        val_dataset = CombinedE2EDataset(
            singlearm_dirs=args.singlearm_dirs,
            twoarm_dirs=args.twoarm_dirs,
            horizon=args.horizon,
            augment=False,
            singlearm_modes=args.singlearm_modes,
            twoarm_modes=args.twoarm_modes,
            rollout_files=val_files,
            num_arms=args.num_arms,
            frame_offsets=args.frame_offsets,
        )
        # Share normalization stats with the training set.
        val_dataset.action_mean = dataset.action_mean
        val_dataset.action_std = dataset.action_std
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            **_train_utils.dataloader_seed_kwargs(
                args.training_seed, stream=2 if args.domain_balanced_batches else 1
            ),
        )
        print(f"Validation rollout files ({len(val_files)}).")
    else:
        print("No held-out validation rollout files available.")

    # Compute sigma_data from normalized actions
    all_ac = np.concatenate(dataset.all_actions, axis=0)
    all_ac_norm = (all_ac - dataset.action_mean) / dataset.action_std
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
        frame_offsets=args.frame_offsets,
    )

    if args.init_model_path and not model.load(args.init_model_path):
        raise FileNotFoundError(f"Could not initialize from checkpoint: {args.init_model_path}")

    n_params_total = sum(p.numel() for p in model.F.parameters())
    n_params_trainable = sum(p.numel() for p in model.F.parameters() if p.requires_grad)
    print(f"Model params: {n_params_total/1e6:.1f}M total, {n_params_trainable/1e6:.1f}M trainable")

    param_difference = (
        n_params_total - args.expected_model_params
        if args.expected_model_params
        else None
    )
    if (
        args.expected_model_params
        and abs(param_difference) > args.model_param_tolerance
    ):
        raise ValueError(
            f"Model has {n_params_total:,} params, expected {args.expected_model_params:,} "
            f"+/- {args.model_param_tolerance:,} (difference {param_difference:+,})"
        )
    if args.expected_model_params:
        print(
            f"Parameter-count check passed: target={args.expected_model_params:,}, "
            f"difference={param_difference:+,}"
        )

    capacity_ablation_payload = None
    if capacity_ablation is not None:
        expected = capacity_ablation["model_counts"]
        observed = int(n_params_total)
        if int(expected["f_params"]) != observed:
            raise ValueError("capacity-ablation parameter mismatch: expected={}, observed={}".format(expected["f_params"], observed))
        capacity_ablation_payload = dict(capacity_ablation)
        capacity_ablation_payload["observed_model_counts"] = {"trainable_F": observed, "deployed_F": observed}

    # ------------------------------------------------------------------
    # Save stats
    # ------------------------------------------------------------------
    prefix = args.checkpoint_prefix
    stats = {
        "training_seed": int(args.training_seed),
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
        "pipeline": "combined_e2e_shoulder",
        "n_params": int(n_params_total),
        "n_params_trainable": int(n_params_trainable),
        "cfg_drop_prob": float(args.cfg_drop_prob),
        "singlearm_dirs": list(args.singlearm_dirs),
        "twoarm_dirs": list(args.twoarm_dirs),
        "camera_views": camera_views,
        "backbone": args.backbone,
        "init_model_path": args.init_model_path,
        "data_fraction": float(args.data_fraction),
        "data_subset_seed": int(args.data_subset_seed),
        "num_arms": int(args.num_arms),
        "data_policy": (
            "sa_matched_density" if split_info.get("exact_manifest") else None
        ),
        "domain_batching": (
            "separate_equal_multiarm_singlearm_loaders"
            if args.domain_balanced_batches
            else "single_unified_sample_loader"
        ),
        "batch_size": int(args.batch_size),
        "batch_size_per_domain": (
            int(args.batch_size) if args.domain_balanced_batches else None
        ),
        "effective_samples_per_optimizer_step": (
            int(args.batch_size) * 2
            if args.domain_balanced_batches
            else int(args.batch_size)
        ),
        "domain_loss_reduction": (
            "independent_mean_then_sum"
            if args.domain_balanced_batches
            else "unified_mean"
        ),
        "singlearm_training_samples": len(dataset.singlearm_sample_indices),
        "multiarm_training_samples": len(dataset.multiarm_sample_indices),
        "expected_model_params": int(args.expected_model_params) or None,
        "model_param_tolerance": int(args.model_param_tolerance),
        "model_param_difference": param_difference,
        "sampling_unit": "agent_timestep_after_multiarm_rollout_expansion",
        "sampling_policy": args.sampling_policy,
        "sampling_hierarchy": (
            HIERARCHY if args.sampling_policy == "hierarchical"
            else "frame_weights-proportional (two-arm stream)" if args.sampling_policy == "weighted"
            else "timestep-indexed shuffle"
        ),
        "effective_sampling_weights": sampling_summaries,
        "train_rollouts_per_mode": int(args.train_rollouts_per_mode),
        "singlearm_full_pool": bool(args.singlearm_full_pool),
        "reference_mode_split": split_info,
        "singlearm_available_rollout_files": sa_pool,
        "twoarm_available_rollout_files": ta_pool,
        "singlearm_selected_rollout_files": sa_train_pool,
        "twoarm_selected_rollout_files": ta_train_pool,
        "singlearm_validation_rollout_files": sa_val_files,
        "twoarm_validation_rollout_files": ta_val_files,
        "available_rollout_files": dataset.available_pkl_files,
        "selected_rollout_files": dataset.selected_pkl_files,
        "validation_rollout_files": val_files,
        "exact_manifest": split_info.get("exact_manifest"),
        "capacity_ablation": capacity_ablation_payload,
    }
    with open(args.stats_path, "wb") as f:
        pkl.dump(stats, f)
    print(f"Saved stats: {args.stats_path}")

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    init_wandb(args, extra_config={
        "sigma_data": sigma_data,
        "n_params": n_params_total,
        "n_params_trainable": n_params_trainable,
        "domain_balanced_batches": bool(args.domain_balanced_batches),
        "sampling_policy": args.sampling_policy,
        "sampling_hierarchy": (
            HIERARCHY if args.sampling_policy == "hierarchical"
            else "frame_weights-proportional (two-arm stream)" if args.sampling_policy == "weighted"
            else "timestep-indexed shuffle"
        ),
        "batch_size_per_domain": (
            int(args.batch_size) if args.domain_balanced_batches else None
        ),
        "effective_samples_per_optimizer_step": (
            int(args.batch_size) * (2 if args.domain_balanced_batches else 1)
        ),
        "dataset_size": len(dataset),
        "pipeline": "combined_e2e_shoulder",
        "camera_views": camera_views,
        "num_cameras": num_cameras,
        "frame_offsets": list(args.frame_offsets),
        "backbone": args.backbone,
        "available_rollout_files": len(dataset.available_pkl_files),
        "selected_rollout_files": len(dataset.selected_pkl_files),
        "validation_rollout_files": len(val_files),
        "singlearm_available_rollout_files": len(sa_pool),
        "twoarm_available_rollout_files": len(ta_pool),
        "singlearm_selected_rollout_files": len(sa_train_pool),
        "twoarm_selected_rollout_files": len(ta_train_pool),
        "num_arms": int(args.num_arms),
        "train_rollouts_per_mode": int(args.train_rollouts_per_mode),
        "singlearm_validation_rollout_files": len(sa_val_files),
        "twoarm_validation_rollout_files": len(ta_val_files),
        "reference_mode_split": True,
        "reference_source": split_info["reference_source"],
        "reference_train_files": split_info["reference_train_files"],
        "reference_modes": split_info["reference_modes"],
        "target_per_mode": split_info["target_per_mode"],
    })

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("Training combined policy...")

    def prepare_batch(batch):
        img_eih, img_shoulder, actions = batch
        img_shoulder = img_shoulder.to(device)
        actions = actions.to(device)
        eih = None if args.shoulder_only else img_eih.to(device)
        return actions, eih, img_shoulder

    singlearm_iter = iter(singlearm_loader) if singlearm_loader is not None else None

    def train_step_fn(batch):
        nonlocal singlearm_iter
        multiarm_or_unified_batch = prepare_batch(batch)
        if singlearm_loader is None:
            return model.update(*multiarm_or_unified_batch)
        try:
            singlearm_batch = next(singlearm_iter)
        except StopIteration:
            singlearm_iter = iter(singlearm_loader)
            singlearm_batch = next(singlearm_iter)
        return model.update_mixed(
            multiarm_or_unified_batch, prepare_batch(singlearm_batch)
        )

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
        plt.title("Combined Policy Training vs Validation Loss")
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

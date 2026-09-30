"""Fine-tune an all-modes single-arm policy jointly on exact SA+multi-arm data.

Each optimizer step consumes one multi-arm and one single-arm batch, concatenates
them in the frozen base policy's normalized action space, and updates the whole
policy. Resolved rollout lists come from either a saved exact-manifest
from-scratch stats file or a canonical three-arm JSON manifest, making this a
data-matched full-policy alternative to the frozen-base coordination head.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from envs.arm.train import _train_utils
from envs.arm.train.exact_threearm_manifest import load_exact_threearm_manifest
from envs.arm.train.train_singlearm_mixedfront_e2e_shoulder import SingleArmShoulderE2EDataset
from envs.arm.train.twoarm_dataset import TwoArmE2EImageDataset
from scripts.experiment_contract import write_training_contract
from src.image_diffusion import ImageConditional_ODE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twoarm-dir", required=True)
    parser.add_argument("--singlearm-dirs", nargs="+", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subset-stats-path",
                        help="Saved exact-manifest source stats from the matched FS run.")
    source.add_argument("--manifest-path",
                        help="Canonical exact three-arm JSON manifest.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--base-stats-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-prefix", default="mixed_finetune_placewipe")
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-domain batch size; each update contains one SA and one TA batch.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--num-arms", type=int, default=2,
                        help="Number of agents represented by each multi-arm rollout.")
    parser.add_argument("--twoarm-modes", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Multi-arm modes retained by the dataset loader.")
    parser.add_argument("--task-family", choices=("placewipe", "wipe"), default="placewipe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate exact source files and exit without creating an output.")
    _train_utils.add_train_loop_args(parser, default_epochs=500)
    parser.set_defaults(max_train_steps=300000, checkpoint_every_steps=30000,
                        val_every_steps=30000, val_holdout_fraction=0.0)
    return parser.parse_args()


def _under(path: str, root: str) -> bool:
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def _normalize_manifest(manifest: dict, twoarm_dir: str, singlearm_dirs: list[str]) -> dict:
    normalized = dict(manifest)
    aliases = {
        "twoarm_train_files": ("twoarm_train_files", "multiarm_train_files"),
        "singlearm_train_files": ("singlearm_train_files",),
        "twoarm_validation_files": ("twoarm_validation_files", "multiarm_validation_files"),
        "singlearm_validation_files": ("singlearm_validation_files",),
    }
    for destination, candidates in aliases.items():
        values = next(
            (manifest.get(key) for key in candidates if manifest.get(key) is not None),
            [],
        )
        normalized[destination] = sorted(
            os.path.abspath(str(value)) for value in values
        )

    all_values = [value for key in aliases for value in normalized[key]]
    if len(all_values) != len(set(all_values)):
        raise ValueError("Exact manifest has duplicate files across train/validation splits")
    if not normalized["twoarm_train_files"] or not normalized["singlearm_train_files"]:
        raise ValueError(
            "Joint finetuning requires non-empty exact SA and multi-arm training lists"
        )
    if any(
        not _under(path, twoarm_dir)
        for path in normalized["twoarm_train_files"]
        + normalized["twoarm_validation_files"]
    ):
        raise ValueError("Multi-arm exact-manifest file is outside --twoarm-dir")
    if any(
        not any(_under(path, root) for root in singlearm_dirs)
        for path in normalized["singlearm_train_files"]
        + normalized["singlearm_validation_files"]
    ):
        raise ValueError("Single-arm exact-manifest file is outside --singlearm-dirs")
    missing = [path for path in all_values if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Exact manifest files missing ({len(missing)}): {missing[:5]}"
        )
    return normalized


def _manifest_from_stats(path: str, twoarm_dir: str, singlearm_dirs: list[str]) -> dict:
    with open(path, "rb") as handle:
        source = pkl.load(handle)
    embedded = source.get("dataset_manifest") if isinstance(source, dict) else None
    if isinstance(embedded, dict):
        manifest = {
            **embedded,
            "twoarm_train_files": list(embedded.get("twoarm_train_files", [])),
            "singlearm_train_files": list(embedded.get("singlearm_train_files", [])),
            "twoarm_validation_files": list(embedded.get("twoarm_validation_files", [])),
            "singlearm_validation_files": list(embedded.get("singlearm_validation_files", [])),
        }
    else:
        required = ("twoarm_selected_rollout_files", "singlearm_selected_rollout_files",
                    "twoarm_validation_rollout_files", "singlearm_validation_rollout_files")
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"Exact source stats lacks manifest fields: {missing}")
        manifest = {
            "twoarm_train_files": list(source["twoarm_selected_rollout_files"]),
            "singlearm_train_files": list(source["singlearm_selected_rollout_files"]),
            "twoarm_validation_files": list(source["twoarm_validation_rollout_files"]),
            "singlearm_validation_files": list(source["singlearm_validation_rollout_files"]),
        }
    manifest["source_stats_path"] = os.path.abspath(path)
    return _normalize_manifest(manifest, twoarm_dir, singlearm_dirs)


def _manifest_from_direct(
    path: str, twoarm_dir: str, singlearm_dirs: list[str]
) -> dict:
    exact = load_exact_threearm_manifest(
        path, multiarm_root=twoarm_dir, singlearm_roots=singlearm_dirs
    )
    return _normalize_manifest(exact, twoarm_dir, singlearm_dirs)


def _loader(dataset, args, *, shuffle: bool, drop_last: bool, stream: int):
    kwargs = dict(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle,
                  drop_last=drop_last, num_workers=args.num_workers,
                  pin_memory=torch.cuda.is_available())
    if args.num_workers:
        kwargs["persistent_workers"] = not args.no_persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    kwargs.update(_train_utils.dataloader_seed_kwargs(args.training_seed, stream=stream))
    return DataLoader(**kwargs)


def _to_device(batch, device, num_cameras):
    eih, shoulder, actions = batch
    return (None if num_cameras == 1 else eih.to(device, non_blocking=True),
            shoulder.to(device, non_blocking=True), actions.to(device, non_blocking=True))


def main():
    args = parse_args()
    _train_utils.seed_training(args.training_seed)
    if args.manifest_path:
        if args.num_arms != 3 or args.task_family != "wipe":
            raise ValueError(
                "Canonical three-arm manifests require --num-arms 3 --task-family wipe"
            )
        manifest = _manifest_from_direct(
            args.manifest_path, args.twoarm_dir, args.singlearm_dirs
        )
    else:
        manifest = _manifest_from_stats(
            args.subset_stats_path, args.twoarm_dir, args.singlearm_dirs
        )
    print(json.dumps({"source_manifest_path": manifest.get("source_manifest_path"),
                      "source_manifest_sha256": manifest.get("source_manifest_sha256"),
                      "source_stats_path": manifest.get("source_stats_path"),
                      "per_mode_demo_budget": manifest.get("per_mode_demo_budget"),
                      "twoarm_train_count": len(manifest["twoarm_train_files"]),
                      "singlearm_train_count": len(manifest["singlearm_train_files"]),
                      "twoarm_validation_count": len(manifest["twoarm_validation_files"]),
                      "singlearm_validation_count": len(manifest["singlearm_validation_files"])}, indent=2))
    if args.validate_only:
        return
    if args.max_train_steps <= 0:
        raise ValueError("--max-train-steps must be positive")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)), exist_ok=True)
    with open(args.base_stats_path, "rb") as handle:
        base_stats = pkl.load(handle)
    horizon = int(base_stats.get("horizon", 20))
    num_cameras = int(base_stats.get("num_cameras", 2))
    device = torch.device(args.device)
    base_mean, base_std = base_stats["action_mean"], base_stats["action_std"]
    ta_train = TwoArmE2EImageDataset(args.twoarm_dir, base_mean, base_std, horizon=horizon,
                                     augment=not args.no_augment, modes=args.twoarm_modes,
                                     num_arms=args.num_arms,
                                     rollout_files=[os.path.basename(os.path.realpath(path)) for path in manifest["twoarm_train_files"]])
    sa_train = SingleArmShoulderE2EDataset(args.singlearm_dirs, horizon=horizon,
                                           augment=not args.no_augment,
                                           rollout_files=[os.path.realpath(path) for path in manifest["singlearm_train_files"]])
    sa_train.action_mean, sa_train.action_std = base_mean, base_std
    ta_loader = _loader(ta_train, args, shuffle=True, drop_last=True, stream=0)
    sa_loader = _loader(sa_train, args, shuffle=True, drop_last=True, stream=1)
    if not len(ta_loader) or not len(sa_loader):
        raise ValueError("Per-domain batch size exceeds available SA or TA dataset samples")
    val_loader = None
    val_sets = []
    if manifest["twoarm_validation_files"]:
        val_sets.append(TwoArmE2EImageDataset(args.twoarm_dir, base_mean, base_std, horizon=horizon,
                        augment=False, modes=args.twoarm_modes, num_arms=args.num_arms,
                        rollout_files=[os.path.basename(os.path.realpath(path)) for path in manifest["twoarm_validation_files"]]))
    if manifest["singlearm_validation_files"]:
        sa_val = SingleArmShoulderE2EDataset(args.singlearm_dirs, horizon=horizon, augment=False,
                                              rollout_files=[os.path.realpath(path) for path in manifest["singlearm_validation_files"]])
        sa_val.action_mean, sa_val.action_std = base_mean, base_std
        val_sets.append(sa_val)
    if val_sets:
        val_loader = _loader(ConcatDataset(val_sets), args, shuffle=False, drop_last=False, stream=2)
    model = ImageConditional_ODE(x_dim=7, sigma_data=float(base_stats.get("sigma_data", 1.0)),
        d_model=int(base_stats.get("d_model", 256)), n_heads=int(base_stats.get("n_heads", 4)),
        depth=int(base_stats.get("depth", 3)), dim_feedforward=int(base_stats.get("dim_feedforward", 1024)),
        horizon=horizon, device=device, lr=args.lr, cfg_drop_prob=args.cfg_drop_prob,
        num_cameras=num_cameras, backbone=str(base_stats.get("backbone", "resnet18")))
    if not model.load(args.base_model_path):
        raise FileNotFoundError(args.base_model_path)
    model.optim = torch.optim.AdamW(model.F.parameters(), lr=args.lr, weight_decay=1e-4)
    stats = {"pipeline": f"{args.task_family}_exact_manifest_joint_full_policy_finetune",
             "task_family": args.task_family, "num_arms": args.num_arms,
             "data_policy": "sa_matched_density",
             "training_seed": args.training_seed, "augment": not args.no_augment,
             "base_model_path": os.path.abspath(args.base_model_path),
             "base_stats_path": os.path.abspath(args.base_stats_path),
             "action_mean": base_mean, "action_std": base_std, "horizon": horizon,
             "sigma_data": float(base_stats.get("sigma_data", 1.0)), "num_cameras": num_cameras,
             "backbone": str(base_stats.get("backbone", "resnet18")), "d_model": int(base_stats.get("d_model", 256)),
             "n_heads": int(base_stats.get("n_heads", 4)), "depth": int(base_stats.get("depth", 3)),
             "dim_feedforward": int(base_stats.get("dim_feedforward", 1024)), "lr": args.lr,
             "per_domain_batch_size": args.batch_size, "effective_batch_size": 2 * args.batch_size,
             "domain_batching": "one_multiarm_plus_one_singlearm_batch_per_optimizer_step",
             "multiarm_loss_weight": 1.0, "singlearm_loss_weight": 1.0,
             "max_train_steps": args.max_train_steps, "checkpoint_every_steps": args.checkpoint_every_steps,
             "dataset_manifest": manifest,
             "resolved_dataset_manifest": {
                 key: [os.path.realpath(path) for path in manifest[key]]
                 for key in ("twoarm_train_files", "singlearm_train_files",
                             "twoarm_validation_files", "singlearm_validation_files")
             }}
    with open(args.stats_path, "wb") as handle:
        pkl.dump(stats, handle)
    with open(os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_data_manifest.json"), "w") as handle:
        json.dump(stats["dataset_manifest"], handle, indent=2)
    ta_iter, sa_iter = iter(ta_loader), iter(sa_loader)
    def train_step_fn(_):
        nonlocal ta_iter, sa_iter
        try:
            ta_batch = next(ta_iter)
        except StopIteration:
            ta_iter = iter(ta_loader)
            ta_batch = next(ta_iter)
        try:
            sa_batch = next(sa_iter)
        except StopIteration:
            sa_iter = iter(sa_loader)
            sa_batch = next(sa_iter)
        ta_eih, ta_shoulder, ta_actions = _to_device(ta_batch, device, num_cameras)
        sa_eih, sa_shoulder, sa_actions = _to_device(sa_batch, device, num_cameras)
        eih = None if num_cameras == 1 else torch.cat((ta_eih, sa_eih), dim=0)
        return model.update(torch.cat((ta_actions, sa_actions), dim=0), eih,
                            torch.cat((ta_shoulder, sa_shoulder), dim=0))
    def val_step_fn(batch, generator, sigma):
        eih, shoulder, actions = _to_device(batch, device, num_cameras)
        return model.validation_loss(actions, eih, shoulder, generator=generator, sigma=sigma), actions.shape[0]
    def save_fn(path):
        model.save(path)
        print(f"Saved experiment contract: {write_training_contract(args=args, stats=stats, checkpoint_path=path, stats_path=args.stats_path, producer_script=__file__, repo_root=REPO_ROOT)}")
    history = _train_utils.train_step_loop(args, ta_loader, train_step_fn, save_fn,
        checkpoint_dir=args.checkpoint_dir, prefix=args.checkpoint_prefix, val_loader=val_loader,
        val_step_fn=val_step_fn, sigma_grid=_train_utils.build_sigma_grid(model, args.val_sigma_grid_size), device=str(device))
    final_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_final.pt")
    save_fn(final_path)
    if history:
        xs, losses, _, val_xs, val_losses = _train_utils.history_to_curves(history)
        plt.plot(xs, losses, label="train")
        if val_xs: plt.plot(val_xs, val_losses, label="validation")
        plt.legend(); plt.xlabel("step"); plt.ylabel("loss"); plt.grid(True)
        plt.savefig(os.path.join(args.checkpoint_dir, "training_curves.png"))


if __name__ == "__main__":
    main()

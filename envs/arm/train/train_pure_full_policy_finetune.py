"""Sealed pure full-policy fine-tuning from an authoritative single-arm policy.

The sealed ``mixed_domain`` treatment concatenates one hierarchical multi-arm
batch and one hierarchical single-arm batch per update. The sealed
``multiarm_only`` treatment consumes exactly one hierarchical multi-arm batch
per update and deliberately never constructs or samples a single-arm dataset.
Both treatments update every original policy parameter. No coordination head,
adapter, replacement architecture, parameter target, or image augmentation is
permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.arm.train import _train_utils
from envs.arm.train.exact_threearm_manifest import load_exact_manifest
from envs.arm.train.hierarchical_sampling import HIERARCHY, make_hierarchical_sampler
from envs.arm.train.lowvar_combined_manifest import (
    HIGHVAR_V3_SCHEMA,
    SCHEMA as LOWVAR_COMBINED_SCHEMA,
    load_lowvar_combined_manifest,
)
from envs.arm.train.train_singlearm_mixedfront_e2e_shoulder import (
    SingleArmShoulderE2EDataset,
)
from envs.arm.train.twoarm_dataset import TwoArmE2EImageDataset
from scripts.experiment_contract import (
    load_pure_finetune_job_contract,
    sha256_file,
    write_training_contract,
)
from src.image_diffusion import ImageConditional_ODE
from src.temporal import frame_offsets_from_stats, normalize_frame_offsets


ARCHITECTURE = {
    "backbone": "resnet18",
    "d_model": 256,
    "n_heads": 4,
    "depth": 3,
    "dim_feedforward": 1024,
    "num_cameras": 1,
}
FRAME_OFFSETS = (0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twoarm-dir", required=True)
    parser.add_argument("--singlearm-dirs", nargs="+", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--base-stats-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-prefix", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--pure-finetune-job-contract", required=True)
    parser.add_argument("--task-family", choices=("placewipe", "wipe"), required=True)
    parser.add_argument("--num-arms", type=int, choices=(2, 3), required=True)
    parser.add_argument("--twoarm-modes", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--frame-offsets", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    # v1 did not name its historical mixed objective. V2 immutable jobs always
    # pass this option explicitly, while the default preserves v1 behavior.
    parser.add_argument(
        "--data-treatment",
        choices=("mixed_domain", "multiarm_only"),
        default="mixed_domain",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling-policy", choices=("hierarchical",), default="hierarchical")
    parser.add_argument("--expected-env-variant", default="hard")
    _train_utils.add_train_loop_args(parser, default_epochs=500)
    parser.set_defaults(
        max_train_steps=900000,
        checkpoint_every_steps=100000,
        val_every_steps=100000,
        val_holdout_fraction=0.0,
        val_batches=0,
    )
    return parser.parse_args()


def _loader(dataset, args: argparse.Namespace, *, stream: int, sampler) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "sampler": sampler,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers:
        kwargs["persistent_workers"] = not args.no_persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    kwargs.update(_train_utils.dataloader_seed_kwargs(args.training_seed, stream=stream))
    return DataLoader(**kwargs)


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    raw = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    schema = raw.get("schema")
    loader = (
        load_lowvar_combined_manifest
        if schema in (LOWVAR_COMBINED_SCHEMA, HIGHVAR_V3_SCHEMA)
        else load_exact_manifest
    )
    exact = loader(
        args.manifest_path,
        multiarm_root=args.twoarm_dir,
        singlearm_roots=args.singlearm_dirs,
    )
    return {
        **exact,
        "twoarm_train_files": list(exact["multiarm_train_files"]),
        "twoarm_validation_files": [],
        "singlearm_validation_files": [],
    }


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError("{} mismatch: observed={!r}, expected={!r}".format(label, observed, expected))


def _validate_runtime(
    args: argparse.Namespace,
    job: dict[str, Any],
    base_stats: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    regime = job["regime"]
    _require_equal("checkpoint_dir", os.path.abspath(args.checkpoint_dir), job["output_dir"])
    _require_equal("base model path", os.path.abspath(args.base_model_path), job["base_policy"]["model_path"])
    _require_equal("base stats path", os.path.abspath(args.base_stats_path), job["base_policy"]["stats_path"])
    _require_equal("manifest path", os.path.abspath(args.manifest_path), job["manifest"]["path"])
    _require_equal("task family", args.task_family, regime["task_family"])
    _require_equal("num arms", int(args.num_arms), int(regime["num_arms"]))
    _require_equal("per-mode demo budget", int(manifest["per_mode_demo_budget"]), int(regime["per_mode_demo_budget"]))
    _require_equal("multi-arm rollout count", len(manifest["multiarm_train_files"]), int(regime["multiarm_train_file_count"]))
    _require_equal("single-arm rollout count", len(manifest["singlearm_train_files"]), int(regime["singlearm_train_file_count"]))
    _require_equal("manifest sha256", manifest["source_manifest_sha256"], job["manifest"]["sha256"])
    _require_equal("frame offsets", list(normalize_frame_offsets(args.frame_offsets)), list(FRAME_OFFSETS))
    _require_equal("base frame offsets", list(frame_offsets_from_stats(base_stats)), list(FRAME_OFFSETS))
    if not args.no_augment:
        raise ValueError("pure full-policy fine-tuning requires --no-augment")
    if args.sampling_policy != "hierarchical":
        raise ValueError("pure full-policy fine-tuning requires hierarchical sampling")
    _require_equal("per-domain batch size", int(args.batch_size), 256)
    _require_equal("learning rate", float(args.lr), 2e-4)
    _require_equal("max train steps", int(args.max_train_steps), 900000)
    _require_equal("checkpoint interval", int(args.checkpoint_every_steps), 100000)

    treatment = str(job.get("data_treatment", regime.get("data_treatment", "mixed_domain")))
    if treatment not in ("mixed_domain", "multiarm_only"):
        raise ValueError("unsupported sealed data treatment {!r}".format(treatment))
    _require_equal("data treatment", args.data_treatment, treatment)

    for key, expected in ARCHITECTURE.items():
        observed = str(base_stats.get(key, expected)) if key == "backbone" else int(base_stats.get(key, expected))
        if key not in base_stats:
            raise ValueError("base stats are missing required architecture field {}".format(key))
        _require_equal("base architecture {}".format(key), observed, expected)
    training_contract = job["training_contract"]
    _require_equal("sealed no-augmentation treatment", regime.get("no_augmentation"), True)
    _require_equal("training-contract augmentation", training_contract.get("augmentation"), False)
    _require_equal("training-contract frame offsets", training_contract.get("frame_offsets"), [0])
    _require_equal("all policy parameters trainable claim", training_contract.get("all_original_policy_parameters_trainable"), True)
    _require_equal("added-module claim", training_contract.get("added_modules"), [])
    _require_equal("frozen-module claim", training_contract.get("frozen_modules_after_initialization"), [])
    if "data_treatment" in training_contract:
        _require_equal("training-contract data treatment", training_contract["data_treatment"], treatment)
    if "effective_batch_size" in training_contract:
        _require_equal(
            "training-contract effective batch size",
            int(training_contract["effective_batch_size"]),
            512 if treatment == "mixed_domain" else 256,
        )
    if "training_consumed_samples_per_update" in training_contract:
        _require_equal(
            "training-contract consumed samples",
            training_contract["training_consumed_samples_per_update"],
            {
                "multiarm": 256,
                "singlearm": 256 if treatment == "mixed_domain" else 0,
            },
        )


def _to_policy_batch(batch, device: torch.device) -> tuple[torch.Tensor, None, torch.Tensor]:
    _, shoulder, actions = batch
    return (
        actions.to(device, non_blocking=True),
        None,
        shoulder.to(device, non_blocking=True),
    )


def main() -> None:
    args = parse_args()
    job = load_pure_finetune_job_contract(args.pure_finetune_job_contract)
    assert job is not None
    _train_utils.seed_training(args.training_seed)

    with open(args.base_stats_path, "rb") as handle:
        base_stats = pkl.load(handle)
    manifest = _load_manifest(args)
    _validate_runtime(args, job, base_stats, manifest)

    for path in (args.base_model_path, args.base_stats_path, args.manifest_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=False)
    stats_parent = Path(args.stats_path).resolve().parent
    stats_parent.mkdir(parents=True, exist_ok=True)

    horizon = int(base_stats.get("horizon", 20))
    action_mean, action_std = base_stats["action_mean"], base_stats["action_std"]
    multiarm_dataset = TwoArmE2EImageDataset(
        rollout_dir=args.twoarm_dir,
        base_action_mean=action_mean,
        base_action_std=action_std,
        horizon=horizon,
        augment=False,
        modes=list(args.twoarm_modes),
        num_arms=args.num_arms,
        rollout_files=manifest["multiarm_train_files"],
        frame_offsets=FRAME_OFFSETS,
    )
    multiarm_sampler, multiarm_sampling = make_hierarchical_sampler(
        multiarm_dataset, seed=args.training_seed, stream=100
    )
    multiarm_loader = _loader(multiarm_dataset, args, stream=0, sampler=multiarm_sampler)
    singlearm_loader = None
    singlearm_sampling = None
    if args.data_treatment == "mixed_domain":
        singlearm_dataset = SingleArmShoulderE2EDataset(
            rollout_dirs=args.singlearm_dirs,
            horizon=horizon,
            augment=False,
            rollout_files=manifest["singlearm_train_files"],
            frame_offsets=FRAME_OFFSETS,
        )
        singlearm_dataset.action_mean, singlearm_dataset.action_std = action_mean, action_std
        singlearm_sampler, singlearm_sampling = make_hierarchical_sampler(
            singlearm_dataset, seed=args.training_seed, stream=101
        )
        singlearm_loader = _loader(singlearm_dataset, args, stream=1, sampler=singlearm_sampler)
    if not len(multiarm_loader) or (singlearm_loader is not None and not len(singlearm_loader)):
        raise ValueError("256-sample per-domain batches exceed available sampled trajectories")

    model = ImageConditional_ODE(
        x_dim=7,
        sigma_data=float(base_stats.get("sigma_data", 1.0)),
        d_model=ARCHITECTURE["d_model"],
        n_heads=ARCHITECTURE["n_heads"],
        depth=ARCHITECTURE["depth"],
        dim_feedforward=ARCHITECTURE["dim_feedforward"],
        horizon=horizon,
        device=device,
        lr=args.lr,
        cfg_drop_prob=args.cfg_drop_prob,
        num_cameras=ARCHITECTURE["num_cameras"],
        backbone=ARCHITECTURE["backbone"],
        frame_offsets=FRAME_OFFSETS,
    )
    if not model.load(args.base_model_path):
        raise FileNotFoundError(args.base_model_path)
    if not all(parameter.requires_grad for parameter in model.F.parameters()):
        raise RuntimeError("the original policy must be entirely trainable")
    if any(parameter.requires_grad for parameter in model.F_ema.parameters()):
        raise RuntimeError("EMA parameters must be maintained, not optimized")
    model.F.train()
    model.F_ema.eval()
    model.optim = torch.optim.AdamW(model.F.parameters(), lr=args.lr, weight_decay=1e-4)

    initial_hashes = {
        "base_model_file_sha256": sha256_file(args.base_model_path),
        "base_stats_file_sha256": sha256_file(args.base_stats_path),
        "manifest_file_sha256": sha256_file(args.manifest_path),
        "initial_F_state_sha256": _state_sha256(model.F),
        "initial_F_ema_state_sha256": _state_sha256(model.F_ema),
    }
    _require_equal("initial base model hash", initial_hashes["base_model_file_sha256"], job["base_policy"]["model_sha256"])
    _require_equal("initial base stats hash", initial_hashes["base_stats_file_sha256"], job["base_policy"]["stats_sha256"])
    _require_equal("initial manifest hash", initial_hashes["manifest_file_sha256"], job["manifest"]["sha256"])

    n_params = sum(parameter.numel() for parameter in model.F.parameters())
    n_trainable = sum(parameter.numel() for parameter in model.F.parameters() if parameter.requires_grad)
    if n_params != n_trainable:
        raise RuntimeError("every original policy parameter must be trainable")

    stats: dict[str, Any] = {
        "pipeline": "pure_finetune_full_policy",
        "pure_finetune_contract": "pure_finetune_full_policy",
        "task_family": args.task_family,
        "env_variant": args.expected_env_variant,
        "num_arms": int(args.num_arms),
        "training_seed": int(args.training_seed),
        "base_model_path": os.path.abspath(args.base_model_path),
        "base_stats_path": os.path.abspath(args.base_stats_path),
        "action_mean": action_mean,
        "action_std": action_std,
        "sigma_data": float(base_stats.get("sigma_data", 1.0)),
        "horizon": horizon,
        "num_cameras": ARCHITECTURE["num_cameras"],
        "frame_offsets": list(FRAME_OFFSETS),
        **{key: value for key, value in ARCHITECTURE.items() if key != "num_cameras"},
        "augment": False,
        "augmentation_treatment": "none",
        "architecture_treatment": "original_singlearm_resnet18_policy_unchanged",
        "added_modules": [],
        "frozen_modules_after_initialization": [],
        "policy_parameters": {
            "all_original_F_parameters_trainable": True,
            "F_total": int(n_params),
            "F_trainable": int(n_trainable),
            "F_ema": int(sum(parameter.numel() for parameter in model.F_ema.parameters())),
        },
        "n_params": int(n_params),
        "n_params_trainable": int(n_trainable),
        "optimizer": {"name": "AdamW", "lr": float(args.lr), "weight_decay": 1e-4},
        "lr": float(args.lr),
        "ema_decay": 0.999,
        "data_treatment": args.data_treatment,
        "per_domain_batch_size": 256,
        "effective_batch_size": 512 if args.data_treatment == "mixed_domain" else 256,
        "domain_batching": (
            "one_hierarchical_multiarm_plus_one_hierarchical_singlearm_concatenated"
            if args.data_treatment == "mixed_domain"
            else "one_hierarchical_multiarm_batch_only"
        ),
        "domain_loss_reduction": (
            "concatenated_unified_mean"
            if args.data_treatment == "mixed_domain"
            else "multiarm_unified_mean"
        ),
        "multiarm_loss_weight": 1.0,
        "singlearm_loss_weight": 1.0 if args.data_treatment == "mixed_domain" else 0.0,
        "rollout_counts": {
            "manifest_available": {
                "multiarm_files": len(manifest["multiarm_train_files"]),
                "singlearm_files": len(manifest["singlearm_train_files"]),
            },
            "training_consumed_samples_per_update": {
                "multiarm": 256,
                "singlearm": 256 if args.data_treatment == "mixed_domain" else 0,
            },
        },
        "sampling_policy": "hierarchical",
        "sampling_hierarchy": HIERARCHY,
        "effective_sampling_weights": {
            "multiarm": multiarm_sampling,
            **({"singlearm": singlearm_sampling} if singlearm_sampling is not None else {}),
        },
        "max_train_steps": int(args.max_train_steps),
        "checkpoint_every_steps": int(args.checkpoint_every_steps),
        "dataset_manifest": manifest,
        "initial_provenance_hashes": initial_hashes,
        "immutable_job": {
            "job_id": job["job_id"],
            "job_spec_path": job["job_spec_path"],
            "job_spec_sha256": job["job_spec_sha256"],
            "command_sha256": job["command_sha256"],
        },
    }
    manifest_path = Path(args.checkpoint_dir) / "{}_data_manifest.json".format(args.checkpoint_prefix)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    multiarm_iter = iter(multiarm_loader)
    singlearm_iter = iter(singlearm_loader) if singlearm_loader is not None else None

    def next_batch(iterator, loader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def train_step_fn(_unused):
        nonlocal multiarm_iter, singlearm_iter
        multiarm_batch, multiarm_iter = next_batch(multiarm_iter, multiarm_loader)
        multiarm_policy = _to_policy_batch(multiarm_batch, device)
        if args.data_treatment == "multiarm_only":
            return model.update(multiarm_policy[0], None, multiarm_policy[2])
        assert singlearm_loader is not None and singlearm_iter is not None
        singlearm_batch, singlearm_iter = next_batch(singlearm_iter, singlearm_loader)
        singlearm_policy = _to_policy_batch(singlearm_batch, device)
        actions = torch.cat((multiarm_policy[0], singlearm_policy[0]), dim=0)
        shoulders = torch.cat((multiarm_policy[2], singlearm_policy[2]), dim=0)
        return model.update(actions, None, shoulders)

    def save_fn(path: str) -> None:
        # Re-hash the sealed inputs at every checkpoint boundary.  They remain
        # immutable inputs even though the in-memory policy is now trainable.
        _require_equal("base model changed during training", sha256_file(args.base_model_path), initial_hashes["base_model_file_sha256"])
        _require_equal("base stats changed during training", sha256_file(args.base_stats_path), initial_hashes["base_stats_file_sha256"])
        _require_equal("manifest changed during training", sha256_file(args.manifest_path), initial_hashes["manifest_file_sha256"])
        model.save(path)
        checkpoint_stats_path = Path(path).with_name(
            "{}_stats.pkl".format(Path(path).stem)
        )
        stats["checkpoint_provenance_hashes"] = {
            "F_state_sha256": _state_sha256(model.F),
            "F_ema_state_sha256": _state_sha256(model.F_ema),
        }
        if Path(path).stem.endswith("_final"):
            stats["final_provenance_hashes"] = {
                "final_F_state_sha256": stats["checkpoint_provenance_hashes"]["F_state_sha256"],
                "final_F_ema_state_sha256": stats["checkpoint_provenance_hashes"]["F_ema_state_sha256"],
            }
        with open(args.stats_path, "wb") as handle:
            pkl.dump(stats, handle)
        with checkpoint_stats_path.open("wb") as handle:
            pkl.dump(stats, handle)
        contract_path = write_training_contract(
            args=args,
            stats=stats,
            checkpoint_path=path,
            stats_path=str(checkpoint_stats_path),
            producer_script=__file__,
            repo_root=str(REPO_ROOT),
            task_family=args.task_family,
        )
        print("Saved experiment contract: {}".format(contract_path), flush=True)

    # The epoch loader only drives the number of optimizer steps; train_step_fn
    # consumes the sealed hierarchical stream(s) itself.
    _train_utils.train_step_loop(
        args,
        multiarm_loader,
        train_step_fn,
        save_fn,
        checkpoint_dir=args.checkpoint_dir,
        prefix=args.checkpoint_prefix,
        val_loader=None,
        val_step_fn=None,
        sigma_grid=None,
        device=str(device),
    )
    save_fn(os.path.join(args.checkpoint_dir, "{}_final.pt".format(args.checkpoint_prefix)))


if __name__ == "__main__":
    main()

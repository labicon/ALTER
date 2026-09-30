"""Shared training-loop and validation utilities for arm trainers.

Callback-based, model-agnostic, dataset-agnostic.

Public API:
    add_train_loop_args(parser, **defaults)
    enumerate_pkl_files(rollout_dir, modes=None)
    mode_key(filename)
    subsample_rollout_files(files, fraction, seed)
    select_rollout_files_per_mode(files, per_mode, seed)
    select_balanced_files(files, max_files, seed)
    split_train_val_files(available_files, holdout_fraction, max_val_files, seed)
    split_sources_by_reference_mode_count(sources, holdout_fraction,
                                          max_val_files, val_seed,
                                          data_fraction, data_seed)
    build_sigma_grid(model, n)
    run_validation(val_loader, val_step_fn, *, val_batches, sigma_grid,
                   noise_seed, device)
    train_step_loop(args, dataloader, train_step_fn, save_fn, ...)
    train_epoch_loop(args, dataloader, train_step_fn, save_fn, ...)

Callback contracts:
    train_step_fn(batch) -> (loss_scalar, grad_norm_scalar[, extra_metrics_dict])
    val_step_fn(batch, generator, sigma_or_None) -> (loss_scalar, n_samples)
    save_fn(path: str) -> None
"""

from __future__ import annotations

import json
import os
import random
import re

import numpy as np
import torch
from tqdm import tqdm

from utils.wandb_utils import log_metrics as _default_log_fn

_MODE_RE = re.compile(r"mode(-?\d+)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_train_loop_args(
    parser,
    *,
    default_epochs: int = 400,
    default_save_every: int = 50,
    default_val_max: int = 16,
    default_val_batches: int = 20,
):
    """Add step/epoch/validation/subsampling CLI args to an argparse parser."""
    g = parser.add_argument_group("train_loop")
    g.add_argument(
        "--training-seed", type=int, default=0,
        help=("Seed Python, NumPy, Torch, DataLoader shuffling, and worker RNGs "
              "for reproducible training."),
    )
    g.add_argument("--epochs", type=int, default=default_epochs)
    g.add_argument(
        "--save-every", type=int, default=default_save_every,
        help="Epoch interval for checkpoints (only used when --max-train-steps=0).",
    )
    g.add_argument(
        "--max-train-steps", type=int, default=0,
        help="If >0, train for this many optimizer steps instead of epochs.",
    )
    g.add_argument(
        "--log-every-steps", type=int, default=50,
        help="Train-loss logging interval (cheap; no validation, no checkpoint).",
    )
    g.add_argument(
        "--val-every-steps", type=int, default=0,
        help="Validation interval; 0 -> use --checkpoint-every-steps.",
    )
    g.add_argument(
        "--checkpoint-every-steps", type=int, default=0,
        help="Step checkpoint interval when --max-train-steps is set; 0 -> max_steps//5.",
    )
    g.add_argument(
        "--data-fraction", type=float, default=1.0,
        help="Fraction of training rollout files (post-holdout) to use.",
    )
    g.add_argument(
        "--data-subset-seed", type=int, default=0,
        help="Seed for deterministic --data-fraction subsampling.",
    )
    g.add_argument(
        "--val-holdout-fraction", type=float, default=0.1,
        help="Fraction of rollout files reserved for validation (carved BEFORE --data-fraction).",
    )
    g.add_argument(
        "--val-max-rollout-files", type=int, default=default_val_max,
        help="Cap on number of validation rollout files.",
    )
    g.add_argument(
        "--val-batches", type=int, default=default_val_batches,
        help="Max validation batches per checkpoint (0 = use entire val loader).",
    )
    g.add_argument(
        "--val-subset-seed", type=int, default=123,
        help="Seed for deterministic held-out validation file selection.",
    )
    g.add_argument(
        "--val-noise-seed", type=int, default=42,
        help="Seed for deterministic noise/cfg-mask sampling during validation.",
    )
    g.add_argument(
        "--val-sigma-grid-size", type=int, default=8,
        help="Number of fixed log-spaced sigmas for validation; 0 = single random sigma per batch.",
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_training(seed: int) -> int:
    """Seed process-level RNGs used by datasets, models, and augmentations."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def seed_dataloader_worker(worker_id: int) -> None:
    """Derive deterministic Python/NumPy worker RNGs from Torch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataloader_seed_kwargs(seed: int, *, stream: int = 0) -> dict:
    """Return deterministic DataLoader generator/worker initialization kwargs."""
    generator = torch.Generator()
    # Keep independent, stable streams for train/validation and data domains.
    generator.manual_seed(int(seed) + 1_000_003 * int(stream))
    return {
        "generator": generator,
        "worker_init_fn": seed_dataloader_worker,
    }


# ---------------------------------------------------------------------------
# Rollout-file selection
# ---------------------------------------------------------------------------


def mode_key(filename: str) -> str:
    """Return e.g. 'mode2' from a filename, else 'mode_unknown'."""
    match = _MODE_RE.search(filename)
    return match.group(0) if match else "mode_unknown"


def rollout_source_name(path: str, rollout_dirs) -> str:
    """Return the configured source-root name containing a rollout path."""
    path_abs = os.path.abspath(path)
    roots = [os.path.abspath(root) for root in rollout_dirs]
    matches = [
        root
        for root in roots
        if os.path.commonpath([path_abs, root]) == root
    ]
    if not matches:
        raise ValueError(f"Rollout is outside configured source roots: {path}")
    source_root = max(matches, key=len)
    return os.path.basename(os.path.normpath(source_root))


def enumerate_pkl_files(rollout_dir: str, modes=None) -> list[str]:
    """Return sorted .pkl filenames in rollout_dir, optionally filtered by modes."""
    files = sorted(f for f in os.listdir(rollout_dir) if f.endswith(".pkl"))
    if modes is not None:
        mode_strs = {f"mode{m}" for m in modes}
        files = [f for f in files if any(ms in f for ms in mode_strs)]
    return files


def subsample_rollout_files(files, fraction: float, seed: int) -> list[str]:
    """Deterministic mode-balanced subsample of rollout files."""
    if fraction >= 1.0:
        return sorted(files)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for mode in sorted({mode_key(f) for f in files}):
        mode_files = [f for f in files if mode_key(f) == mode]
        order = rng.permutation(len(mode_files))
        k = max(1, int(round(len(mode_files) * fraction)))
        k = min(k, len(mode_files))
        selected.extend(mode_files[i] for i in order[:k])
    return sorted(selected)


def select_rollout_files_per_mode(files, per_mode: int, seed: int) -> list[str]:
    """Pick up to `per_mode` rollout files from each filename mode bucket."""
    files = sorted(files)
    per_mode = int(per_mode)
    if per_mode <= 0:
        return files
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for mode in sorted({mode_key(f) for f in files}):
        mode_files = [f for f in files if mode_key(f) == mode]
        order = rng.permutation(len(mode_files))
        k = min(per_mode, len(mode_files))
        selected.extend(mode_files[i] for i in order[:k])
    return sorted(selected)

def select_balanced_files(files, max_files: int, seed: int) -> list[str]:
    """Pick up to max_files balanced across modes (deterministic)."""
    files = sorted(files)
    if max_files <= 0 or len(files) <= max_files:
        return files
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    modes = sorted({mode_key(f) for f in files})
    per_mode = max(1, max_files // max(1, len(modes)))
    for mode in modes:
        mode_files = [f for f in files if mode_key(f) == mode]
        order = rng.permutation(len(mode_files))
        selected.extend(mode_files[i] for i in order[: min(per_mode, len(mode_files))])
    remaining = [f for f in files if f not in set(selected)]
    if len(selected) < max_files and remaining:
        order = rng.permutation(len(remaining))
        selected.extend(remaining[i] for i in order[: max_files - len(selected)])
    return sorted(selected[:max_files])


def select_balanced_files_by_key(files, max_files: int, seed: int, key_fn) -> list[str]:
    """Pick up to max_files balanced across arbitrary key_fn buckets."""
    files = sorted(files)
    max_files = min(int(max_files), len(files))
    if max_files <= 0:
        return []
    if len(files) <= max_files:
        return files

    rng = np.random.default_rng(seed)
    keys = sorted({key_fn(f) for f in files})
    if not keys:
        return []

    key_order = list(keys)
    rng.shuffle(key_order)
    base_quota = max_files // len(keys)
    remainder = max_files % len(keys)
    quotas = {
        key: base_quota + (1 if i < remainder else 0)
        for i, key in enumerate(key_order)
    }

    selected: list[str] = []
    for key in keys:
        key_files = [f for f in files if key_fn(f) == key]
        order = rng.permutation(len(key_files))
        selected.extend(key_files[i] for i in order[: min(quotas[key], len(key_files))])

    remaining = [f for f in files if f not in set(selected)]
    if len(selected) < max_files and remaining:
        order = rng.permutation(len(remaining))
        selected.extend(remaining[i] for i in order[: max_files - len(selected)])
    return sorted(selected[:max_files])


def split_train_val_files(
    available_files,
    holdout_fraction: float,
    max_val_files: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Carve out validation files BEFORE any --data-fraction subsampling.

    Returns (train_pool, val_files), disjoint and mode-balanced. A zero
    holdout explicitly disables validation and keeps every available file in
    the training pool.
    """
    available = sorted(available_files)
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError(
            f"holdout_fraction must be in [0, 1], got {holdout_fraction}"
        )
    if holdout_fraction == 0.0:
        return list(available), []
    if len(available) <= 1:
        return list(available), []
    n_val_target = max(1, int(round(holdout_fraction * len(available))))
    if max_val_files > 0:
        n_val_target = min(n_val_target, max_val_files)
    n_val_target = min(n_val_target, len(available) - 1)
    val_files = select_balanced_files(available, n_val_target, seed)
    val_set = set(val_files)
    train_pool = [f for f in available if f not in val_set]
    return train_pool, val_files


def split_sources_by_reference_mode_count(
    sources,
    *,
    holdout_fraction: float,
    max_val_files: int,
    val_seed: int,
    data_fraction: float = 1.0,
    data_seed: int = 0,
    train_rollouts_per_mode: int = 0,
    source_types: dict = None,
    singlearm_full_pool: bool = False,
):
    """Split rollout sources using source 0 as the per-mode reference.

    Each source is split into train/validation using the requested holdout
    settings. Source 0's post-holdout training density defines the target
    number of demos per filename mode. Every later source is sampled to the
    same per-mode density, so total demos scale with the number of modes
    present in that source.

    Args:
        sources: iterable of (source_name, files) pairs. Files may be basenames
            or absolute paths, but mode IDs must be present in the path/name.
        source_types: optional mapping from source name to a tag (e.g.
            "singlearm"/"twoarm"). Required when ``singlearm_full_pool`` is True.
        singlearm_full_pool: when True, sources tagged "singlearm" skip the
            per-mode cap, reference-density rebalancing, and ``data_fraction``
            subsampling, keeping their full post-holdout training pool.
    """
    sources = [(str(name), sorted(files)) for name, files in sources]
    if not sources:
        return {
            "train_files": [],
            "val_files": [],
            "sources": [],
            "reference_source": None,
            "reference_train_files": 0,
            "reference_modes": [],
            "target_per_mode": 0.0,
        }

    source_types = dict(source_types or {})
    if singlearm_full_pool and not any(source_types.get(name) == "singlearm" for name, _ in sources):
        raise ValueError(
            "singlearm_full_pool=True requires source_types tagging at least one source as 'singlearm'."
        )

    def _is_singlearm(name: str) -> bool:
        return singlearm_full_pool and source_types.get(name) == "singlearm"

    train_rollouts_per_mode = int(train_rollouts_per_mode)
    split_sources = []
    for name, files in sources:
        train_files, val_files = split_train_val_files(
            files,
            holdout_fraction=holdout_fraction,
            max_val_files=max_val_files,
            seed=val_seed,
        )
        if _is_singlearm(name):
            pass  # full post-holdout pool; skip data_fraction and per-mode cap
        elif train_rollouts_per_mode > 0:
            train_files = select_rollout_files_per_mode(
                train_files, train_rollouts_per_mode, data_seed
            )
        elif data_fraction < 1.0:
            train_files = subsample_rollout_files(train_files, data_fraction, data_seed)
        split_sources.append({
            "name": name,
            "available_files": sorted(files),
            "train_pool": sorted(train_files),
            "validation_files": sorted(val_files),
        })

    ref_train = split_sources[0]["train_pool"]
    ref_modes = sorted({mode_key(f) for f in ref_train})
    target_per_mode = len(ref_train) / max(1, len(ref_modes))

    selected_train = []
    selected_val = []
    for idx, source in enumerate(split_sources):
        train_pool = source["train_pool"]
        modes = sorted({mode_key(f) for f in train_pool})
        if _is_singlearm(source["name"]):
            selected = sorted(train_pool)
        elif train_rollouts_per_mode > 0:
            selected = sorted(train_pool)
        elif idx == 0 or not modes:
            selected = sorted(train_pool)
        else:
            target_count = int(round(target_per_mode * len(modes)))
            selected = select_balanced_files_by_key(
                train_pool, target_count, data_seed, mode_key
            )

        source["selected_train_files"] = selected
        source["modes"] = modes
        source["selected_train_count"] = len(selected)
        source["validation_count"] = len(source["validation_files"])
        source["source_type"] = source_types.get(source["name"])
        selected_train.extend(selected)
        selected_val.extend(source["validation_files"])

    return {
        "train_files": sorted(selected_train),
        "val_files": sorted(selected_val),
        "sources": split_sources,
        "reference_source": split_sources[0]["name"],
        "reference_train_files": len(ref_train),
        "reference_modes": ref_modes,
        "target_per_mode": float(target_per_mode),
        "train_rollouts_per_mode": train_rollouts_per_mode,
        "singlearm_full_pool": bool(singlearm_full_pool),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def build_sigma_grid(model, n: int):
    """Build n log-spaced sigmas spanning ±2 std of the model's noise distribution.

    Returns a tensor of shape (n, 1, 1, 1), or None when n <= 0.
    Each sigma broadcasts over (B, H, x_dim) when passed into `validation_loss`.
    """
    if n <= 0:
        return None
    p_mean = float(getattr(model, "p_mean", -1.2))
    p_std = float(getattr(model, "p_std", 1.2))
    device = getattr(model, "device", "cuda")
    log_sigma = torch.linspace(
        p_mean - 2.0 * p_std, p_mean + 2.0 * p_std, n, device=device
    )
    return log_sigma.exp().view(n, 1, 1, 1)


def run_validation(
    val_loader,
    val_step_fn,
    *,
    val_batches: int = 20,
    sigma_grid=None,
    noise_seed: int = 42,
    device: str = "cuda",
):
    """Run validation with seeded noise and optional sigma stratification.

    val_step_fn(batch, generator, sigma) -> (loss_scalar, n_samples)
        sigma is a (1,1,1) tensor when sigma_grid is provided, else None.

    Sample-weighted mean across all (batch × sigma) evaluations.
    Returns float, or None if no val loader / no batches.
    """
    if val_loader is None or val_step_fn is None:
        return None
    gen_device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(int(noise_seed))
    total = 0.0
    n_total = 0
    batch_count = 0
    for batch in val_loader:
        if sigma_grid is None:
            loss, n = val_step_fn(batch, gen, None)
            total += float(loss) * int(n)
            n_total += int(n)
        else:
            for i in range(sigma_grid.shape[0]):
                sigma = sigma_grid[i]  # (1, 1, 1)
                loss, n = val_step_fn(batch, gen, sigma)
                total += float(loss) * int(n)
                n_total += int(n)
        batch_count += 1
        if val_batches > 0 and batch_count >= val_batches:
            break
    if n_total == 0:
        return None
    return total / n_total


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def _save_checkpoint_history(history, checkpoint_dir: str):
    path = os.path.join(checkpoint_dir, "checkpoint_metrics.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def _unpack_train_result(result):
    if len(result) == 2:
        loss, grad = result
        return loss, grad, {}
    if len(result) == 3:
        loss, grad, metrics = result
        return loss, grad, dict(metrics)
    raise ValueError("train_step_fn must return (loss, grad) or (loss, grad, metrics)")


def history_to_curves(history):
    """Convert a history list into plot-ready arrays.

    Returns (train_xs, train_losses, train_grads, val_xs, val_losses).
    Train points are emitted once per x (deduped when log+checkpoint coincide).
    Val points are emitted only when val_loss is non-null.
    """
    seen = set()
    train_xs, train_losses, train_grads = [], [], []
    val_xs, val_losses = [], []
    for h in history:
        x = h.get("step", h.get("epoch"))
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            train_xs.append(x)
            train_losses.append(h["train_loss"])
            train_grads.append(h["train_grad_norm"])
        if h.get("val_loss") is not None:
            val_xs.append(x)
            val_losses.append(h["val_loss"])
    return train_xs, train_losses, train_grads, val_xs, val_losses


def train_step_loop(
    args,
    dataloader,
    train_step_fn,
    save_fn,
    *,
    checkpoint_dir: str,
    prefix: str,
    val_loader=None,
    val_step_fn=None,
    sigma_grid=None,
    log_fn=None,
    device: str = "cuda",
):
    """Step-based training loop with three independent cadences.

    Cadences (all expressed in optimizer steps):
        --log-every-steps        train metrics → wandb (cheap, no val/save)
        --val-every-steps        val metrics    → wandb + history JSON entry
        --checkpoint-every-steps .pt save       → disk + history JSON entry (val also runs)

    If --val-every-steps is 0, val uses the checkpoint cadence.
    History entries record `event=log|val|checkpoint` so plotting can pick rows.
    Returns the full history list.
    """
    if log_fn is None:
        log_fn = _default_log_fn
    max_steps = int(args.max_train_steps)
    log_every = max(1, int(getattr(args, "log_every_steps", 50)))
    checkpoint_every = int(args.checkpoint_every_steps)
    if checkpoint_every <= 0:
        checkpoint_every = max(1, max_steps // 5)
    val_every = int(getattr(args, "val_every_steps", 0))
    if val_every <= 0:
        val_every = checkpoint_every
    print(
        f"Step-based training: max_train_steps={max_steps}, "
        f"log_every={log_every}, val_every={val_every}, checkpoint_every={checkpoint_every}"
    )

    history: list[dict] = []
    train_iter = iter(dataloader)
    global_step = 0
    # Rolling window for the train-log cadence (resets after each log emit).
    log_loss_sum = 0.0
    log_grad_sum = 0.0
    log_steps = 0
    log_extra_sums = {}
    last_train_loss = float("nan")
    last_train_grad = float("nan")
    pbar = tqdm(total=max_steps, desc="Training steps")
    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)

        loss, grad, extra_metrics = _unpack_train_result(train_step_fn(batch))
        global_step += 1
        log_loss_sum += float(loss)
        log_grad_sum += float(grad)
        log_steps += 1
        for key, value in extra_metrics.items():
            log_extra_sums[key] = log_extra_sums.get(key, 0.0) + float(value)
        pbar.update(1)
        pbar.set_postfix({"loss": f"{float(loss):.4f}", "grad": f"{float(grad):.2f}"})

        is_last_step = global_step == max_steps
        emit_train = global_step % log_every == 0 or is_last_step
        emit_val = (val_step_fn is not None and val_loader is not None) and (
            global_step % val_every == 0 or is_last_step
        )
        emit_ckpt = global_step % checkpoint_every == 0 or is_last_step

        if emit_train:
            last_train_loss = log_loss_sum / max(log_steps, 1)
            last_train_grad = log_grad_sum / max(log_steps, 1)
            averaged_extras = {
                key: value / max(log_steps, 1)
                for key, value in log_extra_sums.items()
            }
            log_payload = {
                "train/loss": last_train_loss,
                "train/grad_norm": last_train_grad,
                "train/global_step": global_step,
            }
            log_payload.update({f"train/{k}": v for k, v in averaged_extras.items()})
            log_fn(log_payload, step=global_step)
            history.append({
                "step": int(global_step),
                "event": "log",
                "train_loss": float(last_train_loss),
                "train_grad_norm": float(last_train_grad),
                "val_loss": None,
                "checkpoint_path": None,
                "train_metrics": averaged_extras,
            })
            log_loss_sum = 0.0
            log_grad_sum = 0.0
            log_steps = 0
            log_extra_sums = {}

        event_train_loss = last_train_loss
        event_train_grad = last_train_grad
        if not emit_train and log_steps:
            event_train_loss = log_loss_sum / log_steps
            event_train_grad = log_grad_sum / log_steps

        val_loss = None
        if emit_val or emit_ckpt:
            val_loss = run_validation(
                val_loader, val_step_fn,
                val_batches=int(getattr(args, "val_batches", 20)),
                sigma_grid=sigma_grid,
                noise_seed=int(getattr(args, "val_noise_seed", 42)),
                device=device,
            )
            if val_loss is not None:
                log_fn({"val/loss": val_loss}, step=global_step)

        if emit_ckpt:
            ckpt = os.path.join(checkpoint_dir, f"{prefix}_step{global_step}.pt")
            save_fn(ckpt)
            history.append({
                "step": int(global_step),
                "event": "checkpoint",
                "train_loss": float(event_train_loss),
                "train_grad_norm": float(event_train_grad),
                "val_loss": None if val_loss is None else float(val_loss),
                "checkpoint_path": ckpt,
            })
            _save_checkpoint_history(history, checkpoint_dir)
            print(
                f"Step {global_step}: train_loss={event_train_loss:.6f}, "
                f"grad={event_train_grad:.4f}, val_loss={val_loss}"
            )
        elif emit_val and val_loss is not None:
            history.append({
                "step": int(global_step),
                "event": "val",
                "train_loss": float(event_train_loss),
                "train_grad_norm": float(event_train_grad),
                "val_loss": float(val_loss),
                "checkpoint_path": None,
            })
            _save_checkpoint_history(history, checkpoint_dir)
    pbar.close()
    return history


def train_epoch_loop(
    args,
    dataloader,
    train_step_fn,
    save_fn,
    *,
    checkpoint_dir: str,
    prefix: str,
    val_loader=None,
    val_step_fn=None,
    sigma_grid=None,
    log_fn=None,
    device: str = "cuda",
):
    """Epoch-based training loop. Same callback shape as train_step_loop."""
    if log_fn is None:
        log_fn = _default_log_fn
    history: list[dict] = []
    for epoch in range(int(args.epochs)):
        loss_epoch = 0.0
        grad_epoch = 0.0
        steps = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            loss, grad, _ = _unpack_train_result(train_step_fn(batch))
            loss_epoch += float(loss)
            grad_epoch += float(grad)
            steps += 1
            pbar.set_postfix({"loss": f"{float(loss):.4f}", "grad": f"{float(grad):.2f}"})

        avg_loss = loss_epoch / max(steps, 1)
        avg_grad = grad_epoch / max(steps, 1)
        log_fn(
            {"train/loss": avg_loss, "train/grad_norm": avg_grad, "epoch": epoch + 1},
            step=epoch + 1,
        )
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.6f}, avg_grad={avg_grad:.4f}")

        if (epoch + 1) % int(args.save_every) == 0:
            val_loss = run_validation(
                val_loader, val_step_fn,
                val_batches=int(getattr(args, "val_batches", 20)),
                sigma_grid=sigma_grid,
                noise_seed=int(getattr(args, "val_noise_seed", 42)),
                device=device,
            )
            ckpt = os.path.join(checkpoint_dir, f"{prefix}_epoch{epoch + 1}.pt")
            save_fn(ckpt)
            history.append({
                "epoch": int(epoch + 1),
                "train_loss": float(avg_loss),
                "train_grad_norm": float(avg_grad),
                "val_loss": None if val_loss is None else float(val_loss),
                "checkpoint_path": ckpt,
            })
            _save_checkpoint_history(history, checkpoint_dir)
            if val_loss is not None:
                log_fn({"val/loss": val_loss, "epoch": epoch + 1}, step=epoch + 1)
    return history

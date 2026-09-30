"""Lightweight wandb integration for Co-Diff training scripts.

Usage in a training script:
    from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb

    # In parse_args():
    add_wandb_args(parser)

    # In main(), after setting up the model:
    init_wandb(args, extra_config={...})

    # In the training loop:
    log_metrics({"train/loss": avg_loss, "train/grad_norm": avg_grad}, step=epoch)

    # At the end:
    finish_wandb()
"""

import argparse
import os

_wandb = None  # lazy import


def _import_wandb():
    global _wandb
    if _wandb is None:
        import wandb

        _wandb = wandb
    return _wandb


def add_wandb_args(parser: argparse.ArgumentParser):
    """Add wandb-related CLI arguments to an existing argparse parser."""
    group = parser.add_argument_group("wandb")
    group.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    group.add_argument(
        "--wandb-project",
        type=str,
        default="co-diff",
        help="wandb project name (default: co-diff).",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="wandb entity (team or user). Uses default if not set.",
    )
    group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="wandb run name. Auto-generated if not set.",
    )
    group.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="Optional tags for the wandb run.",
    )
    group.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="wandb group for organizing related runs (e.g. an ablation study).",
    )
    group.add_argument(
        "--wandb-id-file",
        type=str,
        default=None,
        help="Write the wandb run ID to this file after init (for sweep orchestrators).",
    )


def init_wandb(args: argparse.Namespace, extra_config=None):
    """Initialize a wandb run if --wandb is set.

    Logs all argparse arguments plus any extra_config dict as the run config.
    """
    if not getattr(args, "wandb", False):
        return None

    wandb = _import_wandb()

    config = vars(args).copy()
    # Remove wandb meta-args from the logged config
    for key in ("wandb", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_tags", "wandb_group", "wandb_id_file"):
        config.pop(key, None)
    if extra_config:
        config.update(extra_config)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        group=args.wandb_group,
        config=config,
    )

    # Write run ID to file if requested (for sweep orchestrators)
    id_file = getattr(args, "wandb_id_file", None)
    if run and id_file:
        os.makedirs(os.path.dirname(id_file) or ".", exist_ok=True)
        with open(id_file, "w") as f:
            f.write(run.id)

    return run


def log_metrics(metrics: dict, step=None):
    """Log metrics to wandb if a run is active."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.log(metrics, step=step)


def get_run_id():
    """Return the active wandb run ID, or None."""
    if _wandb is not None and _wandb.run is not None:
        return _wandb.run.id
    return None


def resume_wandb(run_id, project="co-diff-sweep", entity=None):
    """Resume an existing wandb run (e.g. to append eval metrics post-training)."""
    wandb = _import_wandb()
    return wandb.init(id=run_id, project=project, entity=entity, resume="allow")


def finish_wandb():
    """Finish the wandb run if one is active."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.finish()

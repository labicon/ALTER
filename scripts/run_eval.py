#!/usr/bin/env python
"""Multi-seed evaluation orchestrator.

Loads a model once, runs it for N seeds with metrics collection enabled,
aggregates the results, and writes structured JSON.

Usage (called by scripts/eval.sh, or directly):
    python scripts/run_eval.py --mode coord --task cube --seeds 0,1,2,3,4 \
        --output-dir results/eval --no-video [--use-landing-pad] \
        [-- extra_args_forwarded_to_player]
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from scripts.experiment_contract import (
    EVALUATION_PROFILES,
    METRIC_SCHEMA_VERSION,
    compact_contract_summary,
    discover_contract,
    eval_context_from_args,
    failure_breakdown_by_mode,
    git_summary,
    success_by_mode,
    validate_eval_contract,
)
from src.temporal import frame_offsets_from_stats

CODIFF_DATA_ROOT = (
    os.environ.get("CODIFF_DATA_ROOT")
    or os.environ.get("DATASET_ROOT")
    or os.path.abspath(os.path.join(REPO_ROOT, "release-artifacts"))
)
DATASET_ROOT = os.environ.get("DATASET_ROOT", CODIFF_DATA_ROOT)
CHECKPOINT_ROOT = os.environ.get(
    "CHECKPOINT_ROOT", os.path.join(CODIFF_DATA_ROOT, "checkpoints", "arm")
)
ROLLOUT_ROOT = os.environ.get(
    "ROLLOUT_ROOT", os.path.join(CODIFF_DATA_ROOT, "envs", "arm", "data_gen", "rollouts")
)
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(REPO_ROOT, "public-validation", "results"))
DEBUG_ROOT = os.environ.get("DEBUG_ROOT", os.path.join(REPO_ROOT, "public-validation", "debug"))
WANDB_DIR = os.environ.get("WANDB_DIR", os.path.join(REPO_ROOT, "public-validation", "wandb"))

os.environ.setdefault("CODIFF_DATA_ROOT", CODIFF_DATA_ROOT)
os.environ.setdefault("DATASET_ROOT", DATASET_ROOT)
os.environ.setdefault("CHECKPOINT_ROOT", CHECKPOINT_ROOT)
os.environ.setdefault("ROLLOUT_ROOT", ROLLOUT_ROOT)
os.environ.setdefault("RESULTS_ROOT", RESULTS_ROOT)
os.environ.setdefault("DEBUG_ROOT", DEBUG_ROOT)
os.environ.setdefault("WANDB_DIR", WANDB_DIR)


def resolve_artifact_path(path, default_root):
    if os.path.isabs(path):
        return path
    norm = os.path.normpath(path)
    if norm in (".", ""):
        return default_root
    parts = norm.split(os.sep)
    artifact_roots = {
        "checkpoints": os.path.join(CODIFF_DATA_ROOT, "checkpoints"),
        "debug": DEBUG_ROOT,
        "results": RESULTS_ROOT,
        "wandb": WANDB_DIR,
    }
    if parts[0] in artifact_roots:
        return os.path.join(artifact_roots[parts[0]], *parts[1:])
    return os.path.join(default_root, norm)



def _aggregate(per_seed_metrics):
    """Compute mean/std for numeric metrics, fraction for booleans."""
    if not per_seed_metrics:
        return {}

    all_keys = set()
    for m in per_seed_metrics:
        all_keys.update(m.keys())

    agg = {}
    for key in sorted(all_keys):
        vals = [m[key] for m in per_seed_metrics if key in m]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            agg[key] = {"fraction": float(sum(vals)) / len(vals), "count": sum(vals), "total": len(vals)}
        elif isinstance(vals[0], (int, float)):
            arr = np.array(vals, dtype=np.float64)
            agg[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        # Skip non-numeric
    return agg


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation orchestrator.")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["coord", "coord-shoulderonly", "fromscratch", "fromscratch-shoulderonly"],
                        help="Use the fromscratch inference route for FS and full-policy FT checkpoints")
    parser.add_argument("--task", type=str, default="placewipe",
                        choices=["placewipe"])
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "dinov2"])
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated seeds (e.g. 0,1,2,3,4)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(RESULTS_ROOT, "eval"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--sponge-scale", type=float, default=1.0,
                        help=("Scale factor for the sponge size in the three-arm wipe "
                              "env. 1.0 (default) keeps the original size. Must match "
                              "the sponge scale the policy was trained on (e.g. 0.85 "
                              "for data generated with --sponge-scale 0.85)."))
    parser.add_argument("--min-video-free-gb", type=float, default=2.0,
                        help=("Skip saving a rollout video when the output filesystem has "
                              "less than this much free space. Default: 2.0"))
    parser.add_argument("--strict-video-errors", action="store_true",
                        help="Raise video writing errors instead of warning and continuing evaluation.")
    parser.add_argument("--use-landing-pad", action="store_true")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--task-variant", type=str, default="auto",
                        choices=["auto", "greenfront", "redfront"],
                        help="Scene/task variant. 'auto' infers from stats.")
    parser.add_argument("--env-variant", type=str, default="standard",
                        choices=["standard", "hard"],
                        help="Placewipe env geometry variant.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g. cuda, cuda:0, cpu). Auto-detects if not set.")
    parser.add_argument("--replan-freq", type=int, default=None,
                        help="Override replan frequency (default: 10)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeat runs per seed (stochastic sampling noise). Default: 1")
    parser.add_argument("--mode-list", type=str, default=None,
                        help="Comma-separated placewipe modes (e.g. '0,1,2,3'). Only used when --task=placewipe.")
    # Sweep overrides — allow pointing at arbitrary checkpoint/stats files
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Override auto-derived checkpoint path (for sweep eval)")
    parser.add_argument("--adapter-stats-path", type=str, required=True,
                        help="Override auto-derived adapter stats path (for sweep eval)")
    parser.add_argument("--base-model-path", type=str, default=None,
                        help="Override auto-derived base model path (e.g. epoch300 instead of epoch200)")
    parser.add_argument("--base-stats-path", type=str, default=None,
                        help="Override auto-derived base stats path")
    parser.add_argument("--selector-path", type=str, default=None,
                        help="Optional learned base-vs-coordination selector checkpoint")
    parser.add_argument("--selector-mode", type=str, default="coord",
                        choices=["coord", "base", "auto"],
                        help="Routing mode for selector-capable players")
    parser.add_argument("--selector-threshold", type=float, default=0.5,
                        help="P(coordination) threshold used when --selector-mode auto")
    parser.add_argument("--eval-purpose", type=str, default=None,
                        help="Purpose label written to eval JSON (e.g. headline_hard, cleanup_standard, diagnostic).")
    parser.add_argument("--strict-experiment-contract", action="store_true",
                        help="Fail fast when a discovered experiment contract mismatches this eval.")
    parser.add_argument(
        "--experiment-profile", choices=sorted(EVALUATION_PROFILES), default=None,
        help="Named evaluation profile. Required with --strict-experiment-contract.",
    )
    # Sealed evaluators own their immutable contract and pass the fixed
    # sampling seed derivation into this generic per-episode execution loop.
    parser.add_argument("--sealed-sampling-seed-base", type=int, default=None,
                        help="Fixed seed base recorded by a sealed evaluation contract.")
    # W&B resume — log eval metrics to an existing training run
    parser.add_argument("--eval-epoch", type=int, default=None,
                        help="Epoch of the checkpoint being evaluated (included in W&B metric keys)")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="Resume this W&B run and log eval metrics to it")
    parser.add_argument("--wandb-project", type=str, default="co-diff-sweep",
                        help="W&B project for resumed run")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity for resumed run")
    parser.add_argument("--wandb-sweep-config", type=str, default=None,
                        help="JSON string of sweep HP config to push to W&B run config")
    parser.add_argument("--wandb-metric-prefix", type=str, default="eval",
                        help="Prefix for W&B metric keys (default: 'eval', use 'forgetting' for forgetting test)")
    # Extra args forwarded to the player
    parser.add_argument("extra", nargs="*", help="Extra args forwarded to sampling script")
    args = parser.parse_args()
    inputs = [("--checkpoint-path", args.checkpoint_path), ("--adapter-stats-path", args.adapter_stats_path)]
    if args.mode.startswith("coord"):
        inputs += [("--base-model-path", args.base_model_path), ("--base-stats-path", args.base_stats_path)]
    for flag, path in inputs:
        if not path or not os.path.isfile(path):
            parser.error(f"{flag}: missing artifact {path!r}. Download the matching release bundle, materialize its inputs, and provide an existing file.")
    return args


def main():
    args = parse_args()
    args.output_dir = resolve_artifact_path(args.output_dir, RESULTS_ROOT)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    bb_suffix = "" if args.backbone == "resnet18" else f"_{args.backbone}"

    # Import the appropriate player factory based on mode
    per_seed = []
    player = None
    run_kwargs_base = {}

    # Determine paths based on mode
    BASE_MODEL = os.path.join(
        CHECKPOINT_ROOT,
        f"singlearm_mixedfront_directgrasp_e2e_shoulder{bb_suffix}",
        "singlearm_mixedfront_e2e_shoulder_epoch200.pt",
    )
    BASE_STATS = f"stats/arm/singlearm_mixedfront_directgrasp_e2e_shoulder{bb_suffix}_stats.pkl"
    BASE_MODEL_SO = os.path.join(
        CHECKPOINT_ROOT,
        f"singlearm_mixedfront_directgrasp_e2e_shoulderonly_h25{bb_suffix}",
        "singlearm_mixedfront_e2e_shoulder_epoch200.pt",
    )
    BASE_STATS_SO = f"stats/arm/singlearm_mixedfront_directgrasp_shoulderonly_e2e{bb_suffix}_stats.pkl"

    is_shoulderonly = args.mode.endswith("-shoulderonly")
    base_mode = args.mode.replace("-shoulderonly", "")
    base_model_path = BASE_MODEL_SO if is_shoulderonly else BASE_MODEL
    base_stats_path = BASE_STATS_SO if is_shoulderonly else BASE_STATS
    if args.base_model_path:
        base_model_path = args.base_model_path
    if args.base_stats_path:
        base_stats_path = args.base_stats_path

    import pickle as pkl
    import torch
    from robosuite.controllers import load_composite_controller_config

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller_config = load_composite_controller_config(
        robot="Kinova3",
        controller="envs/arm/data_gen/kinova.json",
    )

    if args.task == "placewipe":
        # ---- Two-arm place+wipe dispatch (coord / lora / fromscratch) ----
        from envs.arm.data_gen.utils.twoarm_wipe_env import TwoArmPlaceWipe
        if args.env_variant == "hard":
            from envs.arm.data_gen.utils.twoarm_wipe_env_hard import TwoArmPlaceWipeHard
        if args.env_variant == "hard":
            placewipe_env_cls = TwoArmPlaceWipeHard
        else:
            placewipe_env_cls = TwoArmPlaceWipe

        # Combined single-arm placewipe base path (only needed for coord/lora).
        if not args.base_model_path:
            base_model_path = (
                f"{CHECKPOINT_ROOT}/singlearm_placewipe_combined_e2e_shoulderonly{bb_suffix}/"
                f"singlearm_mixedfront_e2e_shoulder_final.pt"
            )
        if not args.base_stats_path:
            base_stats_path = (
                f"{CHECKPOINT_ROOT}/singlearm_placewipe_combined_e2e_shoulderonly{bb_suffix}/"
                f"singlearm_mixedfront_e2e_shoulder_stats.pkl"
            )

        env = placewipe_env_cls(
            robots=["Kinova3", "Kinova3"], gripper_types="default",
            controller_configs=controller_config,
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=False, render_camera=None,
            control_freq=20, horizon=4000,
        )

        if base_mode == "coord":
            with open(base_stats_path, "rb") as f:
                base_stats = pkl.load(f)
            backbone = str(base_stats.get("backbone", "resnet18"))
            num_cameras = int(base_stats.get("num_cameras", 2))
            frame_offsets = frame_offsets_from_stats(base_stats)

            from src.image_diffusion import ImageConditional_ODE
            from src.image_coordination import (
                ImageCoordinationHead, LEGACY_DECODER_EXECUTION,
                plan_memory_kwargs_from_stats, side_net_kwargs_from_stats,
            )
            from envs.arm.sample.sample_coord_twoarm_placewipe_shoulder import (
                CoordinatedPlaceWipeMPCPlayer,
            )
            from src.policy_selector import load_image_policy_selector

            base_model = ImageConditional_ODE(
                x_dim=7, sigma_data=float(base_stats.get("sigma_data", 1.0)),
                d_model=int(base_stats.get("d_model", 256)),
                n_heads=int(base_stats.get("n_heads", 4)),
                depth=int(base_stats.get("depth", 3)),
                dim_feedforward=int(base_stats.get("dim_feedforward", 1024)),
                horizon=int(base_stats.get("horizon", 20)),
                device=device, num_cameras=num_cameras, backbone=backbone,
                frame_offsets=frame_offsets,
            )
            if not base_model.load(base_model_path):
                raise FileNotFoundError(f"Base model not found: {base_model_path}")
            base_model.F.eval()

            so_tag = "shoulderonly" if is_shoulderonly else "shoulder"
            coord_head_path = args.checkpoint_path or (
                f"{CHECKPOINT_ROOT}/coordination_placewipe_e2e_{so_tag}{bb_suffix}/"
                f"coord_head_placewipe_final.pt"
            )
            coord_stats_path = args.adapter_stats_path or (
                f"{CHECKPOINT_ROOT}/coordination_placewipe_e2e_{so_tag}{bb_suffix}/"
                f"coord_head_placewipe_stats.pkl"
            )
            with open(coord_stats_path, "rb") as f:
                coord_stats = pkl.load(f)

            side_net_kwargs = side_net_kwargs_from_stats(coord_stats, base_model.F.tokens_per_camera)
            coord_head = ImageCoordinationHead(
                x_dim=7, base_d_model=int(coord_stats.get("base_d_model", 256)),
                d_model=int(coord_stats["head_d_model"]),
                n_heads=int(coord_stats["head_n_heads"]),
                depth=int(coord_stats["head_depth"]),
                dim_feedforward=int(coord_stats["head_dim_feedforward"]),
                horizon=int(base_stats.get("horizon", 20)),
                sigma_data=float(base_stats.get("sigma_data", 1.0)),
                num_cameras=num_cameras,
                **side_net_kwargs,
                decoder_execution=str(coord_stats.get(
                    "decoder_execution", LEGACY_DECODER_EXECUTION)),
                decoder_conditioning=str(coord_stats.get("decoder_conditioning", "pooled")),
                **plan_memory_kwargs_from_stats(coord_stats),
                frame_offsets=frame_offsets,
            ).to(device)
            coord_head.load(coord_head_path, device=device)
            coord_head.eval()

            selector = None
            if args.selector_mode == "auto":
                if args.selector_path is None:
                    raise ValueError("--selector-mode auto requires --selector-path")
                selector = load_image_policy_selector(args.selector_path, device=device)
                print(f"Loaded policy selector: {args.selector_path}")

            player = CoordinatedPlaceWipeMPCPlayer(
                env=env, base_model=base_model, coord_head=coord_head,
                stats=coord_stats, device=device, num_cameras=num_cameras,
                selector=selector,
                selector_mode=args.selector_mode,
                selector_threshold=args.selector_threshold,
            )
            run_kwargs_base = {
                "max_steps": args.max_steps, "replan_freq": 10, "guidance_w": 1.2,
                "sample_N": 50, "warm_start_skip": 0.0,
            }

        elif base_mode == "fromscratch":
            so_tag = "shoulderonly" if is_shoulderonly else "shoulder"
            stats_path = args.adapter_stats_path or (
                f"{CHECKPOINT_ROOT}/twoarm_fromscratch_placewipe_e2e_{so_tag}{bb_suffix}/"
                f"twoarm_fromscratch_placewipe_stats.pkl"
            )
            model_path = args.checkpoint_path or (
                f"{CHECKPOINT_ROOT}/twoarm_fromscratch_placewipe_e2e_{so_tag}{bb_suffix}/"
                f"twoarm_fromscratch_placewipe_final.pt"
            )
            with open(stats_path, "rb") as f:
                stats = pkl.load(f)
            backbone = str(stats.get("backbone", "resnet18"))
            num_cameras = int(stats.get("num_cameras", 2))

            from src.image_diffusion import ImageConditional_ODE
            from envs.arm.sample.sample_fromscratch_twoarm_placewipe_shoulder import (
                FromScratchPlaceWipeMPCPlayer,
            )
            frame_offsets = frame_offsets_from_stats(stats)

            model = ImageConditional_ODE(
                x_dim=7, sigma_data=float(stats.get("sigma_data", 1.0)),
                d_model=int(stats.get("d_model", 256)),
                n_heads=int(stats.get("n_heads", 4)),
                depth=int(stats.get("depth", 3)),
                dim_feedforward=int(stats.get("dim_feedforward", 1024)),
                horizon=int(stats.get("horizon", 20)),
                device=device, num_cameras=num_cameras, backbone=backbone,
                frame_offsets=frame_offsets,
            )
            if not model.load(model_path):
                raise FileNotFoundError(f"Fromscratch model not found: {model_path}")
            model.F.eval()

            player = FromScratchPlaceWipeMPCPlayer(
                env=env, model=model, stats=stats, device=device,
            )
            run_kwargs_base = {
                "max_steps": args.max_steps, "replan_freq": 10, "guidance_w": 1.2,
                "sample_N": 50, "warm_start_skip": 0.0,
            }

        else:
            print(f"Unknown placewipe mode: {args.mode} (expected coord/lora/fromscratch).")
            sys.exit(1)

    # Apply --replan-freq override if specified
    if args.replan_freq is not None:
        run_kwargs_base["replan_freq"] = args.replan_freq

    # Build the output directory after the effective replan frequency is known.
    effective_rf = run_kwargs_base.get("replan_freq", 10)
    tag = f"{args.mode}_{args.task}_{args.backbone}_rf{effective_rf}"
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    video_dir = os.path.join(out_dir, "videos") if not args.no_video else None
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    progress_path = os.path.join(out_dir, f"eval_progress_{run_timestamp}.jsonl")
    out_path = os.path.join(out_dir, f"eval_{run_timestamp}.json")
    collisions = [path for path in (progress_path, out_path) if os.path.exists(path)]
    if collisions:
        raise FileExistsError(f"Refusing evaluation output collision: {collisions}")

    # ------------------------------------------------------------------ #
    # Run evaluation seeds
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Evaluating: mode={args.mode}, task={args.task}, backbone={args.backbone}")
    print(f"Seeds: {seeds} | Repeats per seed: {args.repeats} | Total runs: {len(seeds) * args.repeats}")
    print(f"{'='*60}\n")

    # When --no-video, disable frame capture to save time and avoid imageio errors
    if args.no_video and hasattr(player, '_capture_video_frame'):
        player._capture_video_frame = lambda: None
        player._save_video = lambda path: None
    elif hasattr(player, '_save_video') and not args.strict_video_errors:
        original_save_video = player._save_video
        min_video_free_bytes = max(0, int(args.min_video_free_gb * (1024 ** 3)))

        def _safe_save_video(path):
            video_parent = os.path.dirname(path) or "."
            os.makedirs(video_parent, exist_ok=True)
            try:
                free_bytes = shutil.disk_usage(video_parent).free
            except OSError as e:
                print(
                    f"[video] Could not check free space for {video_parent}: {e}. "
                    f"Skipping {path}."
                )
                return

            if free_bytes < min_video_free_bytes:
                free_gb = free_bytes / (1024 ** 3)
                print(
                    f"[video] Skipping {path}: only {free_gb:.2f} GiB free "
                    f"(threshold {args.min_video_free_gb:.2f} GiB)."
                )
                return

            try:
                original_save_video(path)
            except Exception as e:
                print(
                    f"[video] Failed to save {path}: {type(e).__name__}: {e}. "
                    "Continuing without this video."
                )
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError as cleanup_error:
                    print(f"[video] Could not remove incomplete video {path}: {cleanup_error}")

        player._save_video = _safe_save_video

    # ------------------------------------------------------------------ #
    # Pre-parse W&B config — one run per (seed, repeat) logged inline.
    # Runs are grouped via W&B `group=` so per-config aggregation still
    # works in the UI (Group by `group` → mean/std across episodes).
    # ------------------------------------------------------------------ #
    wandb_module = None
    wandb_base_cfg = None
    wandb_run_name_base = None
    wandb_group = None
    wandb_tags = []
    wandb_done = set()  # (seed, repeat) pairs already logged for this group
    if args.wandb_sweep_config:
        import wandb as wandb_module  # type: ignore
        sweep_cfg = json.loads(args.wandb_sweep_config)
        rf_cfg = run_kwargs_base.get("replan_freq", 10)
        ep_cfg = args.eval_epoch
        sweep_cfg["eval_epoch"] = ep_cfg
        sweep_cfg["replan_freq"] = rf_cfg
        sweep_cfg["repeats_total"] = args.repeats
        wandb_tags = list(sweep_cfg.pop("_tags", []))
        base_name = sweep_cfg.pop("_run_name", "eval")
        wandb_run_name_base = (
            f"{base_name}-rf{rf_cfg}" if ep_cfg is None
            else f"{base_name}-ep{ep_cfg}-rf{rf_cfg}"
        )
        wandb_group = wandb_run_name_base  # all episodes of this config share a group
        wandb_base_cfg = sweep_cfg

        # --- Resume: query W&B for already-completed (mode, seed, repeat) in this group ---
        # We parse run names (always populated on the iterator) instead of
        # r.config, which is lazily loaded and empty on the bulk list.
        # Name format (cube/pot): "{base}-s{seed}-r{repeat}"
        # Name format (placewipe): "{base}-m{mode}-s{seed}-r{repeat}"
        import re as _re
        if args.task == "placewipe":
            name_re = _re.compile(rf"^{_re.escape(wandb_run_name_base)}-m(\d+)-s(\d+)-r(\d+)$")
        else:
            name_re = _re.compile(rf"^{_re.escape(wandb_run_name_base)}-s(\d+)-r(\d+)$")
        try:
            api = wandb_module.Api()
            entity = args.wandb_entity or api.default_entity
            existing = api.runs(
                f"{entity}/{args.wandb_project}",
                filters={"group": wandb_group, "state": "finished"},
            )
            for r in existing:
                m = name_re.match(r.name or "")
                if m:
                    if args.task == "placewipe":
                        wandb_done.add((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                    else:
                        wandb_done.add((int(m.group(1)), int(m.group(2))))
            if wandb_done:
                print(f"[resume] {len(wandb_done)} episodes already logged "
                      f"for group={wandb_group} — will skip those.")
        except Exception as e:
            print(f"[resume] Could not query W&B for existing runs: {e}. "
                  "Will run full sweep.")

    # Outer mode loop is meaningful for placewipe and wipe; other tasks collapse
    # to a single sentinel iteration (mode=None) that doesn't touch env state
    # or run-name suffixes.
    if args.task in ("placewipe", "wipe"):
        if args.mode_list:
            modes_to_run = [int(m.strip()) for m in args.mode_list.split(",") if m.strip() != ""]
        else:
            modes_to_run = [0, 1, 2, 3]
    else:
        modes_to_run = [None]

    local_paths = locals()
    evaluated_checkpoint_path = (
        args.checkpoint_path
        or local_paths.get("coord_head_path")
        or local_paths.get("adapter_path")
        or local_paths.get("model_path")
        or base_model_path
    )
    evaluated_stats_path = (
        args.adapter_stats_path
        or local_paths.get("coord_stats_path")
        or local_paths.get("adapter_stats_path")
        or local_paths.get("stats_path")
        or base_stats_path
    )
    contract_path, experiment_contract = discover_contract(
        checkpoint_path=evaluated_checkpoint_path,
        stats_path=evaluated_stats_path,
    )
    eval_context = eval_context_from_args(
        args,
        env_class=type(player.env).__name__ if hasattr(player, "env") else None,
        replan_freq=run_kwargs_base.get("replan_freq", 10),
        modes=modes_to_run,
    )
    eval_context["checkpoint_path"] = os.path.abspath(evaluated_checkpoint_path) if evaluated_checkpoint_path else None
    eval_context["adapter_stats_path"] = os.path.abspath(evaluated_stats_path) if evaluated_stats_path else None
    eval_context["base_model_path"] = os.path.abspath(base_model_path) if base_model_path else None
    eval_context["base_stats_path"] = os.path.abspath(base_stats_path) if base_stats_path else None
    evaluated_frame_offsets = [0]
    if evaluated_stats_path and os.path.isfile(evaluated_stats_path):
        with open(evaluated_stats_path, "rb") as f:
            evaluated_frame_offsets = list(frame_offsets_from_stats(pkl.load(f)))
    eval_context["frame_offsets"] = evaluated_frame_offsets
    if args.strict_experiment_contract and args.experiment_profile is None:
        raise SystemExit("--strict-experiment-contract requires --experiment-profile")
    contract_mismatches = validate_eval_contract(
        experiment_contract, eval_context, profile=args.experiment_profile
    )
    if experiment_contract is None:
        contract_match = None
        print("[contract] WARNING: no experiment_contract.json found beside checkpoint/stats; continuing as legacy eval.")
    else:
        contract_match = not contract_mismatches
        if contract_match:
            print(f"[contract] OK: {contract_path}")
        else:
            print(f"[contract] WARNING: mismatches for {contract_path}: {contract_mismatches}")

    explicit_eval_purpose = args.eval_purpose
    profile_purpose = (
        EVALUATION_PROFILES[args.experiment_profile]["purpose"]
        if args.experiment_profile else None
    )
    if explicit_eval_purpose and explicit_eval_purpose.startswith("headline"):
        if args.experiment_profile != "headline_hard":
            raise SystemExit("Only the headline_hard profile may use a headline evaluation purpose.")
        if contract_mismatches:
            raise SystemExit("A contract mismatch cannot be used for headline evaluation.")
    if explicit_eval_purpose is not None:
        eval_purpose = explicit_eval_purpose
    elif profile_purpose is not None:
        eval_purpose = profile_purpose
    elif contract_mismatches:
        eval_purpose = "diagnostic_contract_mismatch"
    else:
        eval_purpose = "diagnostic"
    if args.strict_experiment_contract and contract_mismatches:
        raise SystemExit(
            "Experiment contract mismatch under --strict-experiment-contract: "
            + ", ".join(contract_mismatches)
        )

    for mode_id in modes_to_run:
        # Placewipe: the env carries (dest_mode, sponge_mode) state; set it
        # once per mode block before the seed loop. Three-arm wipe uses
        # (tray_dest, sponge_dest) and is set inside player.reset() via the
        # `mode=mode_id` kwarg threaded through run_kwargs below.
        if (mode_id is not None and args.task == "placewipe"
                and hasattr(player, 'env') and hasattr(player.env, 'set_mode')):
            dest_mode = mode_id // 2
            sponge_mode = mode_id % 2
            player.env.set_mode(dest_mode=dest_mode, sponge_mode=sponge_mode)
            print(f"\n>>> Placewipe mode={mode_id} (dest={dest_mode}, sponge={sponge_mode})")
        elif mode_id is not None and args.task == "wipe":
            tray_dest = mode_id // 2
            sponge_dest = mode_id % 2
            print(f"\n>>> Wipe mode={mode_id} (tray={tray_dest}, sponge={sponge_dest})")

        for seed in seeds:
            # Set obstacle side per seed — only meaningful when the scene has a
            # single active obstacle (greenfront/redfront). Mirrors the
            # `if args.task_variant != "none"` gate in the sample scripts.
            if (args.task_variant in ("greenfront", "redfront")
                    and hasattr(player, 'env')
                    and hasattr(player.env, 'set_obstacle_side')):
                side = str(np.random.default_rng(seed).choice(["left", "right"]))
                player.env.set_obstacle_side(side)
            else:
                side = None

            for repeat in range(args.repeats):
                resume_key = (mode_id, seed, repeat) if mode_id is not None else (seed, repeat)
                if resume_key in wandb_done:
                    print(f"[resume] Skipping {resume_key} — already logged.")
                    continue

                if mode_id is not None:
                    label = f"Mode {mode_id} seed {seed} rep {repeat}"
                else:
                    label = f"Seed {seed}" if args.repeats == 1 else f"Seed {seed} rep {repeat}"
                print(f"\n--- {label} ---")
                if side is not None:
                    print(f"  Obstacle side: {side}")

                if mode_id is not None:
                    vp_name = f"mode{mode_id}_seed{seed}_rep{repeat}.mp4"
                else:
                    vp_name = f"seed_{seed}.mp4" if args.repeats == 1 else f"seed_{seed}_rep{repeat}.mp4"
                vp = os.path.join(video_dir, vp_name) if video_dir else "eval_dummy.mp4"
                if video_dir:
                    os.makedirs(video_dir, exist_ok=True)

                run_kwargs = {**run_kwargs_base, "seed": seed, "video_path": vp, "collect_metrics": True}
                # Wipe player.run accepts mode= directly (sets env via reset()).
                if args.task == "wipe" and mode_id is not None:
                    run_kwargs["mode"] = int(mode_id)
                if args.sealed_sampling_seed_base is not None:
                    sealed_seed = (int(args.sealed_sampling_seed_base) + 1000003 * int(seed)
                                   + 10007 * int(mode_id or 0) + 101 * int(repeat))
                    torch.manual_seed(sealed_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sealed_seed)
                metrics = player.run(**run_kwargs)
                if metrics is None:
                    metrics = {}
                metrics["seed"] = seed
                metrics["repeat"] = repeat
                if mode_id is not None:
                    metrics["mode"] = mode_id
                per_seed.append(metrics)
                with open(progress_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(f"  Metrics: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, indent=None)}")

                # --- Per-episode W&B run (fault-tolerant, with retries) ---
                if wandb_module is not None:
                    ep_cfg = {**wandb_base_cfg, "seed": seed, "repeat": repeat}
                    if mode_id is not None:
                        ep_cfg["mode"] = mode_id
                    log_payload = {}
                    for k, v in metrics.items():
                        if isinstance(v, bool):
                            log_payload[f"{args.wandb_metric_prefix}/{k}"] = int(v)
                        elif isinstance(v, (int, float)):
                            log_payload[f"{args.wandb_metric_prefix}/{k}"] = v

                    if mode_id is not None:
                        run_name = f"{wandb_run_name_base}-m{mode_id}-s{seed}-r{repeat}"
                    else:
                        run_name = f"{wandb_run_name_base}-s{seed}-r{repeat}"

                    logged = False
                    for attempt in range(3):
                        try:
                            wandb_module.init(
                                project=args.wandb_project,
                                entity=args.wandb_entity,
                                name=run_name,
                                group=wandb_group,
                                config=ep_cfg,
                                tags=wandb_tags,
                                reinit=True,
                                settings=wandb_module.Settings(init_timeout=300),
                            )
                            wandb_module.log(log_payload)
                            wandb_module.finish()
                            logged = True
                            break
                        except Exception as e:
                            print(f"  [wandb] init failed (attempt {attempt+1}/3): {e}")
                            try:
                                wandb_module.finish(exit_code=1)
                            except Exception:
                                pass
                            time.sleep(10 * (attempt + 1))
                    if not logged:
                        print(f"  [wandb] giving up on {resume_key}; "
                              "metrics saved to local JSON — re-run to retry.")

    # ------------------------------------------------------------------ #
    # Aggregate and save
    # ------------------------------------------------------------------ #
    aggregate = _aggregate(per_seed)

    contract_summary = compact_contract_summary(experiment_contract)
    trained_env_variant = None if contract_summary is None else contract_summary.get("env_variant")
    result = {
        "model_type": args.mode,
        "task": args.task,
        "backbone": args.backbone,
        "env_variant": args.env_variant,
        "eval_env_variant": args.env_variant,
        "trained_env_variant": trained_env_variant,
        "env_class": type(player.env).__name__ if hasattr(player, "env") else None,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "num_seeds": len(seeds),
        "repeats": args.repeats,
        "modes": modes_to_run,
        "num_runs": len(per_seed),
        "max_steps": args.max_steps,
        "use_landing_pad": args.use_landing_pad,
        "replan_freq": run_kwargs_base.get("replan_freq", 10),
        "frame_offsets": evaluated_frame_offsets,
        "checkpoint_path": evaluated_checkpoint_path,
        "adapter_stats_path": evaluated_stats_path,
        "base_model_path": base_model_path,
        "contract_path": contract_path,
        "contract_id": None if experiment_contract is None else experiment_contract.get("contract", {}).get("contract_id"),
        "source_commit": None if experiment_contract is None else experiment_contract.get("contract", {}).get("git", {}).get("commit"),
        "eval_git": git_summary(REPO_ROOT),
        "contract_match": contract_match,
        "contract_mismatches": contract_mismatches,
        "contract_summary": contract_summary,
        "eval_purpose": eval_purpose,
        "experiment_profile": args.experiment_profile,
        "headline_eligible": (
            args.experiment_profile == "headline_hard" and not contract_mismatches
        ),
        "success_by_mode": success_by_mode(per_seed),
        "failure_breakdown_by_mode": failure_breakdown_by_mode(per_seed, args.task),
        "progress_path": progress_path,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }

    out_path = os.path.join(out_dir, f"eval_{run_timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS:")
    print(f"{'='*60}")
    for key, val in aggregate.items():
        if "fraction" in val:
            print(f"  {key}: {val['fraction']:.2%} ({val['count']}/{val['total']})")
        else:
            print(f"  {key}: {val['mean']:.4f} +/- {val['std']:.4f}")
    print()


if __name__ == "__main__":
    main()

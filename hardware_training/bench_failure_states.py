#!/usr/bin/env python3
"""Offline benchmark on the Sep 5 live-failure states.

Samples N action chunks per saved camera frame (from the live dumps of the
failing runs) and scores exactly the behaviors that failed on hardware.
Run it on every training candidate before it touches the arms.

    # coordination head (either arm):
    python hardware_training/bench_failure_states.py \
        --coord-checkpoint .../mixed_coord_head_placewipe_hardware_final.pt \
        --coord-stats      .../mixed_coord_head_placewipe_hardware_stats.pkl

    # bare base (stage-2 gate):
    python hardware_training/bench_failure_states.py --base-only \
        --base-checkpoint .../singlearm_mixedfront_e2e_shoulder_step100000.pt \
        --base-stats      .../singlearm_mixedfront_e2e_shoulder_stats.pkl

Scenarios (frames = live dumps on the bird PC):
  S1 grasp-commit    bird 12:46 ticks 5-16, flutter at the pickup (lid overhead).
                     straddle = chunk min<300 AND max>600 (the model closing and
                     reopening within one plan). LOWER is better.
  S2 hover-release   bird 15:35 ticks 30-60, parked over the open basket with a
                     firm grip; live: 0/91 chunks sampled the release.
                     release = any row >=500; envelope = chunk min-y < -418.
                     HIGHER is better.
  S3 cardboard-release  cardboard 15:41 ticks 35-40, at the lid placement with
                     the bird arm mid-frame. release = any row >=500. HIGHER.
  S4 re-close        cardboard 15:06 ticks 25-30, just after the lid release;
                     live: the model re-closed the empty gripper and parked.
                     re-close = any of the first 5 rows <=300. LOWER is better.
  S5 return-home     bird 14:26 ticks 22-23, just after the drop. return =
                     chunk moves toward home (last-row y minus first-row y >
                     +30 mm). HIGHER is better.
"""

import argparse
import glob
import json
import os
import pickle as pkl
import sys

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.image_diffusion import ImageConditional_ODE  # noqa: E402
from src.image_coordination import ImageCoordinationHead  # noqa: E402
from hardware_training.current_frame import require_current_frame_hardware

DUMPS = os.environ.get("CODIFF_LIVE_DUMPS", "release-artifacts/hardware/live_dumps")

SCENARIOS = [
    # (id, dump dir, ticks, metrics)
    ("S1 grasp-commit",     f"{DUMPS}/bird_fbbase_0905_1246",   list(range(5, 17)),  ("straddle",)),
    ("S2 hover-release",    f"{DUMPS}/bird_fbbase_0905_1535",   list(range(30, 61, 3)), ("release", "envelope")),
    ("S3 cardboard-release", f"{DUMPS}/cardboard_servo_0905_1541", list(range(35, 41)), ("release",)),
    ("S4 re-close",         f"{DUMPS}/cardboard_servo_0905_1506", list(range(25, 31)), ("reclose",)),
    ("S5 return-home",      f"{DUMPS}/bird_fbbase_0905_1426",   [22, 23],            ("return",)),
]


def preprocess(rgb, device):
    """Mirror the inference nodes: prescale to 192x256 INTER_AREA, then 128 antialias."""
    rgb = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [128, 128], antialias=True)
    return t.unsqueeze(0).to(device)


def load_stack(args, device):
    with open(args.coord_stats if not args.base_only else args.base_stats, "rb") as f:
        stats = pkl.load(f)
    require_current_frame_hardware(stats)
    mean = np.asarray(stats["action_mean"], dtype=np.float32)
    std = np.asarray(stats["action_std"], dtype=np.float32)
    horizon = int(stats["horizon"])

    if args.base_only:
        base_stats = stats
        base_ckpt = args.base_checkpoint
        d_model = int(stats["d_model"])
    else:
        base_ckpt = args.base_checkpoint or stats.get("base_model_path", "")
        missing = {"base_n_heads", "base_depth", "base_dim_feedforward"} - set(stats)
        if missing:
            with open(args.base_stats or stats["base_stats_path"], "rb") as f:
                base_stats = pkl.load(f)
        else:
            base_stats = stats
        d_model = int(stats["base_d_model"])

    base = ImageConditional_ODE(
        x_dim=7,
        sigma_data=float(stats["sigma_data"]),
        d_model=d_model,
        n_heads=int(base_stats.get("base_n_heads", base_stats.get("n_heads"))),
        depth=int(base_stats.get("base_depth", base_stats.get("depth"))),
        dim_feedforward=int(base_stats.get("base_dim_feedforward", base_stats.get("dim_feedforward"))),
        horizon=horizon,
        device=device,
        N=args.n_steps,
        lr=1e-4,
        cfg_drop_prob=float(stats.get("base_cfg_drop_prob", base_stats.get("cfg_drop_prob", 0.2))),
        num_cameras=int(stats.get("num_cameras", 1)),
        backbone=str(stats.get("backbone", base_stats.get("backbone", "resnet18"))),
    )
    if not base.load(os.path.expanduser(base_ckpt)):
        raise RuntimeError(f"failed to load base {base_ckpt}")
    base.F.eval(); base.F_ema.eval()

    head = None
    if not args.base_only:
        head = ImageCoordinationHead(
            x_dim=7,
            base_d_model=int(stats["base_d_model"]),
            d_model=int(stats["head_d_model"]),
            n_heads=int(stats["head_n_heads"]),
            depth=int(stats["head_depth"]),
            dim_feedforward=int(stats["head_dim_feedforward"]),
            horizon=horizon,
            sigma_data=float(stats["sigma_data"]),
            lr=1e-4,
            num_cameras=int(stats.get("num_cameras", 1)),
            tokens_per_camera=int(stats["tokens_per_camera"]),
            d_base_drop_prob=float(stats.get("d_base_drop_prob", 0.1)),
            use_side_net=bool(stats.get("use_side_net", False)),
            side_net_input_size=int(stats.get("side_net_input_size", 128)),
            side_net_fusion=str(stats.get("side_net_fusion", "add")),
            side_net_tokens_per_camera=int(stats.get(
                "side_net_tokens_per_camera", stats["tokens_per_camera"])),
            decoder_execution=str(stats.get("decoder_execution", "legacy_zip")),
            decoder_conditioning=str(stats.get("decoder_conditioning", "pooled")),
        ).to(device)
        if not head.load(args.coord_checkpoint):
            raise RuntimeError(f"failed to load head {args.coord_checkpoint}")
        head.F.eval(); head.F_ema.eval()
    return base, head, mean, std, horizon


@torch.no_grad()
def sample_chunk(base, head, img, anchor_norm, w, device, horizon, anchor_dims=6, head_img=None):
    """One chunk, mirroring the deployed nodes' anchored CFG Euler loop."""
    if anchor_dims not in (6, 7):
        raise ValueError('Expected six pose coordinates or seven state coordinates')
    anchor_t = torch.from_numpy(anchor_norm.astype(np.float32)).to(device)
    enc_cond = base.F_ema.forward_encoder(None, img)
    null_token = base.F_ema.null_token.expand(1, -1, -1)
    enc_uncond = [null_token for _ in enc_cond]
    cfg_false = torch.zeros(1, dtype=torch.bool, device=device)
    cfg_true = torch.ones(1, dtype=torch.bool, device=device)

    x = torch.randn((1, horizon, base.x_dim), device=device) * base.sigma_s[0] * base.scale_s[0]
    for i in range(base.N):
        x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]
        sigma_i = torch.ones((1, 1, 1), device=device) * base.sigma_s[i]
        x_in = x / base.scale_s[i]
        D_cond = base._D_from_enc(x_in, sigma_i, enc_cond)
        D_uncond = base._D_from_enc(x_in, sigma_i, enc_uncond)
        if head is not None:
            D_cond = D_cond + head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True, cfg_mask=cfg_false,
                d_base=D_cond, imgs_eih=None, imgs_shoulder=img if head_img is None else head_img)
            D_uncond = D_uncond + head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True, cfg_mask=cfg_true,
                d_base=D_uncond, imgs_eih=None, imgs_shoulder=img if head_img is None else head_img)
        D = w * D_cond + (1 - w) * D_uncond
        delta_step = base.coeff1[i] * x - base.coeff2[i] * D
        dt = base.t_s[i] - base.t_s[i + 1] if i != base.N - 1 else base.t_s[i]
        x = x - delta_step * dt
    x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]
    return x.squeeze(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coord-checkpoint")
    p.add_argument("--coord-stats")
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--base-stats", default=None)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--n-samples", type=int, default=8, help="chunks sampled per frame")
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--cfg-w", type=float, default=1.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write results JSON here")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump-root", default=DUMPS)
    p.add_argument("--only", default=None,
                   help="run only scenarios whose id starts with this (e.g. 'S1')")
    args = p.parse_args()
    if not args.base_only and not (args.coord_checkpoint and args.coord_stats):
        p.error("--coord-checkpoint/--coord-stats required (or --base-only with base args)")
    if args.base_only and not (args.base_checkpoint and args.base_stats):
        p.error("--base-only requires --base-checkpoint and --base-stats")

    device = torch.device(args.device)
    base, head, mean, std, horizon = load_stack(args, device)
    torch.manual_seed(args.seed)
    label = args.base_checkpoint if args.base_only else args.coord_checkpoint
    print(f"model: {label}\nhorizon={horizon} n_steps={args.n_steps} cfg_w={args.cfg_w} "
          f"n_samples={args.n_samples}\n")

    results = {}
    for sid, dump, ticks, metrics in SCENARIOS:
        dump = os.path.join(args.dump_root, os.path.basename(dump))
        if args.only and not sid.startswith(args.only):
            continue
        fs = sorted(glob.glob(f"{dump}/tick_*.npz"))
        counts = {m: 0 for m in metrics}
        total = 0
        for t in ticks:
            if t >= len(fs):
                continue
            d = np.load(fs[t])
            img = preprocess(d["rgb"], device)
            anchor_raw = np.zeros(7, dtype=np.float32)
            anchor_raw[:6] = d["pose"][:6]
            anchor_norm = (anchor_raw - mean) / std
            for _ in range(args.n_samples):
                cn = sample_chunk(base, head, img, anchor_norm, args.cfg_w, device, horizon)
                chunk = cn * std + mean
                g = chunk[1:, 6]          # published rows (row 0 is the anchor echo)
                y = chunk[1:, 1]
                total += 1
                if "straddle" in counts:
                    # order-aware: only a close that later REOPENS is flutter;
                    # open->close within one chunk is a correct grasp ramp
                    # (and at horizon 50 it is the expected shape).
                    ci = np.where(g < 300)[0]
                    if len(ci) and g[ci[0]:].max() > 600:
                        counts["straddle"] += 1
                if "release" in counts and g.max() >= 500:
                    counts["release"] += 1
                if "envelope" in counts and y.min() < -418:
                    counts["envelope"] += 1
                if "reclose" in counts and g[:5].min() <= 300:
                    counts["reclose"] += 1
                if "return" in counts and (y[-1] - y[0]) > 30:
                    counts["return"] += 1
        rates = {m: (counts[m] / total if total else float("nan")) for m in metrics}
        results[sid] = {"total": total, **{m: round(r, 3) for m, r in rates.items()}}
        pretty = "  ".join(f"{m}={rates[m]*100:5.1f}%" for m in metrics)
        print(f"{sid:22s} ({total:3d} samples)  {pretty}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": label, "cfg_w": args.cfg_w, "n_steps": args.n_steps,
                       "results": results, "seed": args.seed,
                       "history_control": "repeated_current" if history_frames > 1 else "current_only"}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

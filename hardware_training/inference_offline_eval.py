#!/usr/bin/env python3
"""Open-loop evaluation: feed a recorded rollout's images through the policy,
compare the predicted action chunk to the recorded actions.

Two modes:

  default (unanchored)
    At every frame t, run model.sample on the image. Compare pred[0] vs state[t]
    (the immediate-next pose).

  --anchor (anchored / diffusion-inpainting)
    At every frame t, run model.sample but pin x[:, 0, :6] = normalized state[t]
    at every denoising step. Compare pred[1] vs state[t+1].
    Tests "if the policy is told the true current pose, does it predict the
    immediate next pose better?"

  --compare
    Run BOTH on the same rollout and overlay them in the saved plot. Lets you
    see anchored vs unanchored differences at a glance.

Run from the configured hardware Python environment.
"""

import argparse
import os
import pickle as pkl
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

CO_DIFF_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ".."
))
if CO_DIFF_ROOT not in sys.path:
    sys.path.insert(0, CO_DIFF_ROOT)

from src.image_diffusion import ImageConditional_ODE  # noqa: E402


DIM_NAMES = ["x_mm", "y_mm", "z_mm", "roll", "pitch", "yaw", "gripper"]


def resolve_rollout(p: str) -> str:
    p = os.path.expanduser(p)
    return os.path.join(p, "rollout.npz") if os.path.isdir(p) else p


def preprocess(rgb: np.ndarray, device, prescale_hw=None) -> torch.Tensor:
    # With prescale_hw=(H, W), mirror the two-stage training pipeline:
    #   cv2.resize(orig -> HxW, INTER_AREA)  [matches the conversion step]
    #   torchvision.F.resize(HxW -> 128x128, antialias=True)  [matches dataset eval]
    # Feeding a single 480x640 -> 128x128 resize to a model trained on
    # pre-resized 192x256 pkls is a real domain shift and understates the
    # policy. Without prescale_hw, keep the old single-stage path for
    # checkpoints trained on full-resolution pkls.
    if prescale_hw is not None:
        h, w = prescale_hw
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = TF.resize(t, [128, 128], antialias=True)
    else:
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def load_model(args, device):
    with open(args.stats, "rb") as f:
        stats = pkl.load(f)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)
    horizon = int(stats["horizon"])

    model = ImageConditional_ODE(
        x_dim=7,
        sigma_data=float(stats["sigma_data"]),
        d_model=int(stats["d_model"]),
        n_heads=int(stats["n_heads"]),
        depth=int(stats["depth"]),
        dim_feedforward=int(stats["dim_feedforward"]),
        horizon=horizon,
        device=device,
        N=args.n_steps,
        lr=1e-4,
        cfg_drop_prob=float(stats["cfg_drop_prob"]),
        num_cameras=int(stats.get("num_cameras", 1)),
        backbone=str(stats.get("backbone", "resnet18")),
    )
    if not model.load(args.checkpoint):
        raise RuntimeError(f"failed to load {args.checkpoint}")
    print(f"[ok] loaded {args.checkpoint}")
    print(f"     horizon={horizon}, mean={action_mean.round(2)}, std={action_std.round(2)}")
    return model, action_mean, action_std, horizon


def sample_unanchored(model, img_tensor, horizon, cfg_w):
    with torch.no_grad():
        chunk_norm = model.sample(
            imgs_eih=None,
            imgs_shoulder=img_tensor,
            traj_len=horizon,
            n_samples=1,
            w=cfg_w,
        )
    return chunk_norm.squeeze(0).cpu().numpy()  # (H, 7) normalized


def sample_anchored(model, img_tensor, anchor_norm, horizon, cfg_w, anchor_dims=6):
    """Diffusion sampling with x[:, 0, :anchor_dims] pinned to ``anchor_norm`` at every
    denoising step. Re-implements ImageConditional_ODE.sample() inline with the
    inpainting constraint.
    """
    n_samples = 1
    device = model.device
    anchor_t = torch.from_numpy(anchor_norm.astype(np.float32)).to(device)

    with torch.no_grad():
        enc_cond = model.F_ema.forward_encoder(None, img_tensor)
        null_token = model.F_ema.null_token.expand(n_samples, -1, -1)
        enc_uncond = [null_token for _ in enc_cond]

        x = torch.randn(
            (n_samples, horizon, model.x_dim), device=device
        ) * model.sigma_s[0] * model.scale_s[0]

        for i in range(model.N):
            x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]

            sigma_i = torch.ones((n_samples, 1, 1), device=device) * model.sigma_s[i]
            D_cond = model._D_from_enc(x / model.scale_s[i], sigma_i, enc_cond)
            D_uncond = model._D_from_enc(x / model.scale_s[i], sigma_i, enc_uncond)
            D = cfg_w * D_cond + (1 - cfg_w) * D_uncond

            delta = model.coeff1[i] * x - model.coeff2[i] * D
            dt = model.t_s[i] - model.t_s[i + 1] if i != model.N - 1 else model.t_s[i]
            x = x - delta * dt

        x[:, 0, :anchor_dims] = anchor_t[:anchor_dims]
    return x.squeeze(0).cpu().numpy()  # (H, 7) normalized


def run_eval(model, mean, std, H, states, images, eval_frames, overlay_frames,
             cfg_w, anchored: bool, anchor_dims: int = 6, prescale_hw=None):
    """Run inference on all eval_frames.

    Returns:
      pred_next:   (N_eval, 7) "compared-against" prediction.
                   unanchored: pred[0]; anchored: pred[1].
      compare_ts:  (N_eval,) the frame index pred_next is compared against.
                   unanchored: t; anchored: t+1.
      pred_chunks: dict {t: (H, 7)} the full predicted chunk at overlay frames.
    """
    pred_next = np.zeros((len(eval_frames), 7), dtype=np.float32)
    compare_ts = np.zeros(len(eval_frames), dtype=np.int64)
    pred_chunks = {}

    label = "anchored" if anchored else "unanchored"
    for k, t in enumerate(eval_frames):
        img = preprocess(images[t], device=model.device, prescale_hw=prescale_hw)

        if anchored:
            anchor_raw = np.zeros(7, dtype=np.float32)
            anchor_raw[:6] = states[t, :6]
            anchor_norm = (anchor_raw - mean) / std
            chunk_norm = sample_anchored(model, img, anchor_norm, H, cfg_w, anchor_dims=anchor_dims)
            idx_in_chunk = 1
            cmp_t = min(t + 1, len(states) - 1)
        else:
            chunk_norm = sample_unanchored(model, img, H, cfg_w)
            idx_in_chunk = 0
            cmp_t = t

        chunk = chunk_norm * std + mean
        pred_next[k] = chunk[idx_in_chunk]
        compare_ts[k] = cmp_t
        if t in overlay_frames:
            pred_chunks[t] = chunk

        if k % max(1, len(eval_frames) // 20) == 0:
            err = chunk[idx_in_chunk] - states[cmp_t]
            print(f"  [{label}] t={t:4d} -> cmp={cmp_t:4d}  "
                  f"pred={chunk[idx_in_chunk][:3].round(1)}  "
                  f"rec={states[cmp_t][:3].round(1)}  "
                  f"xyz err={np.linalg.norm(err[:3]):.1f} mm")

    return pred_next, compare_ts, pred_chunks


def print_metrics(label, pred_next, compare_ts, states):
    err = pred_next - states[compare_ts]
    pos_mae = np.mean(np.abs(err[:, :3]))
    pos_rmse = np.sqrt(np.mean(err[:, :3] ** 2))
    rot_mae = np.mean(np.abs(err[:, 3:6]))
    grip_mae = np.mean(np.abs(err[:, 6]))
    print(f"[metrics — {label}] over {len(pred_next)} frames")
    print(f"   position    MAE = {pos_mae:.2f} mm   RMSE = {pos_rmse:.2f} mm")
    print(f"   rotation    MAE = {rot_mae:.4f} rad ({np.degrees(rot_mae):.2f} deg)")
    print(f"   gripper     MAE = {grip_mae:.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollout", help="rollout.npz or rollout_<id> dir")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--cfg-w", type=float, default=1.5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--overlay-frames", type=int, nargs="+", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--prescale-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="Intermediate prescale size for the two-stage resize that mirrors "
                        "training. Use when the checkpoint was trained on data pre-resized "
                        "at conversion time (e.g. --prescale-hw 192 256).")
    p.add_argument("--anchor", action="store_true",
                   help="enable diffusion-inpainting anchored sampling.")
    p.add_argument("--anchor-dims", type=int, default=6,
                   help="how many leading dims of pred[0] to pin to the recorded pose. "
                        "6 = TCP only, 7 = TCP + gripper.")
    p.add_argument("--compare", action="store_true",
                   help="run both anchored and unanchored and overlay them in the plot.")
    p.add_argument("--plot-out", default=None)
    p.add_argument("--start-frame", type=int, default=0,
                   help="first frame to evaluate (inclusive).")
    p.add_argument("--end-frame", type=int, default=None,
                   help="last frame to evaluate (exclusive). Default: end of rollout.")
    p.add_argument("--unwrap-rpy", action="store_true",
                   help="np.unwrap the recorded rpy before comparing, matching "
                        "prune_dwells.py. Required for checkpoints trained on unwrapped "
                        "data: those predict in unwrapped space, while a raw npz is "
                        "wrapped, so roll near +/-pi would otherwise score a ~2pi error.")
    p.add_argument("--forward-half", action="store_true",
                   help="evaluate only frames [0, argmax(state[:,1])+1], i.e. the forward "
                        "half as split by convert_npz_to_pkl_split_by_y.py. Use for "
                        "cardboard, whose training pkls contain only that half -- scoring "
                        "a forward-only policy over the backward half is meaningless.")
    args = p.parse_args()
    device = torch.device(args.device)

    rollout_path = resolve_rollout(args.rollout)
    print(f"[load] {rollout_path}")
    z = np.load(rollout_path)
    if "state" not in z.files or "camera_top" not in z.files:
        sys.exit(f"rollout missing 'state' or 'camera_top'; have {z.files}")
    states = np.asarray(z["state"], dtype=np.float32)
    images = np.asarray(z["camera_top"], dtype=np.uint8)
    T = min(len(states), len(images))
    states, images = states[:T], images[:T]
    print(f"       T={T}, state shape={states.shape}, image shape={images.shape}")

    # Restrict to the trained-on window. The cardboard pkls hold only the
    # forward half, so the policy never saw the return stroke.
    if args.forward_half:
        peak = int(np.argmax(states[:, 1]))
        lo, hi = 0, peak + 1
        print(f"       --forward-half: y peaks at {peak} (y={states[peak,1]:.1f}mm) "
              f"-> evaluating frames [0, {hi})")
    else:
        lo = max(0, args.start_frame)
        hi = min(T, args.end_frame if args.end_frame is not None else T)
    if hi - lo < 2:
        sys.exit(f"frame window [{lo}, {hi}) too small")
    if (lo, hi) != (0, T):
        states, images = states[lo:hi], images[lo:hi]
        T = hi - lo
        print(f"       windowed to T={T}")

    if args.unwrap_rpy:
        before = states[:, 3:6].copy()
        states = states.copy()
        states[:, 3:6] = np.unwrap(states[:, 3:6], axis=0)
        moved = int(np.sum(np.abs(states[:, 3:6] - before) > 1e-6))
        print(f"       --unwrap-rpy: adjusted {moved} rpy component(s); "
              f"roll now [{states[:,3].min():+.2f}, {states[:,3].max():+.2f}]")

    model, mean, std, H = load_model(args, device)

    eval_frames = list(range(0, T, args.stride))
    overlay_frames = args.overlay_frames
    if overlay_frames is None:
        overlay_frames = [int(x) for x in np.linspace(0, T - 1, 4)]
    overlay_frames = sorted(set(overlay_frames))
    print(f"[infer] {len(eval_frames)} frames (stride={args.stride}), "
          f"overlay frames={overlay_frames}")

    # decide which runs to do
    runs = []  # list of (label, anchored)
    if args.compare:
        runs = [("unanchored", False), ("anchored", True)]
    elif args.anchor:
        runs = [("anchored", True)]
    else:
        runs = [("unanchored", False)]

    results = {}
    for label, anchored in runs:
        print(f"\n[run] {label}")
        pred_next, compare_ts, pred_chunks = run_eval(
            model, mean, std, H, states, images,
            eval_frames, overlay_frames, args.cfg_w,
            anchored=anchored, anchor_dims=args.anchor_dims,
            prescale_hw=tuple(args.prescale_hw) if args.prescale_hw else None,
        )
        results[label] = (pred_next, compare_ts, pred_chunks)
        print()
        print_metrics(label, pred_next, compare_ts, states)

    # ---- plot
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    t_axis = np.arange(T)
    style = {
        "unanchored": dict(color="red",   marker=".", lookahead="blue"),
        "anchored":   dict(color="green", marker=".", lookahead="purple"),
    }

    for d in range(7):
        ax = axes[d]
        ax.plot(t_axis, states[:, d], "k-", lw=1.2, label="recorded")

        for label, (pred_next, compare_ts, pred_chunks) in results.items():
            s = style[label]
            dot_label = f"{label} pred[{1 if label == 'anchored' else 0}]"
            ax.plot(compare_ts, pred_next[:, d], color=s["color"], marker=s["marker"],
                    ls="", ms=3, label=dot_label if d == 0 else None)

            first_overlay = sorted(pred_chunks.keys())[0] if pred_chunks else None
            for t, chunk in pred_chunks.items():
                xs = np.arange(t, t + H)
                ax.plot(xs, chunk[:, d], color=s["lookahead"], lw=0.8, alpha=0.6,
                        label=f"{label} lookahead"
                              if (d == 0 and t == first_overlay) else None)

        ax.set_ylabel(DIM_NAMES[d])
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("frame")

    if args.compare:
        suffix = "compare"
    elif args.anchor:
        suffix = "anchored"
    else:
        suffix = "open_loop"
    title = f"Eval ({suffix}): {os.path.basename(os.path.dirname(rollout_path))}"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    out = args.plot_out or os.path.join(
        os.path.dirname(rollout_path), f"open_loop_eval_{suffix}.png"
    )
    fig.savefig(out, dpi=120)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

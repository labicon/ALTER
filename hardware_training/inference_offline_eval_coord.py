#!/usr/bin/env python3
"""Open-loop evaluation of the coordination head on two-arm rollout PKLs.

Reads the packed two-arm pkls (NOT the raw npz -- for the Aug 22 session the
pkls are the training data, and some Desktop npz files were later overwritten),
splits each into its two per-arm trajectories exactly as
TwoArmE2EImageDataset does, and at every --stride-th frame runs anchored
inference:

    base+head  -- the coordination sampler (mirrors _sample_coordinated /
                  the ROS node: head residual added to both CFG branches)
    base-only  -- same anchored loop with the residual zeroed (--compare-base)

Predictions are compared against the recorded action at the next frame
(pred[1] vs recorded[t+1], since anchoring pins pred[0] to the pose at t).

Because coordination lives in the WAIT phases, metrics are reported separately
for wait frames (recorded pose still AND gripper closed over the next chunk)
and moving frames. The head should cut the wait-phase error, where the base
wants to proceed, and leave moving-phase error unchanged.

Run from the configured hardware Python environment.
"""

import argparse
import glob
import os
import pickle as pkl
import re
import sys

import numpy as np
import torch
from torchvision.transforms import functional as TVF

CO_DIFF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CO_DIFF_ROOT not in sys.path:
    sys.path.insert(0, CO_DIFF_ROOT)

from src.image_diffusion import ImageConditional_ODE  # noqa: E402
from src.image_coordination import ImageCoordinationHead  # noqa: E402

GRIP_OPEN_THR = 425.0


def load_models(coord_ckpt, coord_stats_path, base_ckpt_arg, base_stats_arg, n_steps, device):
    with open(coord_stats_path, "rb") as f:
        stats = pkl.load(f)
    mean = np.asarray(stats["action_mean"], dtype=np.float32)
    std = np.asarray(stats["action_std"], dtype=np.float32)
    horizon = int(stats["horizon"])
    num_cameras = int(stats.get("num_cameras", 1))

    base_ckpt = base_ckpt_arg or stats.get("base_model_path", "")
    base_stats_path = base_stats_arg or stats.get("base_stats_path", "")
    with open(os.path.expanduser(base_stats_path), "rb") as f:
        base_stats = pkl.load(f)

    base = ImageConditional_ODE(
        x_dim=7,
        sigma_data=float(stats["sigma_data"]),
        d_model=int(stats["base_d_model"]),
        n_heads=int(base_stats["n_heads"]),
        depth=int(base_stats["depth"]),
        dim_feedforward=int(base_stats["dim_feedforward"]),
        horizon=horizon,
        device=device,
        N=n_steps,
        lr=1e-4,
        cfg_drop_prob=float(base_stats.get("cfg_drop_prob", 0.2)),
        num_cameras=num_cameras,
        backbone=str(stats.get("backbone", "resnet18")),
    )
    if not base.load(os.path.expanduser(base_ckpt)):
        sys.exit(f"failed to load base {base_ckpt}")
    base.F.eval()
    base.F_ema.eval()

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
        num_cameras=num_cameras,
        tokens_per_camera=int(stats["tokens_per_camera"]),
        d_base_drop_prob=float(stats.get("d_base_drop_prob", 0.1)),
        use_side_net=bool(stats.get("use_side_net", False)),
        side_net_input_size=int(stats.get("side_net_input_size", 128)),
        side_net_fusion=str(stats.get("side_net_fusion", "add")),
        side_net_tokens_per_camera=int(stats.get("side_net_tokens_per_camera",
                                                 stats["tokens_per_camera"])),
        decoder_execution=str(stats.get("decoder_execution", "legacy_zip")),
        decoder_conditioning=str(stats.get("decoder_conditioning", "pooled")),
    ).to(device)
    if not head.load(coord_ckpt):
        sys.exit(f"failed to load head {coord_ckpt}")
    head.F.eval()
    head.F_ema.eval()
    return base, head, mean, std, horizon


@torch.no_grad()
def sample(base, head, img, anchor_norm, horizon, cfg_w, use_head=True, seed=0):
    """Anchored coordination sampling; mirrors the ROS node / canonical sampler."""
    device = base.device
    torch.manual_seed(seed)
    enc = base.F_ema.forward_encoder(None, img)
    null = base.F_ema.null_token.expand(1, -1, -1)
    enc_u = [null for _ in enc]
    m0 = torch.zeros(1, dtype=torch.bool, device=device)
    m1 = torch.ones(1, dtype=torch.bool, device=device)
    a = torch.from_numpy(anchor_norm.astype(np.float32)).to(device)

    x = torch.randn((1, horizon, 7), device=device) * base.sigma_s[0] * base.scale_s[0]
    for i in range(base.N):
        x[:, 0, :6] = a[:6]
        sg = torch.ones((1, 1, 1), device=device) * base.sigma_s[i]
        Dc = base._D_from_enc(x / base.scale_s[i], sg, enc)
        Du = base._D_from_enc(x / base.scale_s[i], sg, enc_u)
        if use_head:
            dc = head.forward_residual(x, sg, enc, use_ema=True, cfg_mask=m0,
                                       d_base=Dc, imgs_eih=None, imgs_shoulder=img)
            du = head.forward_residual(x, sg, enc, use_ema=True, cfg_mask=m1,
                                       d_base=Du, imgs_eih=None, imgs_shoulder=img)
            Dc, Du = Dc + dc, Du + du
        D = cfg_w * Dc + (1 - cfg_w) * Du
        step = base.coeff1[i] * x - base.coeff2[i] * D
        dt = base.t_s[i] - base.t_s[i + 1] if i != base.N - 1 else base.t_s[i]
        x = x - step * dt
    x[:, 0, :6] = a[:6]
    return x.squeeze(0).cpu().numpy()


def preprocess(frame, device):
    t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
    t = TVF.resize(t, [128, 128], antialias=True)
    return t.unsqueeze(0).to(device)


def wait_mask(actions, horizon, pos_tol=2.0):
    """A frame is a WAIT if the recorded pose barely moves over the next chunk
    AND the gripper is closed (holding something)."""
    T = len(actions)
    out = np.zeros(T, dtype=bool)
    for t in range(T):
        j = min(T, t + horizon)
        path = np.sum(np.linalg.norm(np.diff(actions[t:j, :3], axis=0), axis=1))
        out[t] = path < pos_tol * (j - t) / 4 and actions[t, 6] < GRIP_OPEN_THR
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--twoarm-dir", required=True, help="packed two-arm pkl dir")
    p.add_argument("--coord-checkpoint", required=True)
    p.add_argument("--coord-stats", required=True)
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--base-stats", default=None)
    p.add_argument("--rollouts", type=int, nargs="+", default=None,
                   help="rollout ids to evaluate (default: first --max-rollouts)")
    p.add_argument("--max-rollouts", type=int, default=6)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--cfg-w", type=float, default=1.2)
    p.add_argument("--compare-base", action="store_true",
                   help="also run base-only (head residual zeroed) on every frame")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    base, head, mean, std, H = load_models(
        args.coord_checkpoint, args.coord_stats,
        args.base_checkpoint, args.base_stats, args.n_steps, device)

    files = sorted(glob.glob(os.path.join(args.twoarm_dir, "*.pkl")))
    if args.rollouts:
        want = set(args.rollouts)
        files = [f for f in files
                 if int(re.search(r"rollout_(\d+)", os.path.basename(f)).group(1)) in want]
    else:
        files = files[: args.max_rollouts]
    if not files:
        sys.exit("no rollouts selected")
    print(f"evaluating {len(files)} two-arm rollouts x 2 arms, stride={args.stride}, "
          f"cfg_w={args.cfg_w}, N={args.n_steps}, compare_base={args.compare_base}")

    agg = {}  # (model, phase) -> list of per-frame position errors
    gagg = {}  # (model, phase) -> gripper errors

    for path in files:
        with open(path, "rb") as f:
            r = pkl.load(f)
        acts = np.asarray(r["actions"], dtype=np.float32)
        name = os.path.basename(path).replace(".pkl", "")
        for arm, cam_key in ((0, "camera_obs_shoulder0"), (1, "camera_obs_shoulder1")):
            cams = np.asarray(r[cam_key], dtype=np.uint8)
            act = acts[:, arm * 7:(arm + 1) * 7]
            T = min(len(act), len(cams))
            act = act[:T]
            waits = wait_mask(act, H)
            frames = list(range(0, T - 1, args.stride))
            rows = {"coord": [], "base": []}
            for t in frames:
                img = preprocess(cams[t], device)
                anchor = (act[t] - mean) / std
                runs = [("coord", True)] + ([("base", False)] if args.compare_base else [])
                for label, use_head in runs:
                    ch = sample(base, head, img, anchor, H, args.cfg_w,
                                use_head=use_head, seed=0)
                    pred = ch[1] * std + mean
                    perr = float(np.linalg.norm(pred[:3] - act[t + 1, :3]))
                    gerr = float(abs(pred[6] - act[t + 1, 6]))
                    phase = "wait" if waits[t] else "move"
                    agg.setdefault((label, phase), []).append(perr)
                    gagg.setdefault((label, phase), []).append(gerr)
                    rows[label].append((phase, perr))
            for label in ("coord",) + (("base",) if args.compare_base else ()):
                w = [e for ph, e in rows[label] if ph == "wait"]
                m = [e for ph, e in rows[label] if ph == "move"]
                print(f"  {name} arm{arm} [{label:5s}]  "
                      f"wait: n={len(w):3d} posMAE={np.mean(w) if w else float('nan'):6.1f}mm   "
                      f"move: n={len(m):3d} posMAE={np.mean(m) if m else float('nan'):6.1f}mm",
                      flush=True)

    print("\n================ AGGREGATE (pred[1] vs recorded next pose) ================")
    print(f"{'model':>6s} {'phase':>6s} {'n':>6s} {'pos MAE':>9s} {'pos RMSE':>9s} {'grip MAE':>9s}")
    for (label, phase), v in sorted(agg.items()):
        v = np.asarray(v)
        g = np.asarray(gagg[(label, phase)])
        print(f"{label:>6s} {phase:>6s} {len(v):6d} {v.mean():8.1f}m {np.sqrt((v**2).mean()):8.1f}m {g.mean():9.1f}")


if __name__ == "__main__":
    main()

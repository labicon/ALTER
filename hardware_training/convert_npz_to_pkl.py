#!/usr/bin/env python3
"""Convert hardware rollouts (rollout_<id>/rollout.npz) to Co-Diff's per-rollout .pkl format.

Output dict keys (matching what `SingleArmShoulderE2EDataset` reads):
    actions_single_arm   (T, 7) float32   <- copy of state (TCP xyz mm + rotvec rad + gripper)
    camera_obs_shoulder  (T, H, W, 3) u8  <- copy of camera_top (optionally resized)
    state                (T, 7) float32   <- diagnostic, not used by the trainer
    joint_angles         (T, 7) float32   <- diagnostic, not used by the trainer (may be absent)

Pass --target-h/--target-w to resize the camera frames during conversion. This
trades a one-time cost at convert time for a much smaller on-disk and in-RAM
dataset (e.g. 480x640 -> 192x256 is ~6.25x smaller). The dataset class applies
its own 128x128 resize/crop at training time, so any input HxW works.
"""

import argparse
import glob
import os
import pickle
import re
import sys

import cv2
import numpy as np


ROLLOUT_RE = re.compile(r"rollout_(\d+)$")


def find_rollouts(input_dir: str):
    out = []
    for d in sorted(glob.glob(os.path.join(input_dir, "rollout_*"))):
        m = ROLLOUT_RE.search(os.path.basename(d))
        if not m:
            continue
        npz = os.path.join(d, "rollout.npz")
        if os.path.isfile(npz):
            out.append((int(m.group(1)), npz))
    out.sort(key=lambda x: x[0])
    return out


def unwrap_rotations(state: np.ndarray) -> np.ndarray:
    """Make the rotation-vector columns continuous in time.

    The recorder reports rpy in [-pi, pi], so a pose held steady at the seam is
    logged as +3.141 on one frame and -3.141 on the next. Those are the SAME
    orientation, but 2*pi apart numerically, which makes the training target
    bimodal at a visually identical scene. A diffusion policy then samples
    between the two modes and emits an orientation that is neither -- observed
    live as a 2.8 rad (160 deg) wrist command that tripped --max-rot-step-rad.

    np.unwrap removes the discontinuity by adding multiples of 2*pi, so roll
    becomes a constant ~3.14 instead of flipping. Execution is unaffected: the
    executor wraps rotation deltas before bounding them, and the arm treats
    3.14 and -3.14 as the same pose.

    No-op for trajectories that never cross the seam (e.g. cardboard-forward,
    measured 0 sign flips).
    """
    out = state.copy()
    out[:, 3:6] = np.unwrap(out[:, 3:6], axis=0)
    return out


def resize_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a (T, H, W, 3) uint8 array to (T, target_h, target_w, 3) uint8.

    Uses INTER_AREA which is the right kernel for downsampling -- it's
    equivalent to box filtering and avoids aliasing.
    """
    T = frames.shape[0]
    out = np.empty((T, target_h, target_w, 3), dtype=np.uint8)
    for i in range(T):
        out[i] = cv2.resize(frames[i], (target_w, target_h),
                            interpolation=cv2.INTER_AREA)
    return out


def convert_one(npz_path: str, target_h=None, target_w=None) -> dict:
    z = np.load(npz_path)
    if "state" not in z.files:
        raise KeyError(f"{npz_path}: missing 'state'")
    if "camera_top" not in z.files:
        raise KeyError(f"{npz_path}: missing 'camera_top'")

    state = unwrap_rotations(np.asarray(z["state"], dtype=np.float32))
    camera = np.asarray(z["camera_top"], dtype=np.uint8)

    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(f"{npz_path}: state has shape {state.shape}, expected (T, 7)")
    if camera.ndim != 4 or camera.shape[-1] != 3:
        raise ValueError(f"{npz_path}: camera_top has shape {camera.shape}, expected (T, H, W, 3)")

    T = min(state.shape[0], camera.shape[0])
    camera = camera[:T]

    if target_h is not None and target_w is not None:
        camera = resize_frames(camera, target_h, target_w)

    rollout = {
        "actions_single_arm": state[:T].copy(),
        "camera_obs_shoulder": camera,
        "state": state[:T].copy(),
    }
    if "joint_angles" in z.files:
        ja = np.asarray(z["joint_angles"], dtype=np.float32)
        if ja.shape[0] >= T:
            rollout["joint_angles"] = ja[:T].copy()
    return rollout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help=f"Folder containing rollout_<id>/rollout.npz")
    ap.add_argument("--output-dir", required=True,
                    help=f"Where to write rollout_<id>.pkl")
    ap.add_argument("--target-h", type=int, default=None,
                    help="If set (with --target-w), resize camera frames to this height.")
    ap.add_argument("--target-w", type=int, default=None,
                    help="If set (with --target-h), resize camera frames to this width.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip rollouts whose output .pkl already exists (resume mode).")
    args = ap.parse_args()

    if (args.target_h is None) != (args.target_w is None):
        sys.exit("Pass both --target-h and --target-w to resize, or neither.")

    rollouts = find_rollouts(args.input_dir)
    if not rollouts:
        sys.exit(f"No rollout_*/rollout.npz under {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    msg = f"Converting {len(rollouts)} rollout(s) -> {args.output_dir}"
    if args.target_h is not None:
        msg += f"  (resize -> {args.target_h}x{args.target_w}, INTER_AREA)"
    print(msg)

    import gc
    for rid, npz in rollouts:
        out = os.path.join(args.output_dir, f"rollout_{rid}.pkl")
        if args.skip_existing and os.path.isfile(out):
            print(f"  rollout_{rid}: skip (exists)")
            continue
        try:
            r = convert_one(npz, target_h=args.target_h, target_w=args.target_w)
        except Exception as exc:
            print(f"  rollout_{rid}: SKIP ({exc})")
            continue
        with open(out, "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        T = r["actions_single_arm"].shape[0]
        H, W = r["camera_obs_shoulder"].shape[1:3]
        keys = ", ".join(sorted(r.keys()))
        print(f"  rollout_{rid}: T={T}  cam={H}x{W}  keys={{{keys}}}  -> {out}", flush=True)
        # Explicitly free per-iteration buffers so memory doesn't drift up across
        # many rollouts on a constrained system.
        del r
        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pair two single-arm recordings into the two-arm pkl format that
`envs/arm/train/twoarm_dataset.TwoArmE2EImageDataset` reads.

Inputs:
    --input-arm0-dir/rollout_<i>/rollout.npz   (one arm, with state (T,7) and camera_top)
    --input-arm1-dir/rollout_<i>/rollout.npz   (other arm, same schema)

Pairing: by rollout id. Both folders must contain rollout_<i>; mismatched
ids are warned and skipped. Trajectories of different lengths within a pair
are padded to max(T0, T1) by repeating the last frame/state.

Output dict per pair (keys match what TwoArmE2EImageDataset expects):
    actions              (T, 14) float32   = concat(arm0_state, arm1_state)
    camera_obs_shoulder0 (T, H, W, 3) u8   <- arm0's resized camera_top
    camera_obs_shoulder1 (T, H, W, 3) u8   <- arm1's resized camera_top
    camera_obs0          (T, H, W, 3) u8   <- same buffer as shoulder0 (pickle memoizes; same on disk)
    camera_obs1          (T, H, W, 3) u8   <- same buffer as shoulder1
    state                (T, 14) float32   (diagnostic)
    joint_angles         (T, 14) float32   (diagnostic, only if both npzs have it)

The dataset uses RandomResizedCrop(128) at training time, so we pre-resize to
192x256 (same setup as the existing single-arm hardware pipeline) and keep
disk + RAM footprints small.
"""

import argparse
import gc
import glob
import os
import pickle
import re
import sys

import cv2
import numpy as np


ROLLOUT_RE = re.compile(r"rollout_(\d+)$")


def find_rollouts(input_dir: str) -> dict:
    """Return {id: npz_path} for every rollout_<id>/rollout.npz under input_dir."""
    out = {}
    for d in sorted(glob.glob(os.path.join(input_dir, "rollout_*"))):
        m = ROLLOUT_RE.search(os.path.basename(d))
        if not m:
            continue
        npz = os.path.join(d, "rollout.npz")
        if os.path.isfile(npz):
            out[int(m.group(1))] = npz
    return out


def resize_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    T = frames.shape[0]
    out = np.empty((T, target_h, target_w, 3), dtype=np.uint8)
    for i in range(T):
        out[i] = cv2.resize(frames[i], (target_w, target_h),
                            interpolation=cv2.INTER_AREA)
    return out


def pad_to_length(state: np.ndarray, camera: np.ndarray,
                  joint_angles, target_T: int):
    """Repeat the last frame/state to bring all arrays up to target_T."""
    T = state.shape[0]
    if T == target_T:
        return state, camera, joint_angles
    pad_len = target_T - T
    state = np.concatenate(
        [state, np.tile(state[-1:], (pad_len, 1))], axis=0
    )
    camera = np.concatenate(
        [camera, np.tile(camera[-1:], (pad_len, 1, 1, 1))], axis=0
    )
    if joint_angles is not None:
        joint_angles = np.concatenate(
            [joint_angles, np.tile(joint_angles[-1:], (pad_len, 1))], axis=0
        )
    return state, camera, joint_angles


def load_one(npz_path: str):
    z = np.load(npz_path)
    if "state" not in z.files:
        raise KeyError(f"{npz_path}: missing 'state'")
    if "camera_top" not in z.files:
        raise KeyError(f"{npz_path}: missing 'camera_top'")
    state = np.asarray(z["state"], dtype=np.float32)
    camera = np.asarray(z["camera_top"], dtype=np.uint8)
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(f"{npz_path}: state shape {state.shape}, expected (T, 7)")
    if camera.ndim != 4 or camera.shape[-1] != 3:
        raise ValueError(f"{npz_path}: camera_top shape {camera.shape}, expected (T,H,W,3)")
    T = min(state.shape[0], camera.shape[0])
    state = state[:T]
    camera = camera[:T]
    joint_angles = None
    if "joint_angles" in z.files:
        ja = np.asarray(z["joint_angles"], dtype=np.float32)
        if ja.shape[0] >= T:
            joint_angles = ja[:T]
    return state, camera, joint_angles


def convert_pair(npz0: str, npz1: str, target_h: int, target_w: int) -> dict:
    s0, c0, j0 = load_one(npz0)
    s1, c1, j1 = load_one(npz1)

    T = max(s0.shape[0], s1.shape[0])
    s0, c0, j0 = pad_to_length(s0, c0, j0, T)
    s1, c1, j1 = pad_to_length(s1, c1, j1, T)

    c0 = resize_frames(c0, target_h, target_w)
    c1 = resize_frames(c1, target_h, target_w)

    rollout = {
        "actions": np.concatenate([s0, s1], axis=1).astype(np.float32),
        "camera_obs_shoulder0": c0,
        "camera_obs_shoulder1": c1,
        # Dataset checks `camera_obs<i>` existence before processing agent i; alias the
        # shoulder buffers here so we don't pay 2x disk (pickle memoizes shared objects).
        "camera_obs0": c0,
        "camera_obs1": c1,
        "state": np.concatenate([s0, s1], axis=1).astype(np.float32),
    }
    if j0 is not None and j1 is not None:
        rollout["joint_angles"] = np.concatenate([j0, j1], axis=1).astype(np.float32)
    return rollout, s0.shape[0], s1.shape[0], T


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-arm0-dir", required=True,
                    help=f"arm 0 raw rollouts")
    ap.add_argument("--input-arm1-dir", required=True,
                    help=f"arm 1 raw rollouts")
    ap.add_argument("--output-dir", required=True,
                    help=f"per-pair pkl output")
    ap.add_argument("--target-h", type=int, default=192)
    ap.add_argument("--target-w", type=int, default=256)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip ids whose output rollout_<id>.pkl already exists.")
    args = ap.parse_args()

    arm0_dir = os.path.expanduser(args.input_arm0_dir)
    arm1_dir = os.path.expanduser(args.input_arm1_dir)
    out_dir = os.path.expanduser(args.output_dir)

    arm0 = find_rollouts(arm0_dir)
    arm1 = find_rollouts(arm1_dir)
    if not arm0:
        sys.exit(f"No rollout_*/rollout.npz under {arm0_dir}")
    if not arm1:
        sys.exit(f"No rollout_*/rollout.npz under {arm1_dir}")

    common = sorted(set(arm0).intersection(arm1))
    only0 = sorted(set(arm0) - set(arm1))
    only1 = sorted(set(arm1) - set(arm0))
    if only0:
        print(f"WARN: ids only in arm0 (skipped): {only0}")
    if only1:
        print(f"WARN: ids only in arm1 (skipped): {only1}")
    if not common:
        sys.exit("No paired rollout ids between the two folders.")

    os.makedirs(out_dir, exist_ok=True)
    print(
        f"Pairing {len(common)} rollouts: arm0={arm0_dir} | arm1={arm1_dir}\n"
        f"  -> {out_dir}  (resize {args.target_h}x{args.target_w}, INTER_AREA)"
    )

    n_ok = n_skip = n_err = 0
    for rid in common:
        out = os.path.join(out_dir, f"rollout_{rid}.pkl")
        if args.skip_existing and os.path.isfile(out):
            print(f"  rollout_{rid}: skip (exists)")
            n_skip += 1
            continue
        try:
            r, T0, T1, T = convert_pair(
                arm0[rid], arm1[rid], args.target_h, args.target_w
            )
        except Exception as exc:
            print(f"  rollout_{rid}: SKIP ({exc})", flush=True)
            n_err += 1
            continue
        with open(out, "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        H, W = r["camera_obs_shoulder0"].shape[1:3]
        print(
            f"  rollout_{rid}: T0={T0} T1={T1} -> T={T}  cam={H}x{W}  -> {out}",
            flush=True,
        )
        n_ok += 1
        del r
        gc.collect()

    print(f"Done. ok={n_ok}  skipped={n_skip}  errors={n_err}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Remove replan-boundary micro-jitter from policy-rollout pkls.

Policy rollouts (unlike teleop demos) carry 0.8-1.6 mm direction reversals in
up to ~20 % of moving frames -- chunk-boundary correction wiggles. Dwell
pruning cannot remove them (they ride on top of motion), so this applies a
Savitzky-Golay filter to the pose channels of ``actions_single_arm`` before
pruning. Measured on the Sep-10 cardboard-bwd set (worst case): window 7,
order 3 takes reversals 19.6 % -> 0.3 % with <2 mm max path deviation.

The gripper channel is left untouched (it is clean and step-like; smoothing
would soften the ramps). Rotation channels are unwrapped before filtering so
the +/-pi roll seam cannot corrupt the fit; output stays unwrapped, matching
what prune_dwells produces for the demo pools. ``state`` is mirrored from the
smoothed actions to stay consistent.

    python hardware_training/smooth_action_jitter.py \
        --input-dir <raw pkls> --output-dir <smoothed pkls> [--window 7 --order 3]
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
from scipy.signal import savgol_filter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--order", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prov = {"operation": "smooth_action_jitter", "source_dir": args.input_dir,
            "args": vars(args), "rollouts": {}}
    for f in sorted(glob.glob(os.path.join(args.input_dir, "*.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        a = np.asarray(d["actions_single_arm"], dtype=np.float32).copy()
        T = len(a)
        if T > args.window:
            dd = np.diff(a[:, :3], axis=0)
            dots = (dd[1:] * dd[:-1]).sum(1)
            rev_before = float((dots < 0).mean())
            a[:, :3] = savgol_filter(a[:, :3], args.window, args.order, axis=0)
            rot = np.unwrap(a[:, 3:6], axis=0)
            a[:, 3:6] = savgol_filter(rot, args.window, args.order, axis=0)
            dd = np.diff(a[:, :3], axis=0)
            dots = (dd[1:] * dd[:-1]).sum(1)
            rev_after = float((dots < 0).mean())
        else:
            rev_before = rev_after = float("nan")
        d["actions_single_arm"] = a
        d["state"] = a.copy()
        out = os.path.join(args.output_dir, os.path.basename(f))
        with open(out, "wb") as fh:
            pickle.dump(d, fh)
        prov["rollouts"][os.path.basename(f)] = {
            "T": T, "rev_before": round(rev_before, 4), "rev_after": round(rev_after, 4)}
    with open(os.path.join(args.output_dir, "SMOOTH_PROVENANCE.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    n = len(prov["rollouts"])
    rb = np.nanmedian([r["rev_before"] for r in prov["rollouts"].values()])
    ra = np.nanmedian([r["rev_after"] for r in prov["rollouts"].values()])
    print(f"{n} rollouts smoothed (w={args.window}, o={args.order}): "
          f"median reversal fraction {rb:.3f} -> {ra:.3f}")


if __name__ == "__main__":
    main()

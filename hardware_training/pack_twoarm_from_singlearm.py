#!/usr/bin/env python3
"""Zip two independently processed single-arm pkl pools into the two-arm pkl
schema, by rollout id, without aligning or padding the camera streams.

Why independent: TwoArmE2EImageDataset splits every two-arm pkl back into one
trajectory per arm, using only that arm's own camera and its own 7-D action
slice, with a per-arm T = min(len(actions), len(that arm's images)). The two
arms are never paired frame-to-frame, so each arm can be converted and pruned
on its own clock (convert_npz_to_pkl.py -> prune_dwells.py) and simply zipped
here. That is also the only consistent choice: the arms were recorded as two
separate teleop streams, and per-arm dwell capping changes each arm's length
independently.

Per-arm lengths are preserved. `actions`/`state`/`joint_angles` are (T_max, 14)
with the shorter arm's rows padded by repeating its last row; those rows are
never read, because the dataset's per-arm T is bounded by that arm's camera
length. Each camera stream is stored at its own true length.

Output: rollout_<id>_mode<k>.pkl with
    actions              (T_max, 14) float32   [arm0 | arm1]
    state                (T_max, 14) float32
    joint_angles         (T_max, 14) float32   if both sides have it
    camera_obs_shoulder0 (T0, H, W, 3) u8
    camera_obs_shoulder1 (T1, H, W, 3) u8
    camera_obs0 / camera_obs1             same objects (pickle memoizes)
Writes PACK_PROVENANCE.json.
"""

import argparse
import gc
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

ROLLOUT_RE = re.compile(r"rollout_(\d+)")


def index_dir(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.pkl"))):
        m = ROLLOUT_RE.search(os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def load_arm(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    act = np.asarray(r["actions_single_arm"], dtype=np.float32)
    cam = np.asarray(r["camera_obs_shoulder"], dtype=np.uint8)
    T = min(len(act), len(cam))
    ja = None
    if "joint_angles" in r and len(r["joint_angles"]) >= T:
        ja = np.asarray(r["joint_angles"], dtype=np.float32)[:T]
    return act[:T], cam[:T], ja


def pad_rows(x, T):
    if len(x) == T:
        return x
    return np.concatenate([x, np.tile(x[-1:], (T - len(x), 1))], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm0-dir", required=True, help="processed single-arm pkls for arm 0")
    ap.add_argument("--arm1-dir", required=True, help="processed single-arm pkls for arm 1")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=20,
                    help="reject a pair if either arm is shorter than this (the horizon)")
    ap.add_argument("--allow-unpaired", action="store_true",
                    help="also pack ids present in only one pool as single-sided pkls: "
                         "the missing arm's camera keys are omitted, so the dataset "
                         "skips that agent and never reads its zero-filled action "
                         "columns. Use when per-arm filtering (filter_multiclose.py) "
                         "removed a rollout from one arm but the other arm's demo "
                         "should still train.")
    args = ap.parse_args()

    a0 = index_dir(args.arm0_dir)
    a1 = index_dir(args.arm1_dir)
    ids = sorted(set(a0) & set(a1)) if not args.allow_unpaired \
        else sorted(set(a0) | set(a1))
    if not ids:
        sys.exit("no rollout ids in common")
    only0, only1 = sorted(set(a0) - set(a1)), sorted(set(a1) - set(a0))
    print(f"ids={len(ids)}  unpaired arm0={only0}  unpaired arm1={only1}"
          f"  (unpaired {'packed single-sided' if args.allow_unpaired else 'skipped'})")
    os.makedirs(args.output_dir, exist_ok=True)

    prov = {
        "operation": "pack_twoarm_from_singlearm",
        "note": "arms zipped by rollout id; NOT time-aligned; per-arm camera lengths preserved; "
                "actions/state rows beyond an arm's own length are last-row padding the dataset never reads",
        "arm0_dir": os.path.abspath(args.arm0_dir),
        "arm1_dir": os.path.abspath(args.arm1_dir),
        "unpaired_arm0_ids": only0,
        "unpaired_arm1_ids": only1,
        "pairs": {},
    }
    n_ok = 0
    for rid in ids:
        try:
            act0, cam0, ja0 = load_arm(a0[rid]) if rid in a0 else (None, None, None)
            act1, cam1, ja1 = load_arm(a1[rid]) if rid in a1 else (None, None, None)
        except Exception as exc:
            print(f"  rollout_{rid}: SKIP ({exc})")
            prov["pairs"][str(rid)] = {"skipped": str(exc)}
            continue
        T0 = len(act0) if act0 is not None else None
        T1 = len(act1) if act1 is not None else None
        if min(t for t in (T0, T1) if t is not None) < args.min_frames:
            print(f"  rollout_{rid}: SKIP (T0={T0} T1={T1} < {args.min_frames})")
            prov["pairs"][str(rid)] = {"skipped": "too short", "T0": T0, "T1": T1}
            continue
        T = max(t for t in (T0, T1) if t is not None)
        # A missing arm gets zero-filled action rows and NO camera keys; the
        # dataset skips an agent whose camera_obs<i> is absent before ever
        # reading its action slice, so the zeros are never read.
        z = np.zeros((T, 7), dtype=np.float32)
        half0 = pad_rows(act0, T) if act0 is not None else z
        half1 = pad_rows(act1, T) if act1 is not None else z
        r = {
            "actions": np.concatenate([half0, half1], axis=1),
            "state": np.concatenate([half0, half1], axis=1),
        }
        if cam0 is not None:
            r["camera_obs_shoulder0"] = r["camera_obs0"] = cam0
        if cam1 is not None:
            r["camera_obs_shoulder1"] = r["camera_obs1"] = cam1
        if (ja0 is not None or act0 is None) and (ja1 is not None or act1 is None) \
                and not (ja0 is None and ja1 is None):
            r["joint_angles"] = np.concatenate(
                [pad_rows(ja0, T) if ja0 is not None else z,
                 pad_rows(ja1, T) if ja1 is not None else z], axis=1)
        out = os.path.join(args.output_dir, f"rollout_{rid}_mode{args.mode}.pkl")
        with open(out, "wb") as f:
            pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
        prov["pairs"][str(rid)] = {"T0": T0, "T1": T1, "T_max": T}
        if act0 is None or act1 is None:
            prov["pairs"][str(rid)]["single_sided"] = "arm1" if act0 is None else "arm0"
        print(f"  rollout_{rid}: arm0 T={T0}  arm1 T={T1}  -> {os.path.basename(out)}", flush=True)
        n_ok += 1
        del r, act0, act1, cam0, cam1
        gc.collect()

    with open(os.path.join(args.output_dir, "PACK_PROVENANCE.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"Done. ok={n_ok}/{len(ids)} -> {args.output_dir}")


if __name__ == "__main__":
    main()

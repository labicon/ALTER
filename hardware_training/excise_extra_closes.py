#!/usr/bin/env python3
"""Stage 2 of the two-arm data pipeline: cut every gripper close after the
first out of the trajectory.

A correct demo closes the gripper exactly once (the grasp). Later closes are
operator noise -- on the Sep 13 bird arm, 20-26 frame blips a few seconds
after the drop, before going home. The rollout is KEPT; only the blip's
frames are deleted, actions and camera frames together, so the surviving
trajectory reads as if the blip never happened. (The Aug 22 pipeline instead
relabelled the blip's gripper values to open with
prune_dwells --relabel-accidental-closes; excision replaces that flag.)

Each cut includes the closing/opening ramps: the gripper takes several frames
to pass through the open/closed threshold and back (800 -> 643 -> 538 -> 432
... 429 -> 541 -> 654 -> 800). Cutting only the sub-threshold frames would
leave a dip to half-closed in the labels, which the policy reproduces live as
a re-close after the drop. The cut extends outward from each closed span
while the gripper is more than --ramp-tol below the fully-open reference
(max gripper value in the 60 frames before the span).

A closed run starting at frame 0 is the recorder's unissued gripper=0 init
artifact, not a close (same rule as prune_dwells), and does not count as the
first close. A rollout with zero closes never grasped; it is flagged but
kept. Rollouts with a single close are copied through byte-identical.

Run once per arm: excision is per-arm and never touches the other arm's pool.
Camera keys that alias the same array stay aliased on output (pickle memo),
matching prune_dwells.
"""
import argparse, gc, glob, json, os, pickle, shutil
import numpy as np

from prune_dwells import GRIP_OPEN_THR, KEYS_T


def closed_spans(g):
    """(start, end_exclusive) of contiguous closed runs, excluding the
    frame-0 init artifact."""
    closed = g < GRIP_OPEN_THR
    spans = []
    i, T = 0, len(g)
    while i < T:
        if not closed[i]:
            i += 1
            continue
        j = i
        while j < T and closed[j]:
            j += 1
        if i > 0:
            spans.append((i, j))
        i = j
    return spans


def excise_mask(g, ramp_tol):
    """keep-mask over frames, plus the ramp-extended cut spans."""
    spans = closed_spans(g)
    keep = np.ones(len(g), dtype=bool)
    cuts = []
    for i, j in spans[1:]:
        open_val = float(g[max(0, i - 60):i].max())
        lo = i
        while lo > 0 and g[lo - 1] < open_val - ramp_tol:
            lo -= 1
        hi = j
        while hi < len(g) and g[hi] < open_val - ramp_tol:
            hi += 1
        keep[lo:hi] = False
        cuts.append((lo, hi))
    return keep, cuts, len(spans)


def process(src, dst, ramp_tol):
    with open(src, "rb") as f:
        r = pickle.load(f)
    g = np.asarray(r["actions_single_arm"], np.float32)[:, 6]
    keep, cuts, n_closes = excise_mask(g, ramp_tol)
    T = len(g)
    if not cuts:
        shutil.copy2(src, dst)
        return T, T, cuts, n_closes
    idx = np.where(keep)[0]
    out = {}
    sliced = {}  # id(original) -> sliced copy, so aliased keys stay aliased
    for k, v in r.items():
        if k in KEYS_T and hasattr(v, "__len__") and len(v) == T:
            key = id(v)
            if key not in sliced:
                sliced[key] = np.asarray(v)[idx].copy()
            out[k] = sliced[key]
        else:
            out[k] = v
    with open(dst, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return T, len(idx), cuts, n_closes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="converted single-arm pkls (one arm; run once per arm)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ramp-tol", type=float, default=20.0,
                   help="gripper units below fully-open still counted as part "
                        "of a close's ramp (same tolerance as the old relabel)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    src_files = sorted(glob.glob(os.path.join(a.input_dir, "*.pkl")))
    if not src_files:
        raise SystemExit(f"no .pkl in {a.input_dir}")
    if not a.dry_run:
        os.makedirs(a.output_dir, exist_ok=True)
    print(f"{len(src_files)} rollouts: {a.input_dir}\n  -> {a.output_dir}"
          f"  (ramp_tol={a.ramp_tol}, dry_run={a.dry_run})")

    prov = {"operation": "excise_extra_closes",
            "source_dir": os.path.abspath(a.input_dir),
            "grip_open_thr": GRIP_OPEN_THR, "ramp_tol": a.ramp_tol,
            "rollouts": {}}
    tin = tout = 0
    for s in src_files:
        name = os.path.basename(s)
        if a.dry_run:
            with open(s, "rb") as f:
                g = np.asarray(pickle.load(f)["actions_single_arm"], np.float32)[:, 6]
            keep, cuts, n_closes = excise_mask(g, a.ramp_tol)
            T, K = len(g), int(keep.sum())
        else:
            T, K, cuts, n_closes = process(s, os.path.join(a.output_dir, name), a.ramp_tol)
        tin += T; tout += K
        entry = {"T_in": T, "T_out": K, "n_closes": n_closes,
                 "cut_spans": [[int(i), int(j)] for i, j in cuts]}
        if n_closes == 0:
            entry["warning"] = "no close at all -- never grasped?"
            print(f"  {name:28s} closes=0  WARN: never grasped?")
        if cuts:
            print(f"  {name:28s} closes={n_closes}  cut {cuts}  {T} -> {K}")
        prov["rollouts"][name] = entry
        gc.collect()
    print(f"TOTAL {tin} -> {tout} frames  ({100*(tin-tout)/tin:.1f}% excised)")
    if not a.dry_run:
        with open(os.path.join(a.output_dir, "EXCISE_PROVENANCE.json"), "w") as f:
            json.dump(prov, f, indent=2)


if __name__ == "__main__":
    main()

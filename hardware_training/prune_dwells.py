#!/usr/bin/env python3
"""Cap long stationary runs in converted rollouts, and unwrap rpy.

Two defects in the recorded demos, both measured on hardware:

1. LONG DWELLS. The arm holds a pose for tens of frames -- cardboard has 32-41
   frame still runs sitting 43-64 frames before the grasp; bird has ~20 frames
   at the start. With horizon=20 a dwell longer than the horizon means every
   step of the predicted chunk is identical, so the policy has no signal for
   when to move on and stalls there during execution.

   We CAP rather than delete. Dropping every still frame would teach the policy
   never to pause, and some pauses are real (settling, waiting on the gripper).
   Keeping --keep frames preserves "there is a pause here" while guaranteeing
   the exit is visible inside the horizon.

2. RPY WRAP. Roll sits within 0.01 rad of +/-pi for 78% of bird frames, so a
   0.0024 rad wobble flips the recorded value by 2*pi. That makes the target
   bimodal at a visually identical scene; sampled chunks land between the modes
   and command a ~160 deg wrist flip. np.unwrap removes it -- the flips are
   numerical, not physical.

A frame is "still" only if neither the pose nor the gripper is changing, so
frames where the gripper is closing are never pruned.

Two-arm recordings add two behaviors, both off by default so the single-arm
base data is reproduced byte-for-byte:

3. --protect-closed. A still run with the gripper CLOSED is a coordination
   wait (bird arm hovering with the bird until the lid is off; cardboard arm
   holding the lid aside until the bird is dropped) and is never capped. Only
   open-gripper dwells are dead time.

4. --relabel-accidental-closes. After the drop the operator sometimes pressed
   close/open before going home. A close is accidental iff it is short
   (< --accidental-max-frames) AND the arm barely moved during it
   (< --accidental-max-mm). That catches the end-of-episode blips and a failed
   grasp attempt, and never touches a real grasp, which always carries the
   object. The gripper label is rewritten to the pre-close open value; frames
   are kept. Ordinal rules ("the second close") are wrong here: one cardboard
   rollout's first close is a failed grasp and its second is the real one.

--fix-gripper-init: the first 1-2 recorded frames carry gripper=0, an unissued
command, not a closed gripper. They are set to the first real value so they do
not read as "holding" under --protect-closed.

Camera keys that alias the same array on input stay aliased on output (the
pickle memo stores the pixels once); slicing each key separately tripled the
file size and doubled training RAM.
"""
import argparse, glob, gc, json, os, pickle, re
import numpy as np

KEYS_T = ("actions_single_arm", "camera_obs_shoulder", "state", "joint_angles",
          "camera_obs", "camera_obs0", "camera_obs_shoulder0")
GRIP_OPEN_THR = 425.0  # xArm gripper: ~800-850 open, ~0-160 closed


def fix_gripper_init(act, max_frames=5):
    """Leading gripper=0 frames are an unissued command; set them to the first real value."""
    g = act[:, 6]
    real = np.where(g > GRIP_OPEN_THR)[0]
    if len(real) == 0 or real[0] == 0 or real[0] > max_frames:
        return 0
    act[: real[0], 6] = g[real[0]]
    return int(real[0])


def accidental_close_spans(act, max_frames, max_mm):
    """(start, end_exclusive) of every closed interval that is short and stationary."""
    closed = act[:, 6] < GRIP_OPEN_THR
    spans = []
    i = 0
    T = len(act)
    while i < T:
        if not closed[i]:
            i += 1
            continue
        j = i
        while j < T and closed[j]:
            j += 1
        if i > 0:  # a closed run starting at frame 0 is the init artifact, not a close
            path = float(np.sum(np.linalg.norm(np.diff(act[i:j, :3], axis=0), axis=1))) if j - i > 1 else 0.0
            if (j - i) < max_frames and path < max_mm:
                spans.append((i, j, path))
        i = j
    return spans


def relabel_accidental_closes(act, max_frames, max_mm, ramp_tol=20.0):
    """Relabel each accidental close to fully open, INCLUDING its closing and
    opening ramps. The closed span itself is only the frames below the
    open/closed threshold; the gripper takes several frames to get there and
    back (e.g. 800 -> 643 -> 538 -> 432 -> ... -> 429 -> 541 -> 654 -> 800).
    Relabelling only the sub-threshold frames, to the value just before them,
    left a 2 s dip to half-closed in the labels -- and the policy reproduced
    it live as a re-close after the drop. Extend each span outward while the
    gripper is more than ramp_tol below the fully-open value on either side,
    and set the whole thing to that open value."""
    spans = accidental_close_spans(act, max_frames, max_mm)
    g = act[:, 6]
    out = []
    for i, j, path in spans:
        # fully-open reference: the max gripper value in the 60 frames before the blip
        open_val = float(g[max(0, i - 60):i].max())
        lo = i
        while lo > 0 and g[lo - 1] < open_val - ramp_tol:
            lo -= 1
        hi = j
        while hi < len(g) and g[hi] < open_val - ramp_tol:
            hi += 1
        act[lo:hi, 6] = open_val
        out.append((lo, hi, path))
    return out


def still_mask(actions, pos_tol, grip_tol):
    dp = np.r_[0.0, np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)]
    dg = np.r_[0.0, np.abs(np.diff(actions[:, 6]))]
    return (dp < pos_tol) & (dg < grip_tol)


def keep_indices(actions, keep, pos_tol, grip_tol, protect_closed=False):
    """Indices to retain: cap every contiguous still run at `keep` frames.

    With protect_closed, a still frame whose gripper is closed is treated as
    moving, i.e. never capped: the arm is holding something and waiting.
    """
    still = still_mask(actions, pos_tol, grip_tol)
    if protect_closed:
        still &= actions[:, 6] > GRIP_OPEN_THR
    T = len(actions)
    out = []
    i = 0
    while i < T:
        if not still[i]:
            out.append(i); i += 1
            continue
        j = i
        while j < T and still[j]:
            j += 1
        out.extend(range(i, min(i + keep, j)))   # head of the run only
        i = j
    return np.asarray(out, dtype=np.int64)


def process(src, dst, keep, pos_tol, grip_tol, unwrap, protect_closed=False,
            fix_init=False, relabel=None):
    """relabel: None, or (max_frames, max_mm) for --relabel-accidental-closes.
    Returns (T_in, T_out, info dict)."""
    with open(src, "rb") as f:
        r = pickle.load(f)
    act = np.asarray(r["actions_single_arm"], dtype=np.float32).copy()
    T = len(act)
    info = {}
    if fix_init:
        info["gripper_init_frames_fixed"] = fix_gripper_init(act)
    if relabel is not None:
        spans = relabel_accidental_closes(act, *relabel)
        info["accidental_closes_relabelled"] = [
            {"start": int(i), "end": int(j), "frames": int(j - i), "path_mm": round(p, 1)}
            for i, j, p in spans
        ]
    if unwrap:
        act[:, 3:6] = np.unwrap(act[:, 3:6], axis=0)
    idx = keep_indices(act, keep, pos_tol, grip_tol, protect_closed)

    out = {}
    sliced = {}  # id(original array) -> sliced array, so aliases stay aliased
    for k, v in r.items():
        if k in KEYS_T and hasattr(v, "__len__") and len(v) == T:
            if k in ("actions_single_arm", "state"):
                continue  # rebuilt from act below so gripper/rpy edits land in both
            key = id(v)
            if key not in sliced:
                sliced[key] = np.asarray(v)[idx].copy()
            out[k] = sliced[key]
        else:
            out[k] = v
    out["actions_single_arm"] = act[idx].copy()
    if "state" in r:
        st = np.asarray(r["state"], dtype=np.float32).copy()
        st[:, :7] = act[:, :7]  # state mirrors the corrected action stream
        out["state"] = st[idx].copy()
    with open(dst, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return T, len(idx), info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--keep", type=int, default=5,
                   help="frames retained from each still run. Must be well under the "
                        "training horizon (20) so the chunk always contains the exit.")
    p.add_argument("--pos-tol", type=float, default=0.5, help="mm; below this the pose is 'not moving'")
    p.add_argument("--grip-tol", type=float, default=5.0, help="gripper units; below this it is 'not moving'")
    p.add_argument("--no-unwrap", action="store_true", help="skip the rpy unwrap")
    p.add_argument("--protect-closed", action="store_true",
                   help="never cap a still run while the gripper is closed (coordination wait)")
    p.add_argument("--fix-gripper-init", action="store_true",
                   help="set leading gripper=0 frames (unissued command) to the first real value")
    p.add_argument("--relabel-accidental-closes", action="store_true",
                   help="rewrite short, stationary closes to open (see module doc)")
    p.add_argument("--accidental-max-frames", type=int, default=40)
    p.add_argument("--accidental-max-mm", type=float, default=15.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    relabel = (a.accidental_max_frames, a.accidental_max_mm) if a.relabel_accidental_closes else None

    src_files = sorted(glob.glob(os.path.join(a.input_dir, "*.pkl")))
    if not src_files:
        raise SystemExit(f"no .pkl in {a.input_dir}")
    if not a.dry_run:
        os.makedirs(a.output_dir, exist_ok=True)
    print(f"{len(src_files)} rollouts: {a.input_dir}\n  -> {a.output_dir}"
          f"  (keep={a.keep}, unwrap={not a.no_unwrap}, protect_closed={a.protect_closed}, "
          f"fix_init={a.fix_gripper_init}, relabel={relabel})")
    tin = tout = 0
    prov = {"operation": "prune_dwells", "source_dir": os.path.abspath(a.input_dir),
            "args": vars(a), "rollouts": {}}
    for i, s in enumerate(src_files):
        d = os.path.join(a.output_dir, os.path.basename(s))
        if a.dry_run:
            with open(s, "rb") as f:
                act = np.asarray(pickle.load(f)["actions_single_arm"], np.float32).copy()
            info = {}
            if a.fix_gripper_init:
                info["gripper_init_frames_fixed"] = fix_gripper_init(act)
            if relabel:
                info["accidental_closes_relabelled"] = relabel_accidental_closes(act, *relabel)
            T, K = len(act), len(keep_indices(act, a.keep, a.pos_tol, a.grip_tol, a.protect_closed))
        else:
            T, K, info = process(s, d, a.keep, a.pos_tol, a.grip_tol, not a.no_unwrap,
                                 a.protect_closed, a.fix_gripper_init, relabel)
        tin += T; tout += K
        prov["rollouts"][os.path.basename(s)] = {"T_in": T, "T_out": K, **info}
        flag = ""
        if info.get("accidental_closes_relabelled"):
            flag = f"  RELABELLED {len(info['accidental_closes_relabelled'])} close(s)"
        if i < 3 or i % 25 == 0 or flag:
            print(f"  {os.path.basename(s):32s} {T:5d} -> {K:5d}  ({100*(T-K)/T:4.1f}% pruned){flag}")
        gc.collect()
    print(f"TOTAL {tin} -> {tout} frames  ({100*(tin-tout)/tin:.1f}% pruned)")
    if not a.dry_run:
        with open(os.path.join(a.output_dir, "PRUNE_PROVENANCE.json"), "w") as f:
            json.dump(prov, f, indent=2, default=str)


if __name__ == "__main__":
    main()

"""Prepare audited A/B inputs without changing original recordings or targets."""

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter

from hardware_training.convert_npz_to_pkl import resize_frames, unwrap_rotations
from hardware_training.prune_dwells import fix_gripper_init, keep_indices, relabel_accidental_closes


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sustained_start(grip, closed):
    valid = grip < 300 if closed else grip >= 500
    for start in range(min(21, len(grip) - 5)):
        if valid[start:start + 5].all():
            return start
    raise ValueError("No valid startup gripper state within 20 frames")


def phase_labels(actions, arm, horizon=20):
    grip = actions[:, 6]
    closed = grip < 300
    runs = np.convolve(closed.astype(int), np.ones(10, dtype=int), mode="valid")
    starts = np.flatnonzero(runs == 10)
    starts = starts[starts > 5]
    if not len(starts):
        raise ValueError("No sustained expert grasp")
    grasp = int(starts[0])
    opens = np.flatnonzero((grip >= 500) & (np.arange(len(grip)) > grasp))
    if not len(opens):
        raise ValueError("No expert release after grasp")
    release = int(opens[0])
    labels = np.zeros(len(actions), dtype=np.int64)
    labels[max(0, grasp - horizon + 1):grasp + 10] = 1
    labels[grasp + 10:max(grasp + 10, release - horizon + 1)] = 2
    labels[max(grasp + 10, release - horizon + 1):release + 10] = 4
    labels[release + 10:] = 5
    if arm == 1:
        peak = grasp + int(actions[grasp:release, 1].argmax())
        aside = np.flatnonzero((actions[:, 1] >= actions[peak, 1] - 12) & (np.arange(len(actions)) >= grasp + 10) & (np.arange(len(actions)) < release))
        if len(aside):
            labels[aside[0]:aside[-1] + 1] = 3
            labels[aside[-1] + 1:release + 10] = 4
    else:
        for anchor in range(grasp + 10, max(grasp + 10, release - horizon + 1)):
            chunk = actions[anchor:anchor + horizon]
            near_basket = np.linalg.norm(actions[anchor, :3] - actions[release, :3]) <= 45
            if near_basket and (chunk[:, 6] < 300).all() and np.linalg.norm(chunk[:, :3] - chunk[0, :3], axis=1).max() <= 15:
                labels[anchor] = 3
    return labels, {"grasp": grasp, "release": release}


def render_phases(directory, key, actions, images, indices, labels, events):
    sheet = Image.new("RGB", (1280, 440), "white")
    draw = ImageDraw.Draw(sheet)
    frames = sorted(set([0, len(actions)-1] + [int(group[len(group)//2]) for phase in range(6) if len(group := np.flatnonzero(labels == phase))] + [events["grasp"], events["release"]]))
    for column, frame in enumerate(frames[:10]):
        left, top = column % 5 * 256, column // 5 * 220
        draw.text((left, top), f"{key} f{frame} phase{labels[frame]} G{actions[frame,6]:.0f}", fill="black")
        sheet.paste(Image.fromarray(images[indices[frame]]), (left, top + 20))
    sheet.save(directory / "phase_review.jpg")


def save_trajectory(output, key, task, source, actions, images, timestamps, indices, weights, provenance, arm=None):
    if len(images) != len(timestamps) or not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"Invalid camera timestamps: {key}")
    if not np.isfinite(actions).all() or np.any(np.diff(indices) <= 0):
        raise ValueError(f"Invalid actions/mapping: {key}")
    directory = output / key
    directory.mkdir(parents=True, exist_ok=True)
    labels, events = phase_labels(actions, arm) if arm is not None else (np.zeros(len(actions), dtype=np.int64), {})
    np.save(directory / "images.npy", images)
    np.savez(directory / "trajectory.npz", actions=actions, timestamps_ns=timestamps,
             raw_indices=indices, weights=weights, phases=labels)
    record = dict(key=key, task=task, arm=arm, source=str(source), directory=str(directory.resolve()),
                  frames=len(actions), raw_frames=len(images), actions_sha256=digest(actions),
                  camera_sha256=digest(images), timestamps_sha256=digest(timestamps),
                  phase_counts=np.bincount(labels, minlength=6).tolist(), events=events,
                  provenance=provenance)
    (directory / "record.json").write_text(json.dumps(record, indent=2))
    if arm is not None:
        render_phases(directory, key, actions, images, indices, labels, events)
    print(key, len(actions), "mapped", provenance, flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-phases", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recordings-root", type=Path, required=True, help="Parent of the original twoarm recording directories")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.refresh_phases:
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["records"]:
            if record["arm"] is None:
                continue
            directory = Path(record["directory"])
            with np.load(directory / "trajectory.npz") as source:
                data = {key: source[key] for key in source.files}
            labels, events = phase_labels(data["actions"], record["arm"])
            data["phases"] = labels
            np.savez(directory / "trajectory.npz", **data)
            record.update(phase_counts=np.bincount(labels, minlength=6).tolist(), events=events)
            (directory / "record.json").write_text(json.dumps(record, indent=2))
            images = np.load(directory / "images.npy", mmap_mode="r")
            render_phases(directory, record["key"], data["actions"], images, data["raw_indices"], labels, events)
        manifest["phase_definition_version"] = 2
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return
    root = args.data_root
    baseline_path = root / "manifests/aug22_twoarm_distillh20/hardware_placewipe_distillh20_isw5k20_ta39_sa20.json"
    baseline = json.loads(baseline_path.read_text())
    selection = json.loads(args.selection.read_text())
    records = []

    def cached(key):
        path = output / key / "record.json"
        if args.resume and path.exists():
            record = json.loads(path.read_text())
            records.append(record)
            return True
        return False

    for path_string in baseline["twoarm"]["files_by_mode"]["0"]:
        path = Path(path_string)
        rollout_id = int(re.search(r"rollout_(\d+)", path.name)[1])
        with path.open("rb") as stream:
            packed = pickle.load(stream)
        for arm, name in enumerate(["bird", "cardboard"]):
            key = f"ta_{name}_{rollout_id:02d}"
            if cached(key):
                continue
            raw_path = root / f"rollouts/twoarm_aug22_{name}_arm_192x256/rollout_{rollout_id}.pkl"
            with raw_path.open("rb") as stream:
                raw = pickle.load(stream)
            raw_actions = np.asarray(raw["actions_single_arm"], dtype=np.float32).copy()
            fix_gripper_init(raw_actions)
            relabel_accidental_closes(raw_actions, 40, 15)
            raw_actions[:, 3:6] = np.unwrap(raw_actions[:, 3:6], axis=0)
            indices = keep_indices(raw_actions, 20, .5, 5, True)
            current = packed[f"camera_obs_shoulder{arm}"]
            actions = np.asarray(packed["actions"][:len(current), arm*7:(arm+1)*7], dtype=np.float32)
            if not np.array_equal(raw_actions[indices], actions):
                raise ValueError(f"TA action reconstruction mismatch: {key}")
            if not np.array_equal(raw["camera_obs_shoulder"][indices], current):
                raise ValueError(f"TA image reconstruction mismatch: {key}")
            original_path = args.recordings_root / f"twoarm_{name}_Aug_22_2026/rollout_{rollout_id}/rollout.npz"
            with np.load(original_path) as original:
                timestamps = original["timestamps_ns"]
                if not np.array_equal(original["state"][:, :3].astype(np.float32), raw["state"][:, :3]):
                    raise ValueError(f"TA raw timestamp source mismatch: {key}")
            record = save_trajectory(output, key, "twoarm", path, actions, raw["camera_obs_shoulder"], timestamps, indices,
                                     np.asarray(packed["frame_weights"][:len(actions), arm]),
                                     {"raw_npz": str(original_path), "mapping": "exact reconstructed keep20/protect_closed; all current images/actions identical"}, arm)
            records.append(record)

    for path_string in baseline["singlearm"]["pickup"]["files_by_mode"]["0"]:
        path = Path(path_string)
        rollout_id = int(re.search(r"rollout_(\d+)", path.name)[1])
        key = f"sa_bird_{rollout_id:02d}"
        if cached(key):
            continue
        with path.open("rb") as stream:
            processed = pickle.load(stream)
        original_path = root / f"policy_rollouts/bird_h20s15_sep10/rollout_{rollout_id}/rollout.npz"
        with np.load(original_path) as original:
            raw_actions = unwrap_rotations(original["state"].astype(np.float32))
            timestamps = original["timestamps_ns"]
            images = resize_frames(original["camera_top"], 192, 256)
        raw_actions[:, :3] = savgol_filter(raw_actions[:, :3], 7, 3, axis=0)
        raw_actions[:, 3:6] = savgol_filter(np.unwrap(raw_actions[:, 3:6], axis=0), 7, 3, axis=0)
        raw_actions[:, 3:6] = np.unwrap(raw_actions[:, 3:6], axis=0)
        indices = keep_indices(raw_actions, 5, .5, 5, False)
        actions = np.asarray(processed["actions_single_arm"], dtype=np.float32)
        if not np.array_equal(raw_actions[indices], actions) or not np.array_equal(images[indices], processed["camera_obs_shoulder"]):
            raise ValueError(f"Bird source reconstruction mismatch: {key}")
        records.append(save_trajectory(output, key, "bird", path, actions, images, timestamps, indices, np.ones(len(actions)),
                                       {"raw_npz": str(original_path), "mapping": "exact reconstructed savgol7/3/keep5; current images/actions unchanged"}))

    for mode in ["fwd", "bwd"]:
        for path_string in selection["cardboard"][mode]["source_files"]:
            path = Path(path_string)
            rollout_id = int(re.search(r"rollout_(\d+)", str(path))[1])
            key = f"sa_{mode}_{rollout_id:02d}"
            if cached(key):
                continue
            with np.load(path) as original:
                raw_actions = original["state"].astype(np.float32)
                start = sustained_start(raw_actions[:, 6], mode == "bwd")
                stop = 581 if mode == "bwd" and rollout_id == 26 else len(raw_actions)
                timestamps = original["timestamps_ns"][start:stop]
                images = resize_frames(original["camera_top"][start:stop], 192, 256)
            actions = unwrap_rotations(raw_actions[start:stop])
            if np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).max() >= 25:
                raise ValueError(f"Unresolved pose discontinuity: {key}")
            indices = keep_indices(actions, 20, .5, 5, True)
            records.append(save_trajectory(output, key, mode, path, actions[indices], images, timestamps, indices,
                                           np.ones(len(indices)), {"trim_start": start, "trim_stop": stop, "original_frames": len(raw_actions),
                                           "pose_filter": "none", "pruning": "keep20 protect_closed", "reason": "verified startup cache; bwd26 tail ends at reviewed clean frame580"}))
    if len(records) != 146:
        raise ValueError(f"Expected 78 TA + 20 bird + 48 cardboard trajectories, got {len(records)}")
    manifest = {"schema": "coordination_ab_raw_history.v1", "records": records, "baseline_manifest": str(baseline_path),
                "source_selection": selection, "history_seconds": [1.0, .5, 0.0], "singlearm_task_mass": {"bird": 16766, "fwd": 9024, "bwd": 8122}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("READY", output / "manifest.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate distilled learned-policy rollouts using the canonical success gate."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from robosuite.controllers import load_composite_controller_config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.arm.data_gen.utils.singlearm_wipe_env_hard import SingleArmPlaceWipeHard
from utils.eval_metrics import SingleArmPlaceWipeMetricsCollectorMixin


class CanonicalReplay(SingleArmPlaceWipeMetricsCollectorMixin):
    def __init__(self, env):
        self.env = env


def make_env(task: str):
    controller = load_composite_controller_config(robot="Kinova3", controller="envs/arm/data_gen/kinova.json")
    return SingleArmPlaceWipeHard(robots=["Kinova3"], task=task, gripper_types="default",
        controller_configs=controller, has_renderer=False, render_camera=None,
        has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True,
        control_freq=20, horizon=4000)


def replay(path: Path, envs: dict[str, SingleArmPlaceWipeHard], scale: float) -> dict:
    with path.open("rb") as handle:
        row = pickle.load(handle)
    required = {"observations", "actions", "object_state_local_robot0", "camera_obs0", "camera_obs_shoulder0", "seed", "task", "mode", "initial_sim_state", "metrics"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    task, mode, seed = row["task"], int(row["mode"]), int(row["seed"])
    if task not in envs:
        envs[task] = make_env(task)
    env = envs[task]
    np.random.seed(seed); env.set_mode(mode=mode); obs = env.reset()
    body_id = env.handle_cube_body_id if task == "place_return" else env.sponge_body_id
    if not hasattr(env, "_distill_base_mass"):
        env._distill_base_mass = {}
        env._distill_base_inertia = {}
    if body_id not in env._distill_base_mass:
        env._distill_base_mass[body_id] = float(env.sim.model.body_mass[body_id])
        env._distill_base_inertia[body_id] = env.sim.model.body_inertia[body_id].copy()
    env.sim.model.body_mass[body_id] = env._distill_base_mass[body_id] * scale
    env.sim.model.body_inertia[body_id] = env._distill_base_inertia[body_id] * scale
    env.sim.set_state_from_flattened(np.asarray(row["initial_sim_state"]))
    env.sim.forward()
    collector = CanonicalReplay(env); collector._metrics_on_reset(obs)
    for action in np.asarray(row["actions"], dtype=np.float32):
        obs, _, done, _ = env.step(action)
        collector._metrics_on_step(obs, action)
        if done:
            break
    metrics = collector._metrics_on_done()
    saved = row["metrics"]
    return {"path": str(path), "task": task, "mode": mode, "seed": seed,
            "saved_success": bool(saved["task_success"]), "replay_success": bool(metrics["task_success"]),
            "saved_metrics": saved, "replay_metrics": metrics,
            "schema_lengths": {k: len(row[k]) for k in ("observations", "actions", "object_state_local_robot0", "camera_obs0", "camera_obs_shoulder0")}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--per-mode", type=int, default=1, help="Representative files per mode; 0 validates all.")
    p.add_argument("--object-mass-scale", type=float, default=0.1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    files: list[Path] = []
    for root in args.roots:
        by_mode: dict[int, list[Path]] = {}
        for path in sorted(root.glob("*.pkl")):
            with path.open("rb") as handle:
                by_mode.setdefault(int(pickle.load(handle)["mode"]), []).append(path)
        for mode in sorted(by_mode):
            files.extend(by_mode[mode] if args.per_mode == 0 else by_mode[mode][:args.per_mode])
    envs: dict[str, SingleArmPlaceWipeHard] = {}
    try:
        rows = [replay(path, envs, args.object_mass_scale) for path in files]
    finally:
        for env in envs.values(): env.close()
    for row in rows:
        lengths = set(row["schema_lengths"].values())
        if row["saved_success"] is not True or row["replay_success"] is not True or len(lengths) != 1:
            raise RuntimeError(f"validation failed: {row['path']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"validated": len(rows), "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(f"validated {len(rows)} rollout(s): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

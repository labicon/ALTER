#!/usr/bin/env python3
"""Distill successful forward-only hard single-arm learned-policy rollouts.

The saved PKLs retain the standard single-arm training fields and add only
provenance / replay metadata.  Candidate failures are recorded in the per-task
manifest but never written into the nested training manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from robosuite.controllers import load_composite_controller_config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from envs.arm.data_gen.utils.singlearm_wipe_env_hard import SingleArmPlaceWipeHard
from envs.arm.sample.sample_single_arm_placewipe_shoulder import SingleArmPlaceWipeMPCPlayer
from src.image_diffusion import ImageConditional_ODE
from src.temporal import frame_offsets_from_stats


DEFAULT_DATA_ROOT = Path(os.environ.get("CODIFF_DATA_ROOT", "release-artifacts"))
DEFAULT_MODEL = DEFAULT_DATA_ROOT / "checkpoints/arm/singlearm_placewipe_e2e_shoulderonly_hard_big_smalltable_spongein_returnsettle_forwardonly_noaugment_100permode/singlearm_mixedfront_e2e_shoulder_final.pt"
DEFAULT_STATS = DEFAULT_MODEL.with_name("singlearm_mixedfront_e2e_shoulder_stats.pkl")
SELECTION_SEED = 20260728
TASKS = {"place_return": (0, 1), "wipe": (0, 1)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DistillPlayer(SingleArmPlaceWipeMPCPlayer):
    """Sampler variant that serializes exactly what the learned policy executed."""

    def run_and_record(self, seed: int, mode: int, max_steps: int, replan_freq: int,
                       guidance_w: float, sample_N: int, gripper_threshold: float,
                       warm_start_skip: float, use_warm_start: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = self.reset(seed, mode)
        self._metrics_on_reset(obs)
        target_body_id = self.env.handle_cube_body_id if self.env.task == "place_return" else self.env.sponge_body_id
        rollout: dict[str, Any] = {
            "observations": [], "actions": [], "object_state_local_robot0": [],
            "camera_obs0": [], "camera_obs_shoulder0": [],
            "seed": int(seed), "task": self.env.task, "mode": int(mode),
            "initial_qpos": self.env.sim.data.qpos.copy(),
            "initial_qvel": self.env.sim.data.qvel.copy(),
            "initial_sim_state": self.env.sim.get_state().flatten().copy(),
        }
        step = 0
        done = False
        prev_traj_norm = None
        while step < max_steps and not done:
            img_eih, img_shoulder = self._get_images()
            x_init = None
            if use_warm_start and prev_traj_norm is not None:
                shifted = prev_traj_norm[replan_freq:]
                pad = shifted[-1:].expand(replan_freq, -1)
                x_init = torch.cat([shifted, pad], dim=0).unsqueeze(0).to(self.device)
            shoulder_only = self.num_cameras == 1
            traj_norm = self.model.sample(
                imgs_eih=None if shoulder_only else img_eih,
                imgs_shoulder=img_shoulder, traj_len=self.horizon, n_samples=1,
                w=guidance_w, N=sample_N, x_init=x_init,
                warm_start_skip=warm_start_skip,
            ).squeeze(0)
            prev_traj_norm = traj_norm.clone()
            traj = traj_norm.cpu().numpy() * self.std_action + self.mean_action
            for t in range(min(replan_freq, len(traj), max_steps - step)):
                action = self._postprocess_gripper(traj[t], gripper_threshold).astype(np.float32)
                obs, _, done, _ = self.env.step(action)
                self.image_history.append(self._render_temporal_frames())
                self._metrics_on_step(obs, action)
                # Match the established training schema: action-aligned obs/cameras.
                rollout["observations"].append({k: np.asarray(v).copy() for k, v in obs.items()})
                rollout["actions"].append(action.copy())
                rollout["object_state_local_robot0"].append(
                    self.env.sim.data.body_xpos[target_body_id].copy()
                )
                rollout["camera_obs0"].append(self._render_camera(self.eih_camera_name))
                rollout["camera_obs_shoulder0"].append(self._render_camera(self.shoulder_camera_name))
                step += 1
                if done or step >= max_steps:
                    break
        for key in ("actions", "object_state_local_robot0", "camera_obs0", "camera_obs_shoulder0"):
            rollout[key] = np.asarray(rollout[key])
        metrics = self._metrics_on_done()
        rollout["metrics"] = metrics
        return rollout, metrics


def load_model(model_path: Path, stats_path: Path, device: torch.device) -> tuple[ImageConditional_ODE, dict[str, Any]]:
    with stats_path.open("rb") as handle:
        stats = pickle.load(handle)
    model = ImageConditional_ODE(
        x_dim=7, sigma_data=float(stats.get("sigma_data", 1.0)),
        d_model=int(stats.get("d_model", 256)), n_heads=int(stats.get("n_heads", 4)),
        depth=int(stats.get("depth", 3)), dim_feedforward=int(stats.get("dim_feedforward", 1024)),
        horizon=int(stats.get("horizon", 20)), device=device,
        num_cameras=int(stats.get("num_cameras", 2)), backbone=str(stats.get("backbone", "resnet18")),
        frame_offsets=frame_offsets_from_stats(stats),
    )
    if not model.load(str(model_path)):
        raise FileNotFoundError(model_path)
    model.F.eval(); model.F_ema.eval()
    return model, stats


def make_env(task: str) -> SingleArmPlaceWipeHard:
    config = load_composite_controller_config(robot="Kinova3", controller="envs/arm/data_gen/kinova.json")
    return SingleArmPlaceWipeHard(robots=["Kinova3"], task=task, gripper_types="default",
        controller_configs=config, has_renderer=False, render_camera=None, has_offscreen_renderer=True,
        use_camera_obs=False, use_object_obs=True, control_freq=20, horizon=4000)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def task_root(base: Path, task: str) -> Path:
    return base / ("singlearm_place_return_hard_big_smalltable_spongein_forwardonly_policy_distill15" if task == "place_return" else "singlearm_wipe_hard_big_smalltable_spongein_forwardonly_policy_distill15")


def build_nested(root: Path, accepted: list[dict[str, Any]], task: str) -> dict[str, Any]:
    rng = random.Random(SELECTION_SEED)
    ordered: list[dict[str, Any]] = []
    for mode in TASKS[task]:
        rows = sorted((x for x in accepted if x["mode"] == mode), key=lambda x: x["path"])
        rng.shuffle(rows)
        ordered.extend(rows)
    out: dict[str, Any] = {"selection_seed": SELECTION_SEED, "nested": {}}
    for n in (5, 10, 15):
        files = []
        for mode in TASKS[task]:
            rows = [x for x in ordered if x["mode"] == mode][:n]
            files.extend(rows)
        out["nested"][f"{n}_per_mode"] = {"files": files, "count": len(files)}
    atomic_json(root / "nested_manifests.json", out)
    return out


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--stats-path", type=Path, default=DEFAULT_STATS)
    p.add_argument("--rollout-root", type=Path, default=DEFAULT_DATA_ROOT / "envs/arm/data_gen/rollouts")
    p.add_argument("--target-per-mode", type=int, default=15)
    p.add_argument("--smoke", action="store_true", help="Collect one accepted rollout per bucket in *_smoke roots.")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--replan-freq", type=int, default=None)
    p.add_argument("--guidance-w", type=float, default=1.2)
    p.add_argument("--sample-N", type=int, default=50)
    p.add_argument("--gripper-threshold", type=float, default=0.2)
    p.add_argument("--warm-start-skip", type=float, default=0.0)
    p.add_argument("--object-mass-scale", type=float, default=0.1)
    p.add_argument("--max-candidate-seed", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    cfg = args()
    if not cfg.model_path.is_file() or not cfg.stats_path.is_file():
        raise FileNotFoundError("model/stats path missing")
    device = torch.device(cfg.device)
    model, stats = load_model(cfg.model_path, cfg.stats_path, device)
    replan_freq = cfg.replan_freq or max(1, int(stats["horizon"]) // 2)
    target = 1 if cfg.smoke else cfg.target_per_mode
    common = {"model_path": str(cfg.model_path.resolve()), "model_sha256": sha256(cfg.model_path),
        "stats_path": str(cfg.stats_path.resolve()), "stats_sha256": sha256(cfg.stats_path),
        "environment": "SingleArmPlaceWipeHard", "sampler": {"sample_N": cfg.sample_N,
        "guidance_w": cfg.guidance_w, "warm_starts": True, "warm_start_skip": cfg.warm_start_skip,
        "replan_freq": replan_freq, "gripper_threshold": cfg.gripper_threshold,
        "object_mass_scale": cfg.object_mass_scale, "max_steps": cfg.max_steps}, "selection_seed": SELECTION_SEED}
    for task, modes in TASKS.items():
        root = task_root(cfg.rollout_root, task)
        if cfg.smoke:
            root = root.with_name(root.name + "_smoke")
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "provenance_manifest.json"
        manifest: dict[str, Any] = {**common, "task": task, "accepted": [], "rejected": []}
        if manifest_path.exists():
            manifest.update(json.loads(manifest_path.read_text()))
        accepted = [x for x in manifest["accepted"] if (root / x["path"]).is_file()]
        env = make_env(task)
        player = DistillPlayer(env, model, stats, device, obj_mass_scale=cfg.object_mass_scale)
        player.record_video = False
        try:
            for mode in modes:
                count = sum(x["mode"] == mode for x in accepted)
                seed = 0
                tried = {(x["mode"], x["seed"]) for x in accepted + manifest["rejected"]}
                while count < target:
                    while (mode, seed) in tried:
                        seed += 1
                    rollout, metrics = player.run_and_record(seed, mode, cfg.max_steps, replan_freq,
                        cfg.guidance_w, cfg.sample_N, cfg.gripper_threshold, cfg.warm_start_skip, True)
                    record = {"seed": seed, "mode": mode, "metrics": metrics}
                    if metrics["task_success"]:
                        name = f"policy_seed{seed}_mode{mode}.pkl"
                        with (root / name).open("wb") as handle:
                            pickle.dump(rollout, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        record["path"] = name
                        accepted.append(record); manifest["accepted"] = accepted; count += 1
                        print(f"ACCEPT task={task} mode={mode} seed={seed} ({count}/{target})")
                    else:
                        manifest["rejected"].append(record)
                        print(f"REJECT task={task} mode={mode} seed={seed}")
                    atomic_json(manifest_path, manifest)
                    seed += 1
                    if seed > cfg.max_candidate_seed:
                        raise RuntimeError(f"candidate-seed budget exhausted for {task} mode={mode}")
            if not cfg.smoke:
                build_nested(root, accepted, task)
        finally:
            env.close()
        rates = {}
        for mode in modes:
            a = sum(x["mode"] == mode for x in manifest["accepted"])
            r = sum(x["mode"] == mode for x in manifest["rejected"])
            rates[str(mode)] = {"accepted": a, "rejected": r, "acceptance_rate": a / max(a + r, 1)}
        manifest["acceptance_rates"] = rates
        atomic_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

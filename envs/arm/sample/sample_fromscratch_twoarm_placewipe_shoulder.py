"""Sample from-scratch two-arm policy on TwoArmPlaceWipe.

Loads a fresh ImageConditional_ODE trained directly on two-arm data.
Mirrors `sample_fromscratch_twoarm_e2e_shoulder.py` for `TwoArmPlaceWipe`.
"""

import argparse
import os
import pickle as pkl
import sys
from collections import deque

import imageio
import numpy as np
import torch
from robosuite.controllers import load_composite_controller_config

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from envs.arm.data_gen.utils.twoarm_wipe_env import TwoArmPlaceWipe
from envs.arm.sample.trajectory_viz_mixin import TrajectoryVizMixin
from envs.arm.sample.temporal_observation import TemporalObservationBuffer
from src.image_diffusion import ImageConditional_ODE
from src.temporal import frame_offsets_from_stats
from utils.eval_metrics import PlaceWipeMetricsCollectorMixin


class FromScratchPlaceWipeMPCPlayer(PlaceWipeMetricsCollectorMixin, TrajectoryVizMixin):
    def __init__(self, env, model, stats, device, camera_size=(128, 128),
                 obj_mass_scale=0.1, debug_conditioning=False):
        self.env = env
        self.model = model
        self.device = device
        self.camera_height = int(camera_size[0])
        self.camera_width = int(camera_size[1])
        self.obj_mass_scale = obj_mass_scale
        self.debug_conditioning = debug_conditioning

        self.mean_action = np.asarray(stats["action_mean"], dtype=np.float32)
        self.std_action = np.asarray(stats["action_std"], dtype=np.float32)
        self.horizon = int(stats["horizon"])
        self.num_cameras = int(stats.get("num_cameras", 2))
        self.frame_offsets = frame_offsets_from_stats(stats)
        self.image_history = TemporalObservationBuffer(self.frame_offsets)

        self.frames = []
        self._init_traj_viz()

    def _render_camera(self, camera_name: str) -> np.ndarray:
        for name in (camera_name, "agentview", "frontview"):
            try:
                frame = self.env.sim.render(
                    height=self.camera_height, width=self.camera_width, camera_name=name,
                )
                return np.flipud(frame).astype(np.uint8)
            except Exception:
                continue
        raise RuntimeError(f"Unable to render camera {camera_name}")

    @torch.no_grad()
    def _get_images(self, agent_idx):
        t_shoulder = self.image_history.batch(f"shoulder{agent_idx}", self.device)
        if self.num_cameras == 1:
            return None, t_shoulder
        t_eih = self.image_history.batch(f"eih{agent_idx}", self.device)
        return t_eih, t_shoulder

    def _render_temporal_frames(self):
        frames = {}
        for agent_idx in (0, 1):
            frames[f"shoulder{agent_idx}"] = self._render_camera(
                f"robot{agent_idx}_agentview_shoulder"
            )
            if self.num_cameras != 1:
                frames[f"eih{agent_idx}"] = self._render_camera(
                    f"robot{agent_idx}_eye_in_hand"
                )
        return frames

    def _capture_video_frame(self):
        main = np.flipud(self.env.sim.render(height=512, width=512, camera_name="sideview"))
        shoulder0 = np.flipud(self.env.sim.render(height=256, width=256,
                                                   camera_name="robot0_agentview_shoulder"))
        shoulder1 = np.flipud(self.env.sim.render(height=256, width=256,
                                                   camera_name="robot1_agentview_shoulder"))
        main = self._draw_trajectory_on_frame(main, "sideview", 512, 512)
        shoulder0 = self._draw_trajectory_on_frame(shoulder0, "robot0_agentview_shoulder", 256, 256)
        shoulder1 = self._draw_trajectory_on_frame(shoulder1, "robot1_agentview_shoulder", 256, 256)
        side_panel = np.concatenate([shoulder0, shoulder1], axis=0)
        frame = np.concatenate([main, side_panel], axis=1)
        self.frames.append(frame)

    def _save_video(self, path: str):
        if self.frames:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            imageio.mimsave(path, self.frames, fps=30)
            print(f"Saved video: {path}")

    @staticmethod
    def _postprocess_gripper(action, threshold):
        a = action.copy()
        g = float(a[6])
        a[6] = 1.0 if g > threshold else (-1.0 if g < -threshold else 0.0)
        return a

    def reset(self, seed):
        np.random.seed(seed)
        obs = self.env.reset()

        if not hasattr(self, "_cube_mass_base"):
            self._cube_mass_base = float(self.env.sim.model.body_mass[self.env.handle_cube_body_id])
            self._cube_inertia_base = self.env.sim.model.body_inertia[self.env.handle_cube_body_id].copy()
            self._sponge_mass_base = float(self.env.sim.model.body_mass[self.env.sponge_body_id])
            self._sponge_inertia_base = self.env.sim.model.body_inertia[self.env.sponge_body_id].copy()
        self.env.sim.model.body_mass[self.env.handle_cube_body_id] = self._cube_mass_base * self.obj_mass_scale
        self.env.sim.model.body_inertia[self.env.handle_cube_body_id] = self._cube_inertia_base * self.obj_mass_scale
        self.env.sim.model.body_mass[self.env.sponge_body_id] = self._sponge_mass_base * self.obj_mass_scale
        self.env.sim.model.body_inertia[self.env.sponge_body_id] = self._sponge_inertia_base * self.obj_mass_scale

        self.frames = []
        self.image_history.reset(self._render_temporal_frames())
        return obs

    def run(self, seed, max_steps=1200, replan_freq=10, gripper_threshold=0.2,
            guidance_w=1.2, sample_N=None, warm_start_skip=0.0, smooth_alpha=1.0,
            stall_window=0, stall_eef_thresh=0.003,
            video_path="results/fromscratch_twoarm_placewipe.mp4",
            collect_metrics=False):
        obs = self.reset(seed)
        self._metrics_on_reset(obs)
        shoulder_only = self.num_cameras == 1

        step = 0
        done = False
        prev_action_0 = None
        prev_action_1 = None
        prev_traj_norm_0 = None
        prev_traj_norm_1 = None
        eef_hist_0 = deque(maxlen=max(1, int(stall_window)))
        eef_hist_1 = deque(maxlen=max(1, int(stall_window)))
        eef_hist_0.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        eef_hist_1.append(np.asarray(obs["robot1_eef_pos"], dtype=np.float32))

        while step < max_steps and not done:
            trajs = []
            prev_trajs = [prev_traj_norm_0, prev_traj_norm_1]
            for agent_idx in [0, 1]:
                imgs_eih, imgs_shoulder = self._get_images(agent_idx)
                effective_replan_freq = replan_freq
                use_warm_start = prev_trajs[agent_idx] is not None

                x_init = None
                if use_warm_start:
                    shifted = prev_trajs[agent_idx][effective_replan_freq:]
                    pad = shifted[-1:].expand(effective_replan_freq, -1)
                    x_init = torch.cat([shifted, pad], dim=0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    traj_norm = self.model.sample(
                        imgs_eih=None if shoulder_only else imgs_eih,
                        imgs_shoulder=imgs_shoulder,
                        traj_len=self.horizon, n_samples=1, w=guidance_w,
                        N=sample_N, x_init=x_init, warm_start_skip=warm_start_skip,
                    ).squeeze(0)
                trajs.append(traj_norm)

            prev_traj_norm_0 = trajs[0].clone()
            prev_traj_norm_1 = trajs[1].clone()

            traj_0 = trajs[0].cpu().numpy() * self.std_action + self.mean_action
            traj_1 = trajs[1].cpu().numpy() * self.std_action + self.mean_action
            self._set_plan(traj_0, obs["robot0_eef_pos"], "robot0_base", plan_key="robot0")
            self._set_plan(traj_1, obs["robot1_eef_pos"], "robot1_base", plan_key="robot1")

            steps_to_exec = min(replan_freq, len(traj_0), len(traj_1), max_steps - step)
            for t in range(steps_to_exec):
                self._set_plan_step(t, plan_key="robot0")
                self._set_plan_step(t, plan_key="robot1")
                action_0 = self._postprocess_gripper(traj_0[t], gripper_threshold)
                action_1 = self._postprocess_gripper(traj_1[t], gripper_threshold)

                if smooth_alpha < 1.0 and prev_action_0 is not None:
                    action_0[:6] = smooth_alpha * action_0[:6] + (1 - smooth_alpha) * prev_action_0[:6]
                    action_0[6] = self._postprocess_gripper(action_0, gripper_threshold)[6]
                if smooth_alpha < 1.0 and prev_action_1 is not None:
                    action_1[:6] = smooth_alpha * action_1[:6] + (1 - smooth_alpha) * prev_action_1[:6]
                    action_1[6] = self._postprocess_gripper(action_1, gripper_threshold)[6]

                prev_action_0 = action_0.copy()
                prev_action_1 = action_1.copy()

                full_action = np.concatenate([action_0, action_1])
                obs, _, done, _ = self.env.step(full_action.astype(np.float32))
                self.image_history.append(self._render_temporal_frames())
                self._metrics_on_step(obs, full_action)
                self._capture_video_frame()
                step += 1
                eef_hist_0.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                eef_hist_1.append(np.asarray(obs["robot1_eef_pos"], dtype=np.float32))
                if done or step >= max_steps:
                    break

        self._save_video(video_path)
        metrics = self._metrics_on_done()
        ok = bool(metrics["task_success"])
        print(
            f"[seed={seed}] {'SUCCESS' if ok else 'FAIL'} "
            f"task_success={metrics['task_success']} "
            f"cube_returned={metrics['cube_returned']} "
            f"sponge_home={metrics['sponge_home']} "
            f"cube_visited_dest={metrics['cube_visited_dest']} "
            f"coverage_ok={metrics['coverage_ok']} "
            f"cube_to_return_final={metrics['cube_to_return_final']:.3f} "
            f"sponge_to_pad_final={metrics['sponge_to_pad_final']:.3f} "
            f"sponge_sweep_y={metrics['sponge_sweep_y']:.3f} "
            f"wipe_coverage_final={metrics['wipe_coverage_final']:.3f} "
            f"cube_to_destination_min={metrics['cube_to_destination_min']:.3f} "
            f"cube_to_return_min={metrics['cube_to_return_min']:.3f} "
            f"sponge_cube_collision={metrics['sponge_cube_collision']} "
            f"steps={step}"
        )
        if collect_metrics:
            return metrics
        return ok, metrics


def parse_args():
    p = argparse.ArgumentParser(description="Sample fromscratch two-arm policy on TwoArmPlaceWipe.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--stats-path", type=str, required=True)
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--replan-freq", type=int, default=None)
    p.add_argument("--gripper-threshold", type=float, default=0.2)
    p.add_argument("--guidance-w", type=float, default=1.2)
    p.add_argument("--sample-N", type=int, default=50)
    p.add_argument("--warm-start-skip", type=float, default=0.0)
    p.add_argument("--smooth-alpha", type=float, default=1.0)
    p.add_argument("--stall-window", type=int, default=0)
    p.add_argument("--stall-eef-thresh", type=float, default=0.003)
    p.add_argument("--camera-height", type=int, default=128)
    p.add_argument("--camera-width", type=int, default=128)
    p.add_argument("--debug-conditioning", action="store_true")
    p.add_argument("--video-path", type=str, default=None)
    p.add_argument("--env-variant", type=str, default="standard",
                   choices=["standard"],
                   help="Environment geometry variant to sample.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.stats_path, "rb") as f:
        stats = pkl.load(f)

    horizon = int(stats.get("horizon", 20))
    num_cameras = int(stats.get("num_cameras", 2))
    backbone = str(stats.get("backbone", "resnet18"))
    frame_offsets = frame_offsets_from_stats(stats)

    model = ImageConditional_ODE(
        x_dim=7,
        sigma_data=float(stats.get("sigma_data", 1.0)),
        d_model=int(stats.get("d_model", 256)),
        n_heads=int(stats.get("n_heads", 4)),
        depth=int(stats.get("depth", 3)),
        dim_feedforward=int(stats.get("dim_feedforward", 1024)),
        horizon=horizon, device=device,
        num_cameras=num_cameras, backbone=backbone,
        frame_offsets=frame_offsets,
    )
    if not model.load(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    model.F.eval(); model.F_ema.eval()

    controller_config = load_composite_controller_config(
        robot="Kinova3", controller="envs/arm/data_gen/kinova.json",
    )
    env_cls = TwoArmPlaceWipe
    env = env_cls(
        robots=["Kinova3", "Kinova3"],
        gripper_types="default", controller_configs=controller_config,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, render_camera=None,
    )

    dest_mode = args.mode // 2
    sponge_mode = args.mode % 2
    env.set_mode(dest_mode=dest_mode, sponge_mode=sponge_mode)
    print(f"mode={args.mode} -> (dest={dest_mode}, sponge={sponge_mode})")

    player = FromScratchPlaceWipeMPCPlayer(
        env=env, model=model, stats=stats, device=device,
        camera_size=(args.camera_height, args.camera_width),
        debug_conditioning=args.debug_conditioning,
    )

    replan_freq = args.replan_freq if args.replan_freq is not None else max(1, horizon // 2)

    video_path = args.video_path or (
        f"results/fromscratch_twoarm_placewipe_e2e_shoulder/"
        f"fromscratch_twoarm_placewipe_mode{args.mode}_seed-{args.seed}.mp4"
    )

    player.run(
        seed=args.seed, max_steps=args.max_steps, replan_freq=replan_freq,
        gripper_threshold=args.gripper_threshold, guidance_w=args.guidance_w,
        sample_N=args.sample_N, warm_start_skip=args.warm_start_skip,
        smooth_alpha=args.smooth_alpha,
        stall_window=args.stall_window, stall_eef_thresh=args.stall_eef_thresh,
        video_path=video_path,
    )


if __name__ == "__main__":
    main()

"""Sample coordinated two-arm policy on TwoArmPlaceWipe with e2e image coord head.

Both arms active. Each arm renders its own EIH + shoulder views, passes them
through the shared base encoder, and runs D = D_base + delta_head with CFG.

Mirrors `sample_coord_cube_e2e_shoulder.py` but targets `TwoArmPlaceWipe`.
"""

import argparse
import os
import pickle as pkl
import sys

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
from src.image_coordination import (
    ImageCoordinationHead, LEGACY_DECODER_EXECUTION, side_net_kwargs_from_stats,
)
from src.policy_selector import load_image_policy_selector
from src.temporal import frame_offsets_from_stats
from utils.eval_metrics import PlaceWipeMetricsCollectorMixin


class CoordinatedPlaceWipeMPCPlayer(PlaceWipeMetricsCollectorMixin, TrajectoryVizMixin):
    def __init__(self, env, base_model, coord_head, stats, device,
                 camera_size=(128, 128), obj_mass_scale=0.1, num_cameras=2,
                 selector=None, selector_mode="coord", selector_threshold=0.5):
        self.env = env
        self.base_model = base_model
        self.coord_head = coord_head
        self.selector = selector
        self.selector_mode = selector_mode
        self.selector_threshold = float(selector_threshold)
        self.device = device
        self.camera_height = int(camera_size[0])
        self.camera_width = int(camera_size[1])
        self.obj_mass_scale = obj_mass_scale
        self.num_cameras = num_cameras

        self.mean_action = stats["action_mean"]
        self.std_action = stats["action_std"]
        self.horizon = int(stats["horizon"])
        self.frame_offsets = frame_offsets_from_stats(stats)
        self.image_history = TemporalObservationBuffer(self.frame_offsets)

        self.frames = []
        self._init_traj_viz()

        robot0_base_body_id = self.env.sim.model.body_name2id("robot0_base")
        self.robot0_base_pos = self.env.sim.data.body_xpos[robot0_base_body_id]
        self.robot0_base_ori_rotm = self.env.sim.data.body_xmat[robot0_base_body_id].reshape((3, 3))
        robot1_base_body_id = self.env.sim.model.body_name2id("robot1_base")
        self.robot1_base_pos = self.env.sim.data.body_xpos[robot1_base_body_id]
        self.robot1_base_ori_rotm = self.env.sim.data.body_xmat[robot1_base_body_id].reshape((3, 3))

        self.agent_cameras = {
            0: {"eih": "robot0_eye_in_hand", "shoulder": "robot0_agentview_shoulder"},
            1: {"eih": "robot1_eye_in_hand", "shoulder": "robot1_agentview_shoulder"},
        }

    def _render_camera(self, camera_name: str) -> np.ndarray:
        for name in (camera_name, "agentview", "sideview"):
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
        for agent_idx, cameras in self.agent_cameras.items():
            frames[f"shoulder{agent_idx}"] = self._render_camera(cameras["shoulder"])
            if self.num_cameras != 1:
                frames[f"eih{agent_idx}"] = self._render_camera(cameras["eih"])
        return frames

    def _capture_video_frame(self):
        main = np.flipud(self.env.sim.render(height=512, width=512, camera_name="frontview"))
        shoulder0 = np.flipud(self.env.sim.render(height=256, width=256,
                                                   camera_name="robot0_agentview_shoulder"))
        shoulder1 = np.flipud(self.env.sim.render(height=256, width=256,
                                                   camera_name="robot1_agentview_shoulder"))
        main = self._draw_trajectory_on_frame(main, "frontview", 512, 512)
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

    @torch.no_grad()
    def _sample_base(self, imgs_eih, imgs_shoulder, x_init=None,
                     warm_start_skip=0.0, guidance_w=1.2, sample_N=None):
        shoulder_only = self.num_cameras == 1
        return self.base_model.sample(
            imgs_eih=None if shoulder_only else imgs_eih,
            imgs_shoulder=imgs_shoulder,
            traj_len=self.horizon,
            n_samples=1,
            w=guidance_w,
            N=sample_N,
            x_init=x_init,
            warm_start_skip=warm_start_skip,
        ).squeeze(0)

    @torch.no_grad()
    def _sample_coordinated(self, imgs_eih, imgs_shoulder, x_init=None,
                            warm_start_skip=0.0, guidance_w=1.2, sample_N=None):
        model = self.base_model
        n_samples = 1

        if sample_N is not None and sample_N != model.N:
            model.set_N(sample_N)

        enc_cond = model.F_ema.forward_encoder(imgs_eih, imgs_shoulder)
        null_token = model.F_ema.null_token.expand(n_samples, -1, -1)
        enc_uncond = [null_token for _ in enc_cond]
        cfg_mask_cond = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        cfg_mask_uncond = torch.ones(n_samples, dtype=torch.bool, device=self.device)

        if x_init is not None:
            i_start = int(model.N * warm_start_skip)
            i_start = min(i_start, model.N - 1)
            x = x_init + torch.randn_like(x_init) * model.sigma_s[i_start]
        else:
            i_start = 0
            x = torch.randn(
                (n_samples, self.horizon, 7), device=self.device
            ) * model.sigma_s[0] * model.scale_s[0]

        for i in range(i_start, model.N):
            sigma_i = torch.ones((n_samples, 1, 1), device=self.device) * model.sigma_s[i]
            D_base_cond = model._D_from_enc(x / model.scale_s[i], sigma_i, enc_cond)
            D_base_uncond = model._D_from_enc(x / model.scale_s[i], sigma_i, enc_uncond)
            delta_cond = self.coord_head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True, cfg_mask=cfg_mask_cond,
                d_base=D_base_cond, imgs_eih=imgs_eih, imgs_shoulder=imgs_shoulder,
            )
            delta_uncond = self.coord_head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True, cfg_mask=cfg_mask_uncond,
                d_base=D_base_uncond, imgs_eih=imgs_eih, imgs_shoulder=imgs_shoulder,
            )
            D = (guidance_w * (D_base_cond + delta_cond)
                 + (1 - guidance_w) * (D_base_uncond + delta_uncond))
            delta_step = model.coeff1[i] * x - model.coeff2[i] * D
            dt = model.t_s[i] - model.t_s[i + 1] if i != model.N - 1 else model.t_s[i]
            x = x - delta_step * dt

        return x.squeeze(0)

    @torch.no_grad()
    def _select_route(self, imgs_eih, imgs_shoulder, agent_idx):
        if self.selector_mode == "base":
            return False, 0.0
        if self.selector_mode == "coord":
            return True, 1.0
        if self.selector is None:
            raise ValueError("--selector-mode auto requires --selector-path")

        shoulder_only = self.num_cameras == 1
        prob = self.selector.predict_proba_from_images(
            self.base_model,
            None if shoulder_only else imgs_eih,
            imgs_shoulder,
        )
        p_coord = float(prob.item())
        use_coord = p_coord >= self.selector_threshold
        route = "coordination" if use_coord else "base"
        print(
            f"Selector route agent{agent_idx}: {route} "
            f"(p_coord={p_coord:.3f}, threshold={self.selector_threshold:.3f})"
        )
        return use_coord, p_coord

    def _sample_routed(self, use_coord, imgs_eih, imgs_shoulder, x_init=None,
                       warm_start_skip=0.0, guidance_w=1.2, sample_N=None):
        if use_coord:
            return self._sample_coordinated(
                imgs_eih, imgs_shoulder, x_init=x_init,
                warm_start_skip=warm_start_skip, guidance_w=guidance_w, sample_N=sample_N,
            )
        return self._sample_base(
            imgs_eih, imgs_shoulder, x_init=x_init,
            warm_start_skip=warm_start_skip, guidance_w=guidance_w, sample_N=sample_N,
        )

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

    def run(self, seed, max_steps=1200, replan_freq=10, replan_freq_grasping=None,
            gripper_threshold=0.2, guidance_w=1.2, sample_N=None, warm_start_skip=0.0,
            video_path="results/coord_twoarm_placewipe.mp4", collect_metrics=True):
        obs = self.reset(seed)
        self._metrics_on_reset(obs)

        step = 0
        done = False
        prev_traj_norm0 = None
        prev_traj_norm1 = None
        effective_replan = replan_freq
        use_coord0 = None
        use_coord1 = None

        while step < max_steps and not done:
            imgs_eih0, imgs_shoulder0 = self._get_images(0)
            imgs_eih1, imgs_shoulder1 = self._get_images(1)

            if use_coord0 is None:
                use_coord0, _ = self._select_route(imgs_eih0, imgs_shoulder0, agent_idx=0)
            if use_coord1 is None:
                use_coord1, _ = self._select_route(imgs_eih1, imgs_shoulder1, agent_idx=1)

            x_init0 = None
            x_init1 = None
            if prev_traj_norm0 is not None:
                shifted = prev_traj_norm0[effective_replan:]
                pad = shifted[-1:].expand(effective_replan, -1)
                x_init0 = torch.cat([shifted, pad], dim=0).unsqueeze(0).to(self.device)
            if prev_traj_norm1 is not None:
                shifted = prev_traj_norm1[effective_replan:]
                pad = shifted[-1:].expand(effective_replan, -1)
                x_init1 = torch.cat([shifted, pad], dim=0).unsqueeze(0).to(self.device)

            traj0_norm = self._sample_routed(
                use_coord0, imgs_eih0, imgs_shoulder0, x_init=x_init0,
                warm_start_skip=warm_start_skip, guidance_w=guidance_w, sample_N=sample_N,
            )
            traj1_norm = self._sample_routed(
                use_coord1, imgs_eih1, imgs_shoulder1, x_init=x_init1,
                warm_start_skip=warm_start_skip, guidance_w=guidance_w, sample_N=sample_N,
            )

            prev_traj_norm0 = traj0_norm.clone()
            prev_traj_norm1 = traj1_norm.clone()

            traj0 = traj0_norm.cpu().numpy() * self.std_action + self.mean_action
            traj1 = traj1_norm.cpu().numpy() * self.std_action + self.mean_action
            self._set_plan(traj0, obs["robot0_eef_pos"], "robot0_base", plan_key="robot0")
            self._set_plan(traj1, obs["robot1_eef_pos"], "robot1_base", plan_key="robot1")

            if replan_freq_grasping is not None:
                window = min(replan_freq, len(traj0), len(traj1))
                grasping0 = all(
                    self._postprocess_gripper(traj0[k], gripper_threshold)[6] == 1.0
                    for k in range(window)
                )
                grasping1 = all(
                    self._postprocess_gripper(traj1[k], gripper_threshold)[6] == 1.0
                    for k in range(window)
                )
                freq0 = replan_freq_grasping if grasping0 else replan_freq
                freq1 = replan_freq_grasping if grasping1 else replan_freq
                effective_replan = min(freq0, freq1)
            else:
                effective_replan = replan_freq

            steps_to_exec = min(effective_replan, len(traj0), len(traj1), max_steps - step)
            for t in range(steps_to_exec):
                self._set_plan_step(t, plan_key="robot0")
                self._set_plan_step(t, plan_key="robot1")
                action0 = self._postprocess_gripper(traj0[t], gripper_threshold)
                action1 = self._postprocess_gripper(traj1[t], gripper_threshold)
                full_action = np.concatenate([action0, action1])
                obs, _, done, _ = self.env.step(full_action)
                self.image_history.append(self._render_temporal_frames())
                self._metrics_on_step(obs, full_action)
                self._capture_video_frame()
                step += 1
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
        return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Sample coord head on TwoArmPlaceWipe.")
    p.add_argument("--base-model-path", type=str, required=True)
    p.add_argument("--base-stats-path", type=str, required=True)
    p.add_argument("--coord-head-path", type=str, required=True)
    p.add_argument("--coord-stats-path", type=str, required=True)
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--replan-freq", type=int, default=10)
    p.add_argument("--replan-freq-grasping", type=int, default=None)
    p.add_argument("--gripper-threshold", type=float, default=0.2)
    p.add_argument("--guidance-w", type=float, default=1.2)
    p.add_argument("--sample-N", type=int, default=None)
    p.add_argument("--warm-start-skip", type=float, default=0.0)
    p.add_argument("--selector-path", type=str, default=None,
                   help="Optional learned selector checkpoint for auto routing.")
    p.add_argument("--selector-mode", type=str, default="coord",
                   choices=["coord", "base", "auto"],
                   help="Routing mode: current behavior, base-only ablation, or learned selector.")
    p.add_argument("--selector-threshold", type=float, default=0.5,
                   help="P(coordination) threshold used when --selector-mode auto.")
    p.add_argument("--camera-height", type=int, default=128)
    p.add_argument("--camera-width", type=int, default=128)
    p.add_argument("--video-path", type=str, default=None)
    p.add_argument("--env-variant", type=str, default="standard",
                   choices=["standard", "hard"],
                   help="Environment variant to sample.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.base_stats_path, "rb") as f:
        base_stats = pkl.load(f)

    sigma_data = float(base_stats.get("sigma_data", 1.0))
    d_model = int(base_stats.get("d_model", 256))
    n_heads = int(base_stats.get("n_heads", 4))
    depth = int(base_stats.get("depth", 3))
    dim_feedforward = int(base_stats.get("dim_feedforward", 1024))
    horizon = int(base_stats.get("horizon", 20))
    num_cameras = int(base_stats.get("num_cameras", 2))
    backbone = str(base_stats.get("backbone", "resnet18"))
    frame_offsets = frame_offsets_from_stats(base_stats)

    base_model = ImageConditional_ODE(
        x_dim=7, sigma_data=sigma_data,
        d_model=d_model, n_heads=n_heads, depth=depth,
        dim_feedforward=dim_feedforward, horizon=horizon,
        device=device, num_cameras=num_cameras, backbone=backbone,
        frame_offsets=frame_offsets,
    )
    if not base_model.load(args.base_model_path):
        raise FileNotFoundError(f"Base model not found: {args.base_model_path}")
    base_model.F.eval()

    with open(args.coord_stats_path, "rb") as f:
        coord_stats = pkl.load(f)

    head_d_model = int(coord_stats["head_d_model"])
    head_n_heads = int(coord_stats["head_n_heads"])
    head_depth = int(coord_stats["head_depth"])
    head_dim_ff = int(coord_stats["head_dim_feedforward"])
    base_d_model_from_stats = int(coord_stats.get("base_d_model", d_model))
    side_net_kwargs = side_net_kwargs_from_stats(coord_stats, base_model.F.tokens_per_camera)

    coord_head = ImageCoordinationHead(
        x_dim=7,
        base_d_model=base_d_model_from_stats,
        d_model=head_d_model, n_heads=head_n_heads,
        depth=head_depth, dim_feedforward=head_dim_ff,
        horizon=horizon, sigma_data=sigma_data,
        num_cameras=num_cameras,
        **side_net_kwargs,
        decoder_execution=str(coord_stats.get(
            "decoder_execution", LEGACY_DECODER_EXECUTION)),
        decoder_conditioning=str(coord_stats.get("decoder_conditioning", "pooled")),
        frame_offsets=frame_offsets,
    ).to(device)
    coord_head.load(args.coord_head_path, device=device)
    coord_head.eval()

    selector = None
    if args.selector_mode == "auto":
        if args.selector_path is None:
            raise ValueError("--selector-mode auto requires --selector-path")
        selector = load_image_policy_selector(args.selector_path, device=device)
        print(f"Loaded policy selector: {args.selector_path}")

    controller_config = load_composite_controller_config(
        robot="Kinova3", controller="envs/arm/data_gen/kinova.json",
    )
    if args.env_variant == "hard":
        from envs.arm.data_gen.utils.twoarm_wipe_env_hard import TwoArmPlaceWipeHard
        env_cls = TwoArmPlaceWipeHard
    else:
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

    player = CoordinatedPlaceWipeMPCPlayer(
        env=env, base_model=base_model, coord_head=coord_head,
        stats=coord_stats, device=device,
        camera_size=(args.camera_height, args.camera_width),
        num_cameras=num_cameras,
        selector=selector,
        selector_mode=args.selector_mode,
        selector_threshold=args.selector_threshold,
    )

    video_path = args.video_path or (
        f"results/coord_twoarm_placewipe_e2e_shoulder/"
        f"coord_twoarm_placewipe_mode{args.mode}_seed-{args.seed}.mp4"
    )

    player.run(
        seed=args.seed, max_steps=args.max_steps,
        replan_freq=args.replan_freq, replan_freq_grasping=args.replan_freq_grasping,
        gripper_threshold=args.gripper_threshold, guidance_w=args.guidance_w,
        sample_N=args.sample_N, warm_start_skip=args.warm_start_skip,
        video_path=video_path,
    )


if __name__ == "__main__":
    main()

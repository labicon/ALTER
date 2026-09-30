"""Sample the single-arm placewipe base policy.

Runs `SingleArmPlaceWipe` in one of the two subtasks:
  --subtask place_return  -> grasp cube, hold on dest pad, put back on return pad.
  --subtask wipe          -> grasp sponge, sweep dirt, return sponge to pad.

Mirrors the cube-sampler structure but targets `SingleArmPlaceWipe`.
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

from envs.arm.data_gen.utils.singlearm_wipe_env import SingleArmPlaceWipe
from envs.arm.data_gen.utils.singlearm_wipe_env_hard import SingleArmPlaceWipeHard
from envs.arm.sample.temporal_observation import TemporalObservationBuffer
from src.image_diffusion import ImageConditional_ODE
from src.image_coordination import (
    ImageCoordinationHead, LEGACY_DECODER_EXECUTION, side_net_kwargs_from_stats,
)
from src.policy_selector import load_image_policy_selector
from src.temporal import frame_offsets_from_stats
from utils.eval_metrics import SingleArmPlaceWipeMetricsCollectorMixin


class SingleArmPlaceWipeMPCPlayer(SingleArmPlaceWipeMetricsCollectorMixin):
    def __init__(self, env, model, stats, device, camera_size=(128, 128),
                 obj_mass_scale=0.1, selector=None, selector_mode="base",
                 selector_threshold=0.5, selector_strict=False, coord_head=None):
        self.env = env
        self.model = model
        self.selector = selector
        self.selector_mode = selector_mode
        self.selector_threshold = float(selector_threshold)
        self.selector_strict = bool(selector_strict)
        self.coord_head = coord_head
        self.device = device
        self.camera_height = int(camera_size[0])
        self.camera_width = int(camera_size[1])
        self.obj_mass_scale = obj_mass_scale

        self.eih_camera_name = "robot0_eye_in_hand"
        self.shoulder_camera_name = "robot0_agentview_shoulder"

        self.mean_action = np.asarray(stats["action_mean"], dtype=np.float32)
        self.std_action = np.asarray(stats["action_std"], dtype=np.float32)
        self.horizon = int(stats["horizon"])
        self.num_cameras = int(stats.get("num_cameras", 2))
        self.frame_offsets = frame_offsets_from_stats(stats)
        self.image_history = TemporalObservationBuffer(self.frame_offsets)
        self.frames = []

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
    def _get_images(self):
        t_shoulder = self.image_history.batch("shoulder", self.device)
        if self.num_cameras == 1:
            return None, t_shoulder
        t_eih = self.image_history.batch("eih", self.device)
        return t_eih, t_shoulder

    def _render_temporal_frames(self):
        frames = {"shoulder": self._render_camera(self.shoulder_camera_name)}
        if self.num_cameras != 1:
            frames["eih"] = self._render_camera(self.eih_camera_name)
        return frames

    def _capture_video_frame(self):
        if getattr(self, "record_video", True) is False:
            return
        # For the wipe subtask the arm sits on the +Y side and `sideview` only
        # captures the back of the arm, so use `frontview` instead to show the
        # arm and the dirt grid head-on.
        main_camera = "frontview" if self.env.task == "wipe" else "sideview"
        side = np.flipud(self.env.sim.render(height=512, width=512, camera_name=main_camera))
        shoulder = np.flipud(self.env.sim.render(
            height=256, width=256, camera_name=self.shoulder_camera_name))
        eih = np.flipud(self.env.sim.render(
            height=256, width=256, camera_name=self.eih_camera_name))
        side_panel = np.concatenate([shoulder, eih], axis=0)
        frame = np.concatenate([side, side_panel], axis=1)
        self.frames.append(frame)

    def _save_video(self, path: str):
        if getattr(self, "record_video", True) is False or not path:
            return
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
    def _select_route(self, img_eih, img_shoulder):
        if self.selector_mode == "base":
            return False, 0.0
        if self.selector_mode == "coord":
            return True, 1.0
        if self.selector is None:
            raise ValueError("--selector-mode auto requires --selector-path")

        shoulder_only = self.num_cameras == 1
        prob = self.selector.predict_proba_from_images(
            self.model,
            None if shoulder_only else img_eih,
            img_shoulder,
        )
        p_coord = float(prob.item())
        use_coord = p_coord >= self.selector_threshold
        route = "coordination" if use_coord else "base"
        print(
            f"Selector route singlearm: {route} "
            f"(p_coord={p_coord:.3f}, threshold={self.selector_threshold:.3f})"
        )
        return use_coord, p_coord

    @torch.no_grad()
    def _sample_coordinated(self, imgs_eih, imgs_shoulder, x_init=None,
                            warm_start_skip=0.0, guidance_w=1.2, sample_N=None):
        """Sample D_base + residual head directly, without a route selector."""
        if self.coord_head is None:
            raise ValueError("coord route requires a loaded coordination head")
        model = self.model
        if sample_N is not None and sample_N != model.N:
            model.set_N(sample_N)

        enc_cond = model.F_ema.forward_encoder(imgs_eih, imgs_shoulder)
        null_token = model.F_ema.null_token.expand(1, -1, -1)
        enc_uncond = [null_token for _ in enc_cond]
        cfg_cond = torch.zeros(1, dtype=torch.bool, device=self.device)
        cfg_uncond = torch.ones(1, dtype=torch.bool, device=self.device)

        if x_init is not None:
            i_start = min(int(model.N * warm_start_skip), model.N - 1)
            x = x_init + torch.randn_like(x_init) * model.sigma_s[i_start]
        else:
            i_start = 0
            x = torch.randn((1, self.horizon, 7), device=self.device)
            x = x * model.sigma_s[0] * model.scale_s[0]

        for i in range(i_start, model.N):
            sigma_i = torch.ones((1, 1, 1), device=self.device) * model.sigma_s[i]
            scaled_x = x / model.scale_s[i]
            base_cond = model._D_from_enc(scaled_x, sigma_i, enc_cond)
            base_uncond = model._D_from_enc(scaled_x, sigma_i, enc_uncond)
            delta_cond = self.coord_head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True, cfg_mask=cfg_cond,
                d_base=base_cond, imgs_eih=imgs_eih, imgs_shoulder=imgs_shoulder,
            )
            delta_uncond = self.coord_head.forward_residual(
                x, sigma_i, enc_cond, use_ema=True, cfg_mask=cfg_uncond,
                d_base=base_uncond, imgs_eih=imgs_eih, imgs_shoulder=imgs_shoulder,
            )
            denoised = (
                guidance_w * (base_cond + delta_cond)
                + (1.0 - guidance_w) * (base_uncond + delta_uncond)
            )
            delta_step = model.coeff1[i] * x - model.coeff2[i] * denoised
            dt = model.t_s[i] - model.t_s[i + 1] if i != model.N - 1 else model.t_s[i]
            x = x - delta_step * dt
        return x.squeeze(0)

    def reset(self, seed: int, mode: int = 0):
        np.random.seed(seed)
        if hasattr(self.env, "set_mode"):
            self.env.set_mode(mode=mode)
        obs = self.env.reset()

        # Lighten the manipulated object (cube for place_return, sponge for wipe).
        if self.env.task == "place_return":
            body_id = self.env.handle_cube_body_id
        else:
            body_id = self.env.sponge_body_id
        if not hasattr(self, "_obj_mass_base"):
            self._obj_mass_base = float(self.env.sim.model.body_mass[body_id])
            self._obj_inertia_base = self.env.sim.model.body_inertia[body_id].copy()
        self.env.sim.model.body_mass[body_id] = self._obj_mass_base * self.obj_mass_scale
        self.env.sim.model.body_inertia[body_id] = self._obj_inertia_base * self.obj_mass_scale

        self.frames = []
        self.image_history.reset(self._render_temporal_frames())
        return obs

    def run(self, seed, mode, max_steps, replan_freq, guidance_w, sample_N,
            gripper_threshold, warm_start_skip, video_path, use_warm_start=True):
        obs = self.reset(seed, mode)
        self._metrics_on_reset(obs)
        step = 0
        done = False
        prev_traj_norm = None
        selector_checked = False

        while step < max_steps and not done:
            img_eih, img_shoulder = self._get_images()

            if not selector_checked:
                use_coord, _ = self._select_route(img_eih, img_shoulder)
                selector_checked = True
                if use_coord and self.coord_head is None:
                    msg = "selector chose coordination in single-arm environment"
                    if self.selector_strict:
                        print(f"[seed={seed} subtask={self.env.task} mode={mode}] FAIL {msg} steps={step}")
                        self._save_video(video_path)
                        return False
                    print(f"Warning: {msg}; continuing with base policy for this single-arm rollout.")
                    use_coord = False

            x_init = None
            if use_warm_start and prev_traj_norm is not None:
                shifted = prev_traj_norm[replan_freq:]
                pad = shifted[-1:].expand(replan_freq, -1)
                x_init = torch.cat([shifted, pad], dim=0).unsqueeze(0).to(self.device)

            shoulder_only = self.num_cameras == 1
            if use_coord:
                traj_norm = self._sample_coordinated(
                    None if shoulder_only else img_eih,
                    img_shoulder,
                    x_init=x_init,
                    warm_start_skip=warm_start_skip,
                    guidance_w=guidance_w,
                    sample_N=sample_N,
                )
            else:
                traj_norm = self.model.sample(
                    imgs_eih=None if shoulder_only else img_eih,
                    imgs_shoulder=img_shoulder,
                    traj_len=self.horizon, n_samples=1, w=guidance_w,
                    N=sample_N, x_init=x_init, warm_start_skip=warm_start_skip,
                ).squeeze(0)

            prev_traj_norm = traj_norm.clone()
            traj = traj_norm.cpu().numpy() * self.std_action + self.mean_action

            steps_to_exec = min(replan_freq, len(traj), max_steps - step)
            for t in range(steps_to_exec):
                action = self._postprocess_gripper(traj[t], gripper_threshold)
                obs, _, done, _ = self.env.step(action.astype(np.float32))
                self.image_history.append(self._render_temporal_frames())
                self._metrics_on_step(obs, action)
                self._capture_video_frame()
                step += 1
                if done or step >= max_steps:
                    break

        self._save_video(video_path)
        metrics = self._metrics_on_done()
        ok = bool(metrics["task_success"])
        if metrics["subtask"] == "place_return":
            tail = (
                f"direction={metrics['direction']} "
                f"cube_to_target_final={metrics['cube_to_target_final']:.3f} "
                f"cube_to_destination_final={metrics['cube_to_destination_final']:.3f} "
                f"cube_to_return_final={metrics['cube_to_return_final']:.3f} "
                f"cube_to_destination_min={metrics['cube_to_destination_min']:.3f} "
                f"cube_to_return_min={metrics['cube_to_return_min']:.3f}"
            )
        else:
            tail = (
                f"wipe_coverage_final={metrics['wipe_coverage_final']:.3f} "
                f"sponge_sweep_y={metrics['sponge_sweep_y']:.3f} "
                f"sponge_home={metrics['sponge_home']} "
                f"sponge_to_pad_final={metrics['sponge_to_pad_final']:.3f} "
                f"coverage_ok={metrics['coverage_ok']}"
            )
        print(
            f"[seed={seed} subtask={metrics['subtask']} mode={mode}] "
            f"{'SUCCESS' if ok else 'FAIL'} task_success={metrics['task_success']} "
            f"{tail} steps={step}"
        )
        return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Sample single-arm placewipe base policy.")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--stats-path", type=str, required=True)
    p.add_argument("--subtask", type=str, default="place_return",
                   choices=["place_return", "wipe"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3],
                   help="For place_return: 0/1 = forward (center->dest pad 0/1), "
                        "2/3 = backward (dest pad 0/1->center). "
                        "For wipe: 0/1 = sponge pad position.")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--env-horizon", type=int, default=4000)
    p.add_argument("--replan-freq", type=int, default=None)
    p.add_argument("--sampling-seed", type=int, default=None,
                   help="Optional fixed Torch sampling seed supplied by a sealed evaluation contract.")
    p.add_argument("--guidance-w", type=float, default=1.2)
    p.add_argument("--sample-N", type=int, default=50)
    p.add_argument("--gripper-threshold", type=float, default=0.2)
    p.add_argument("--warm-start-skip", type=float, default=0.0)
    p.add_argument("--no-warm-start", action="store_true",
                   help="Disable trajectory warm starts between replans.")
    p.add_argument("--camera-height", type=int, default=128)
    p.add_argument("--camera-width", type=int, default=128)
    p.add_argument("--video-path", type=str, default=None)
    p.add_argument("--no-video", action="store_true",
                   help="Skip capturing/saving the rollout video.")
    p.add_argument("--env-variant", type=str, default="standard",
                   choices=["standard", "hard"],
                   help="Environment geometry/randomization variant to sample.")
    p.add_argument("--selector-path", type=str, default=None,
                   help="Optional learned selector checkpoint for route validation.")
    p.add_argument("--selector-mode", type=str, default="base",
                   choices=["base", "coord", "auto"],
                   help="Route mode. Use auto to validate that the selector chooses base.")
    p.add_argument("--selector-threshold", type=float, default=0.5,
                   help="P(coordination) threshold used when --selector-mode auto.")
    p.add_argument("--selector-strict", action="store_true",
                   help="Fail the rollout if auto selector chooses coordination.")
    p.add_argument("--coord-head-path", type=str, default=None,
                   help="Mixed coordination-head checkpoint for direct base+head inference.")
    p.add_argument("--coord-stats-path", type=str, default=None,
                   help="Stats pkl paired with --coord-head-path.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sampling_seed is not None:
        torch.manual_seed(args.sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.sampling_seed)
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

    coord_head = None
    if args.coord_head_path is not None or args.coord_stats_path is not None:
        if not args.coord_head_path or not args.coord_stats_path:
            raise ValueError("--coord-head-path and --coord-stats-path must be provided together")
        with open(args.coord_stats_path, "rb") as f:
            coord_stats = pkl.load(f)
        side_net_kwargs = side_net_kwargs_from_stats(coord_stats, model.F.tokens_per_camera)
        coord_head = ImageCoordinationHead(
            x_dim=7,
            base_d_model=int(coord_stats.get("base_d_model", stats.get("d_model", 256))),
            d_model=int(coord_stats["head_d_model"]),
            n_heads=int(coord_stats["head_n_heads"]),
            depth=int(coord_stats["head_depth"]),
            dim_feedforward=int(coord_stats["head_dim_feedforward"]),
            horizon=horizon,
            sigma_data=float(stats.get("sigma_data", 1.0)),
            num_cameras=num_cameras,
            **side_net_kwargs,
            decoder_execution=str(coord_stats.get(
                "decoder_execution", LEGACY_DECODER_EXECUTION
            )),
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
        env_cls = SingleArmPlaceWipeHard
    else:
        env_cls = SingleArmPlaceWipe
    env = env_cls(
        robots=["Kinova3"], task=args.subtask,
        gripper_types="default", controller_configs=controller_config,
        has_renderer=False, render_camera=None,
        has_offscreen_renderer=True, use_camera_obs=False,
        use_object_obs=True, control_freq=20, horizon=args.env_horizon,
    )

    player = SingleArmPlaceWipeMPCPlayer(
        env=env, model=model, stats=stats, device=device,
        camera_size=(args.camera_height, args.camera_width),
        selector=selector,
        selector_mode=args.selector_mode,
        selector_threshold=args.selector_threshold,
        selector_strict=args.selector_strict,
        coord_head=coord_head,
    )
    player.record_video = not args.no_video

    replan_freq = args.replan_freq if args.replan_freq is not None else max(1, horizon // 2)

    video_path = args.video_path or os.path.join(
        "results/singlearm_placewipe_e2e_shoulder",
        f"singlearm_placewipe_seed-{args.seed}_subtask-{args.subtask}_mode{args.mode}.mp4",
    )

    player.run(
        seed=args.seed, mode=args.mode,
        max_steps=args.max_steps, replan_freq=replan_freq,
        guidance_w=args.guidance_w, sample_N=args.sample_N,
        gripper_threshold=args.gripper_threshold,
        warm_start_skip=args.warm_start_skip,
        video_path=video_path,
        use_warm_start=not args.no_warm_start,
    )


if __name__ == "__main__":
    main()

"""Mixin for overlaying planned 3D trajectories on rendered camera frames.

Usage:
    1. Add TrajectoryVizMixin to your player class's bases:
           class MyPlayer(TrajectoryVizMixin):
    2. Call `self._init_traj_viz()` at the end of __init__.
    3. After denormalizing a trajectory, call:
           self._set_plan(traj, eef_world_pos, robot_base_name, plan_key="default")
       For two-arm setups, use different plan_key per arm (e.g. "robot0", "robot1").
    4. Update the step index during execution:
           self._set_plan_step(step_idx, plan_key="default")
    5. After rendering a camera frame (np.flipud already applied), call:
           frame = self._draw_trajectory_on_frame(frame, camera_name, h, w)

Requires: self.env.sim (MuJoCo sim), cv2, numpy.
"""

import cv2
import numpy as np


class TrajectoryVizMixin:
    """Mixin that draws planned EEF trajectories onto rendered camera frames."""

    def _init_traj_viz(self):
        self._plans = {}  # plan_key -> {positions_world, current_step}

    def _set_plan(self, traj, eef_world_pos, robot_base_name="robot0_base", plan_key="default"):
        """Store a planned trajectory, transformed to world frame and anchored at the EEF.

        Args:
            traj: (H, >=3) array of planned actions (positions in robot base frame).
            eef_world_pos: (3,) current EEF position in world frame.
            robot_base_name: MuJoCo body name for the robot base.
            plan_key: identifier for this plan (use different keys for multi-arm).
        """
        base_id = self.env.sim.model.body_name2id(robot_base_name)
        base_pos = self.env.sim.data.body_xpos[base_id]
        base_rot = self.env.sim.data.body_xmat[base_id].reshape(3, 3)
        positions = (base_rot @ traj[:, :3].T).T + base_pos
        # Anchor at actual EEF world position
        positions += np.asarray(eef_world_pos, dtype=np.float64) - positions[0]
        self._plans[plan_key] = {"positions_world": positions, "current_step": 0}

    def _set_plan_step(self, step_idx, plan_key="default"):
        if plan_key in self._plans:
            self._plans[plan_key]["current_step"] = step_idx

    def _project_to_pixels(self, points_3d, camera_name, render_h, render_w):
        """Project 3D world points to 2D pixel coordinates for a MuJoCo camera."""
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        cam_pos = self.env.sim.data.cam_xpos[cam_id]
        cam_rot = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        fovy = np.radians(self.env.sim.model.cam_fovy[cam_id])
        f = (render_h / 2.0) / np.tan(fovy / 2.0)

        pixels = []
        for p in points_3d:
            p_cam = cam_rot.T @ (p - cam_pos)
            if p_cam[2] >= -1e-3:
                pixels.append(None)
                continue
            u = -f * p_cam[0] / p_cam[2] + render_w / 2.0
            v = -f * p_cam[1] / p_cam[2] + render_h / 2.0
            v = render_h - 1 - v  # flipud correction
            pixels.append((int(round(u)), int(round(v))))
        return pixels

    def _draw_trajectory_on_frame(self, frame, camera_name, render_h, render_w):
        """Draw all stored planned trajectories on a rendered frame."""
        if not self._plans:
            return frame
        frame = frame.copy()
        for plan in self._plans.values():
            positions = plan["positions_world"]
            current_step = plan["current_step"]
            pixels = self._project_to_pixels(positions, camera_name, render_h, render_w)
            n = len(pixels)
            for i in range(n):
                if pixels[i] is None:
                    continue
                t = i / max(n - 1, 1)
                color = (int(255 * t), int(255 * (1 - t)), 0)
                radius = 3 if render_h >= 512 else 2
                if i > 0 and pixels[i - 1] is not None:
                    cv2.line(frame, pixels[i - 1], pixels[i], color, 1, cv2.LINE_AA)
                cv2.circle(frame, pixels[i], radius, color, -1, cv2.LINE_AA)
            if 0 <= current_step < n and pixels[current_step] is not None:
                big_r = 6 if render_h >= 512 else 4
                cv2.circle(frame, pixels[current_step], big_r, (255, 255, 0), 2, cv2.LINE_AA)
        return frame

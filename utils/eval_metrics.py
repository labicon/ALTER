"""Evaluation metrics mixin for Player classes.

Mix ``MetricsCollectorMixin`` into any Player to get automatic per-step data
collection and summary metric computation.  Three hooks:

    _metrics_on_reset(obs)   — call after env.reset()
    _metrics_on_step(obs, action)  — call after env.step()
    _metrics_on_done() -> dict     — call when episode ends

The mixin auto-detects single-arm vs two-arm and cube vs pot from the env.
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

# Half-length of the cube along its long (X) axis — used to compute grasp endpoints.
_CUBE_HALF_X = 0.055


def _quat_to_tilt_deg(quat_xyzw):
    """Return tilt angle (degrees) of the object's local Z-axis from world Z."""
    rotm = R.from_quat(quat_xyzw).as_matrix()
    local_z = rotm @ np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(local_z, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class MetricsCollectorMixin:
    """Mix into any Player class to collect evaluation metrics.

    Expects ``self.env`` to be the robosuite environment instance.
    """

    # ------------------------------------------------------------------ #
    # Hook: after env.reset()
    # ------------------------------------------------------------------ #

    def _metrics_on_reset(self, obs):
        # Detect env type
        self._is_pot = hasattr(self.env, "pot")
        self._is_twoarm = "robot1_eef_pos" in obs

        # Object position key
        if self._is_pot:
            self._obj_pos_key = "pot_pos"
            self._obj_quat_key = "pot_quat"
        else:
            self._obj_pos_key = "target_cube_pos"
            self._obj_quat_key = "target_cube_quat"

        # Initial object position
        self._initial_obj_pos = np.array(obs[self._obj_pos_key], dtype=np.float64).copy()

        # Drop detection reference: use landing pad top surface if available,
        # otherwise fall back to table height.  The object starts on a pedestal
        # which is higher than the pad, so comparing against initial height
        # would flag every successful transport as a "drop".
        has_pad = (
            hasattr(self.env, "landing_pad_body_id")
            and self.env.landing_pad_body_id is not None
        )
        self._has_pad = has_pad
        if has_pad:
            pad_pos = self.env.sim.data.body_xpos[self.env.landing_pad_body_id].copy()
            self._pad_pos = np.asarray(pad_pos, dtype=np.float64).copy()
            pad_half_z = float(self.env.landing_pad.size[2]) if hasattr(self.env, "landing_pad") else 0.02
            self._drop_floor_z = float(pad_pos[2]) + pad_half_z
        else:
            self._pad_pos = None
            # No landing pad — use table height as the floor
            table_top_id = self.env.sim.model.site_name2id("table_top")
            self._drop_floor_z = float(self.env.sim.data.site_xpos[table_top_id][2])

        # Buffers
        self._actions = []
        self._obj_heights = []
        self._obj_displacements = []
        self._tilts = []                    # degrees, two-arm only
        self._arm0_endpoint_dists = []      # two-arm only
        self._arm1_endpoint_dists = []      # two-arm only
        self._arm0_endpoint_dists_closed = []   # only steps where arm0 gripper closed
        self._arm1_endpoint_dists_closed = []   # only steps where arm1 gripper closed
        self._was_lifted = False            # for drop detection
        self._was_dropped = False
        self._pad_approach_min = float("inf")   # min ||obj - pad|| over episode

    # ------------------------------------------------------------------ #
    # Hook: after env.step()
    # ------------------------------------------------------------------ #

    def _metrics_on_step(self, obs, action):
        action = np.asarray(action, dtype=np.float64)
        self._actions.append(action.copy())

        obj_pos = np.array(obs[self._obj_pos_key], dtype=np.float64)
        self._obj_heights.append(float(obj_pos[2]))
        self._obj_displacements.append(
            float(np.linalg.norm(obj_pos - self._initial_obj_pos))
        )

        # Drop detection: object was lifted, then fell below the landing pad surface
        lift_thresh = 0.02
        if obj_pos[2] > self._initial_obj_pos[2] + lift_thresh:
            self._was_lifted = True
        if self._was_lifted and obj_pos[2] < self._drop_floor_z:
            self._was_dropped = True

        if self._has_pad and self._pad_pos is not None:
            d_pad = float(np.linalg.norm(obj_pos - self._pad_pos))
            if d_pad < self._pad_approach_min:
                self._pad_approach_min = d_pad

        # Two-arm metrics
        if self._is_twoarm:
            # Object tilt
            obj_quat = np.array(obs[self._obj_quat_key], dtype=np.float64)
            self._tilts.append(_quat_to_tilt_deg(obj_quat))

            # Arm-to-endpoint distances
            eef0 = np.array(obs["robot0_eef_pos"], dtype=np.float64)
            eef1 = np.array(obs["robot1_eef_pos"], dtype=np.float64)

            if self._is_pot:
                # Pot handles are directly in obs
                target0 = np.array(obs["handle0_xpos"], dtype=np.float64)
                target1 = np.array(obs["handle1_xpos"], dtype=np.float64)
            else:
                # Cube: grasp endpoints = center +/- half-X along world X
                center = obj_pos
                target0 = center + np.array([-_CUBE_HALF_X, 0.0, 0.0])
                target1 = center + np.array([_CUBE_HALF_X, 0.0, 0.0])

            d0 = float(np.linalg.norm(eef0 - target0))
            d1 = float(np.linalg.norm(eef1 - target1))
            self._arm0_endpoint_dists.append(d0)
            self._arm1_endpoint_dists.append(d1)

            # Gripper-closed-only distances. Action layout (two-arm shoulder):
            # 7 dims per arm, gripper at index 6; full action length 14.
            # A closed gripper is encoded as 1.0 after postprocessing.
            if action.shape[0] >= 14:
                if action[6] >= 0.5:
                    self._arm0_endpoint_dists_closed.append(d0)
                if action[13] >= 0.5:
                    self._arm1_endpoint_dists_closed.append(d1)

    # ------------------------------------------------------------------ #
    # Hook: after episode ends
    # ------------------------------------------------------------------ #

    def _metrics_on_done(self):
        metrics = {}
        actions = np.array(self._actions)
        n = len(actions)

        # Action smoothness
        if n > 1:
            diffs = np.linalg.norm(np.diff(actions, axis=0), axis=1)
            metrics["action_smoothness"] = float(np.mean(diffs))
        else:
            metrics["action_smoothness"] = 0.0

        # Max object displacement from initial position
        metrics["max_object_displacement"] = float(max(self._obj_displacements)) if self._obj_displacements else 0.0

        # Two-arm only metrics
        if self._is_twoarm:
            metrics["object_max_tilt_deg"] = float(max(self._tilts)) if self._tilts else 0.0
            metrics["object_dropped"] = self._was_dropped

            arm0_mean = float(np.mean(self._arm0_endpoint_dists)) if self._arm0_endpoint_dists else 0.0
            arm1_mean = float(np.mean(self._arm1_endpoint_dists)) if self._arm1_endpoint_dists else 0.0
            metrics["arm0_to_endpoint_mean"] = arm0_mean
            metrics["arm1_to_endpoint_mean"] = arm1_mean
            metrics["arm_endpoint_distance_asymmetry"] = abs(arm0_mean - arm1_mean)

            metrics["arm0_to_endpoint_max_gripper_closed"] = (
                float(max(self._arm0_endpoint_dists_closed))
                if self._arm0_endpoint_dists_closed else 0.0
            )
            metrics["arm1_to_endpoint_max_gripper_closed"] = (
                float(max(self._arm1_endpoint_dists_closed))
                if self._arm1_endpoint_dists_closed else 0.0
            )
            metrics["arm0_to_endpoint_mean_gripper_closed"] = (
                float(np.mean(self._arm0_endpoint_dists_closed))
                if self._arm0_endpoint_dists_closed else 0.0
            )
            metrics["arm1_to_endpoint_mean_gripper_closed"] = (
                float(np.mean(self._arm1_endpoint_dists_closed))
                if self._arm1_endpoint_dists_closed else 0.0
            )

        if self._has_pad and math.isfinite(self._pad_approach_min):
            metrics["pad_approach_min"] = float(self._pad_approach_min)

        # Final object-to-landing-pad distance (works for both single-arm and two-arm)
        has_pad = (
            hasattr(self.env, "landing_pad_body_id")
            and self.env.landing_pad_body_id is not None
        )
        if has_pad and self._actions:
            pad_pos = self.env.sim.data.body_xpos[self.env.landing_pad_body_id].copy()
            if self._is_pot:
                obj_body_id = self.env.sim.model.body_name2id(self.env.pot.root_body)
            else:
                obj_body_id = self.env.sim.model.body_name2id(self.env.target_cube.root_body)
            obj_pos = self.env.sim.data.body_xpos[obj_body_id].copy()
            metrics["final_object_to_pad_distance"] = float(np.linalg.norm(obj_pos - pad_pos))

        metrics["episode_length"] = n
        return metrics


class PlaceWipeMetricsCollectorMixin:
    """Metric collector for the two-arm place+wipe task.

    Single source of truth for placewipe success and observables. Reads
    everything from ``self.env`` (sim state + ``_wipe_coverage``) so the same
    numbers come out whether the rollout is driven by the coord head, a
    LoRA adapter, or a from-scratch policy.

    Success gate (all four must hold):
      - ``cube_returned``      : final cube→return_pad < ``CUBE_RETURNED_THRESH``
      - ``sponge_home``        : final sponge→sponge_pad < ``SPONGE_HOME_THRESH``
      - ``cube_visited_dest``  : min cube→dest_pad over trajectory <
                                 ``CUBE_VISITED_DEST_THRESH`` (guards against
                                 trajectories that skip the holding step and
                                 wipe "from the wrong side").
      - ``coverage_ok``        : ``wipe_coverage_final`` ≥
                                 ``WIPE_COVERAGE_THRESH``. Coverage strictly
                                 dominates a sponge-Y-range gate — a policy
                                 can only clean ≥ X% of the dirt grid by
                                 actually sweeping over it.

    ``sponge_sweep_y`` is still reported as a raw observable (useful for
    diagnosing "sponge moved a lot but in the wrong place" failure modes),
    but no longer gates ``task_success``.
    """

    # Success thresholds — also the de facto contract for downstream
    # comparisons. Bump them here and every script picks up the change.
    CUBE_RETURNED_THRESH = 0.08
    SPONGE_HOME_THRESH = 0.10
    CUBE_VISITED_DEST_THRESH = 0.08
    WIPE_COVERAGE_THRESH = 0.8

    def _metrics_on_reset(self, obs):
        env = self.env
        self._pw_cube_body_id = env.handle_cube_body_id
        self._pw_sponge_body_id = env.sponge_body_id
        self._pw_destination_pad_site_id = env.destination_pad_site_id
        self._pw_return_pad_site_id = env.return_pad_site_id
        self._pw_sponge_pad_site_id = env.sponge_pad_site_id
        self._pw_cube_geom_ids = set(self._pw_geoms_for_body(self._pw_cube_body_id))
        self._pw_sponge_geom_ids = set(self._pw_geoms_for_body(self._pw_sponge_body_id))

        self._pw_actions = []
        self._pw_cube_to_dest = []
        self._pw_cube_to_return = []
        self._pw_sponge_y = []
        self._pw_collided = False
        self._pw_collision_steps = 0
        self._pw_coverage_curve = []

    def _metrics_on_step(self, obs, action):
        self._pw_actions.append(np.asarray(action, dtype=np.float64).copy())

        sim = self.env.sim
        cube_xy = sim.data.body_xpos[self._pw_cube_body_id][:2]
        dest_xy = sim.data.site_xpos[self._pw_destination_pad_site_id][:2]
        return_xy = sim.data.site_xpos[self._pw_return_pad_site_id][:2]
        sponge_y = float(sim.data.body_xpos[self._pw_sponge_body_id][1])
        self._pw_cube_to_dest.append(float(np.linalg.norm(cube_xy - dest_xy)))
        self._pw_cube_to_return.append(float(np.linalg.norm(cube_xy - return_xy)))
        self._pw_sponge_y.append(sponge_y)

        self._pw_coverage_curve.append(float(self.env._wipe_coverage))

        ncon = int(sim.data.ncon)
        cube_set = self._pw_cube_geom_ids
        sponge_set = self._pw_sponge_geom_ids
        for i in range(ncon):
            c = sim.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 in cube_set and g2 in sponge_set) or (g2 in cube_set and g1 in sponge_set):
                self._pw_collided = True
                self._pw_collision_steps += 1
                break

    def _metrics_on_done(self):
        sim = self.env.sim
        cube_xy = sim.data.body_xpos[self._pw_cube_body_id][:2].copy()
        sponge_xy = sim.data.body_xpos[self._pw_sponge_body_id][:2].copy()
        return_xy = sim.data.site_xpos[self._pw_return_pad_site_id][:2].copy()
        sponge_pad_xy = sim.data.site_xpos[self._pw_sponge_pad_site_id][:2].copy()

        cube_to_dest_min = float(min(self._pw_cube_to_dest)) if self._pw_cube_to_dest else float("nan")
        cube_to_return_min = float(min(self._pw_cube_to_return)) if self._pw_cube_to_return else float("nan")
        cube_to_return_final = float(np.linalg.norm(cube_xy - return_xy))
        sponge_to_pad_final = float(np.linalg.norm(sponge_xy - sponge_pad_xy))
        if self._pw_sponge_y:
            sponge_sweep_y = float(max(self._pw_sponge_y) - min(self._pw_sponge_y))
        else:
            sponge_sweep_y = 0.0
        coverage = float(self.env._wipe_coverage)
        cube_returned = cube_to_return_final < self.CUBE_RETURNED_THRESH
        sponge_home = sponge_to_pad_final < self.SPONGE_HOME_THRESH
        cube_visited_dest = cube_to_dest_min < self.CUBE_VISITED_DEST_THRESH
        coverage_ok = coverage >= self.WIPE_COVERAGE_THRESH
        task_success = bool(
            cube_returned and sponge_home
            and cube_visited_dest and coverage_ok
        )
        generator_length_ok = len(self._pw_actions) <= 1100
        generator_acceptance_success = bool(task_success and generator_length_ok)

        return {
            "cube_to_destination_min": cube_to_dest_min,
            "cube_to_return_min": cube_to_return_min,
            "cube_to_return_final": cube_to_return_final,
            "sponge_to_pad_final": sponge_to_pad_final,
            "sponge_sweep_y": sponge_sweep_y,
            "wipe_coverage_final": coverage,
            "cube_returned": bool(cube_returned),
            "sponge_home": bool(sponge_home),
            "cube_visited_dest": bool(cube_visited_dest),
            "coverage_ok": bool(coverage_ok),
            "task_success": task_success,
            "placewipe_success": task_success,
            "generator_length_ok": bool(generator_length_ok),
            "generator_acceptance_success": generator_acceptance_success,
            "sponge_cube_collision": bool(self._pw_collided),
            "sponge_cube_collision_steps": int(self._pw_collision_steps),
            "episode_length": len(self._pw_actions),
        }

    def _pw_geoms_for_body(self, body_id):
        model = self.env.sim.model
        return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id]


class SingleArmPlaceWipeMetricsCollectorMixin:
    """Metric collector for the single-arm placewipe task.

    Single-arm runs one subtask at a time (``env.task in {"place_return",
    "wipe"}``), and the relevant obs/sites differ between the two. This
    mixin computes the per-subtask metrics + success gate so the sampler
    never has to reimplement them.

    place_return:
      - direction (= ``env._mode // 2``):
          0 = forward (center → dest_pad), success at dest_pad
          1 = backward (dest_pad → return_pad), success at return_pad
      - ``cube_to_target_final`` < per-direction threshold = success
        (forward uses the tighter ``CUBE_ON_DEST_THRESH`` because the dest
        pad is the intermediate hold; backward uses ``CUBE_RETURNED_THRESH``).

    wipe:
      - ``wipe_coverage_final`` ≥ ``WIPE_COVERAGE_THRESH`` AND
        ``sponge_home`` (final sponge→sponge_pad < ``SPONGE_HOME_THRESH``).
      - ``sponge_sweep_y`` is reported but not gated, matching the
        existing single-arm wipe criterion (coverage already implies it).
    """

    CUBE_ON_DEST_THRESH = 0.06
    CUBE_RETURNED_THRESH = 0.08
    SPONGE_HOME_THRESH = 0.10
    WIPE_COVERAGE_THRESH = 0.8

    def _metrics_on_reset(self, obs):
        env = self.env
        self._sa_task = env.task
        self._sa_actions = []

        if self._sa_task == "place_return":
            self._sa_cube_body_id = env.handle_cube_body_id
            self._sa_destination_pad_site_id = env.destination_pad_site_id
            self._sa_return_pad_site_id = env.return_pad_site_id
            self._sa_cube_to_dest = []
            self._sa_cube_to_return = []
            self._sa_direction = int(env._mode) // 2  # 0=forward, 1=backward
        else:  # wipe
            self._sa_sponge_body_id = env.sponge_body_id
            self._sa_sponge_pad_site_id = env.sponge_pad_site_id
            self._sa_sponge_y = []

    def _metrics_on_step(self, obs, action):
        self._sa_actions.append(np.asarray(action, dtype=np.float64).copy())
        sim = self.env.sim
        if self._sa_task == "place_return":
            cube_xy = sim.data.body_xpos[self._sa_cube_body_id][:2]
            dest_xy = sim.data.site_xpos[self._sa_destination_pad_site_id][:2]
            return_xy = sim.data.site_xpos[self._sa_return_pad_site_id][:2]
            self._sa_cube_to_dest.append(float(np.linalg.norm(cube_xy - dest_xy)))
            self._sa_cube_to_return.append(float(np.linalg.norm(cube_xy - return_xy)))
        else:
            self._sa_sponge_y.append(
                float(sim.data.body_xpos[self._sa_sponge_body_id][1])
            )

    def _metrics_on_done(self):
        sim = self.env.sim
        out = {
            "subtask": self._sa_task,
            "episode_length": len(self._sa_actions),
        }

        if self._sa_task == "place_return":
            cube_xy = sim.data.body_xpos[self._sa_cube_body_id][:2].copy()
            dest_xy = sim.data.site_xpos[self._sa_destination_pad_site_id][:2].copy()
            return_xy = sim.data.site_xpos[self._sa_return_pad_site_id][:2].copy()
            cube_to_dest_final = float(np.linalg.norm(cube_xy - dest_xy))
            cube_to_return_final = float(np.linalg.norm(cube_xy - return_xy))
            cube_to_dest_min = float(min(self._sa_cube_to_dest)) if self._sa_cube_to_dest else float("nan")
            cube_to_return_min = float(min(self._sa_cube_to_return)) if self._sa_cube_to_return else float("nan")

            if self._sa_direction == 0:  # forward: success at dest
                target_dist = cube_to_dest_final
                thresh = self.CUBE_ON_DEST_THRESH
            else:  # backward: success at return
                target_dist = cube_to_return_final
                thresh = self.CUBE_RETURNED_THRESH
            task_success = bool(target_dist < thresh)

            out.update({
                "direction": self._sa_direction,
                "cube_to_destination_final": cube_to_dest_final,
                "cube_to_return_final": cube_to_return_final,
                "cube_to_destination_min": cube_to_dest_min,
                "cube_to_return_min": cube_to_return_min,
                "cube_to_target_final": target_dist,
                "task_success": task_success,
                "placewipe_success": task_success,
            })
            return out

        # wipe
        sponge_xy = sim.data.body_xpos[self._sa_sponge_body_id][:2].copy()
        sponge_pad_xy = sim.data.site_xpos[self._sa_sponge_pad_site_id][:2].copy()
        sponge_to_pad_final = float(np.linalg.norm(sponge_xy - sponge_pad_xy))
        if self._sa_sponge_y:
            sponge_sweep_y = float(max(self._sa_sponge_y) - min(self._sa_sponge_y))
        else:
            sponge_sweep_y = 0.0
        coverage = float(getattr(self.env, "_wipe_coverage", 0.0))
        sponge_home = sponge_to_pad_final < self.SPONGE_HOME_THRESH
        coverage_ok = coverage >= self.WIPE_COVERAGE_THRESH
        task_success = bool(sponge_home and coverage_ok)
        out.update({
            "sponge_to_pad_final": sponge_to_pad_final,
            "sponge_sweep_y": sponge_sweep_y,
            "wipe_coverage_final": coverage,
            "sponge_home": bool(sponge_home),
            "coverage_ok": bool(coverage_ok),
            "task_success": task_success,
            "placewipe_success": task_success,
        })
        return out


class ThreeArmWipeMetricsCollectorMixin:
    """Metric collector for the three-arm wipe task.

    Mirrors ``PlaceWipeMetricsCollectorMixin`` (cube→tray naming) plus
    failure-mode metrics specific to two-handed tray transport:

      Geometry / coverage (analogues of placewipe):
      - ``tray_to_return_final``: final ‖tray_xy − return_pad_xy‖.
      - ``sponge_to_pad_final``:  final ‖sponge_xy − sponge_pad_xy‖.
      - ``wipe_coverage_final``:  env-tracked dirt-grid coverage.
      - ``sponge_tray_collision`` (bool) + ``sponge_tray_collision_steps``
        (int): same MuJoCo contact-walk pattern as placewipe's
        ``sponge_cube_collision``, with the cube geom set replaced by all
        ``TrayObject`` geoms.
      - ``episode_length``: total steps executed.

      Tray-drop detector (per-arm sphere-vs-OBB rigid body check; only
      active when grippers are commanded closed, so approach + set-down
      phases auto-suppress):
      - Each closed gripper defines a virtual grip volume — a sphere of
        radius ``DISK_RADIUS_M`` centered at the midpoint of its two
        fingerpad geoms.
      - The handle bar is modeled as an oriented box with the actual geom
        half-extents ``BAR_HALF_EXTENTS_M = (hs_half+ht, ht, ht) ≈
        (0.054, 0.004, 0.004)`` from ``wipe_scene.py``, oriented by the
        tray's body quaternion.
      - For each carrier arm with gripper command ``≥ GRIP_CLOSED_ACTION_THRESH``,
        compute the distance from the sphere center to the nearest point
        on the bar OBB (clamp the disk center into the bar's local box
        extents, then take ‖offset − clamped‖). If that distance exceeds
        ``DISK_RADIUS_M``, the bar's surface is outside the grip volume →
        tray dropped (per-arm latch).
      - ``tray_dropped`` (bool): OR over both arms.
      - ``tray_dropped_robot0`` / ``tray_dropped_robot1`` (bool): per-arm
        latch — useful for diagnosing which carrier slipped first.
      - ``tray_drop_step`` (int, ``-1`` if no drop): earliest step either
        arm latched.
      - ``tray_drop_step_robot0`` / ``tray_drop_step_robot1`` (int, ``-1``
        if that arm never slipped): per-arm drop step.

      Coordination quality:
      - ``tray_tilt_max_deg``: max tilt of the tray's local +Z axis from
        world +Z (uses ``_quat_to_tilt_deg``).
      - ``handle_separation_drift_max``: max ‖handle_dist(t) − handle_dist(0)‖
        over the episode — direct measure of the carrier arms drifting
        apart, the failure mode the recent ``grip_compress_dx`` waypoint fix
        was meant to prevent.
    """

    # Sphere-vs-OBB drop detector tunables.
    GRIP_CLOSED_ACTION_THRESH = 0.5
    DISK_RADIUS_M = 0.05

    # Match envs.arm.data_gen.generate_threearm_wipe.evaluate_rollout,
    # except sampling task_success is not constrained by rollout length.
    TRAY_PAD_VISIT_THRESH = 0.10
    TRAY_RETURN_THRESH = 0.08
    SPONGE_HOME_THRESH = 0.10
    WIPE_COVERAGE_THRESH = 0.80
    MAX_TRAY_TILT_DEG = 20.0
    MAX_HANDLE_DRIFT = 0.07

    # Bar OBB half-extents in tray local frame (from wipe_scene.py:201:
    # geom_sizes=(hs_half+ht, ht, ht), with hs_half=0.05, ht=0.004).
    BAR_HALF_EXTENTS_M = (0.054, 0.004, 0.004)

    def _metrics_on_reset(self, obs):
        env = self.env
        sim = env.sim
        model = sim.model

        # --- Body / site IDs ---
        self._taw_tray_body_id = int(env.tray_body_id)
        self._taw_sponge_body_id = int(env.sponge_body_id)
        self._taw_tray_pad_site_id = int(env.tray_pad_site_id)
        self._taw_sponge_pad_site_id = int(env.sponge_pad_site_id)
        self._taw_return_pad_site_id = int(env.return_pad_site_id)

        # --- Tray + sponge geoms (for sponge-tray contact, mirrors placewipe) ---
        self._taw_tray_geom_ids = set(self._taw_geoms_for_body(self._taw_tray_body_id))
        self._taw_sponge_geom_ids = set(self._taw_geoms_for_body(self._taw_sponge_body_id))

        # --- Fingerpad geoms per carrier arm (the closed-gripper "disk" is
        #     centered at the midpoint of these two pads). Robotiq85 names:
        #     gripper{i}_*_left_fingerpad_collision / *_right_fingerpad_collision.
        self._taw_fingerpad_geom_ids = []
        for i in (0, 1):
            left_id = right_id = None
            for name in model.geom_names:
                if not name or not name.startswith(f"gripper{i}_"):
                    continue
                if name.endswith("left_fingerpad_collision"):
                    left_id = int(model.geom_name2id(name))
                elif name.endswith("right_fingerpad_collision"):
                    right_id = int(model.geom_name2id(name))
            if left_id is None or right_id is None:
                raise RuntimeError(
                    f"Could not resolve fingerpad geoms for gripper{i} — "
                    "gripper geom naming may have changed."
                )
            self._taw_fingerpad_geom_ids.append((left_id, right_id))

        # --- Per-step buffers / state ---
        self._taw_actions = []
        self._taw_dropped_per_arm = [False, False]   # latched per carrier arm
        self._taw_drop_step_per_arm = [-1, -1]
        self._taw_max_tilt_deg = 0.0
        self._taw_tray_pad_dist_min = float(np.linalg.norm(
            sim.data.body_xpos[self._taw_tray_body_id][:2]
            - sim.data.site_xpos[self._taw_tray_pad_site_id][:2]
        ))

        # Initial inter-handle separation for drift metric (None if env doesn't
        # expose handle sites — defensive for backward compat).
        self._taw_max_handle_sep_drift = 0.0
        if hasattr(env, "handle_left_site_id") and hasattr(env, "handle_right_site_id"):
            self._taw_initial_handle_sep = float(np.linalg.norm(
                sim.data.site_xpos[env.handle_left_site_id][:2]
                - sim.data.site_xpos[env.handle_right_site_id][:2]
            ))
        else:
            self._taw_initial_handle_sep = None

        # Sponge-tray collision (mirror placewipe pattern)
        self._taw_sponge_tray_collided = False
        self._taw_sponge_tray_collision_steps = 0

    def _metrics_on_step(self, obs, action):
        env = self.env
        sim = env.sim
        action_arr = np.asarray(action, dtype=np.float64)
        self._taw_actions.append(action_arr.copy())
        step_idx = len(self._taw_actions)  # 1-indexed

        tray_pad_dist = float(np.linalg.norm(
            sim.data.body_xpos[self._taw_tray_body_id][:2]
            - sim.data.site_xpos[self._taw_tray_pad_site_id][:2]
        ))
        if tray_pad_dist < self._taw_tray_pad_dist_min:
            self._taw_tray_pad_dist_min = tray_pad_dist

        # Sponge-tray contact (single contact-list walk, mirrors placewipe).
        ncon = int(sim.data.ncon)
        sponge_set = self._taw_sponge_geom_ids
        tray_set = self._taw_tray_geom_ids
        sponge_tray_contact = False
        for i in range(ncon):
            c = sim.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if ((g1 in tray_set and g2 in sponge_set)
                    or (g2 in tray_set and g1 in sponge_set)):
                sponge_tray_contact = True
                break
        if sponge_tray_contact:
            self._taw_sponge_tray_collided = True
            self._taw_sponge_tray_collision_steps += 1

        # Sphere-vs-OBB tray-drop detector. Only checks when the gripper is
        # commanded closed — approach + set-down phases (grippers open) are
        # auto-suppressed. For each carrier arm: the closed gripper is a
        # sphere of radius DISK_RADIUS_M at the fingerpad midpoint; the bar
        # is an OBB (extents BAR_HALF_EXTENTS_M, oriented by the tray) at
        # the handle site. Distance from sphere center to the nearest point
        # on the OBB > radius → bar slipped out of the grip.
        # Per-arm latch: if one arm slips, keep checking the other so both
        # drop steps are recorded for debugging.
        if (not all(self._taw_dropped_per_arm)
                and action_arr.shape[0] >= 14):
            # Tray rotation matrix (local→world). Bar's local axes are the
            # tray's local axes (geom_quats=(1,0,0,0) in wipe_scene.py).
            wxyz = sim.data.body_xquat[self._taw_tray_body_id]
            tray_R = R.from_quat(
                [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
            ).as_matrix()
            bar_half_extents = np.asarray(self.BAR_HALF_EXTENTS_M)

            for arm, action_idx, handle_site_id in (
                (0, 6, env.handle_left_site_id),
                (1, 13, env.handle_right_site_id),
            ):
                if self._taw_dropped_per_arm[arm]:
                    continue  # already latched
                if action_arr[action_idx] < self.GRIP_CLOSED_ACTION_THRESH:
                    continue  # gripper open — not held, not a "drop"

                left_id, right_id = self._taw_fingerpad_geom_ids[arm]
                disk_center = 0.5 * (
                    sim.data.geom_xpos[left_id] + sim.data.geom_xpos[right_id]
                )

                # Sphere-vs-OBB nearest-point distance: transform sphere
                # center into bar local frame, clamp to box extents to find
                # nearest point on the bar surface, then take the offset.
                bar_center = sim.data.site_xpos[handle_site_id]
                offset_local = tray_R.T @ (disk_center - bar_center)
                clamped_local = np.clip(
                    offset_local, -bar_half_extents, bar_half_extents
                )
                dist_to_bar = float(np.linalg.norm(offset_local - clamped_local))
                if dist_to_bar > self.DISK_RADIUS_M:
                    self._taw_dropped_per_arm[arm] = True
                    self._taw_drop_step_per_arm[arm] = step_idx

        # Tilt — body_xquat is wxyz in MuJoCo; reorder to xyzw for the helper.
        wxyz = sim.data.body_xquat[self._taw_tray_body_id]
        xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        tilt = _quat_to_tilt_deg(xyzw)
        if tilt > self._taw_max_tilt_deg:
            self._taw_max_tilt_deg = tilt

        # Inter-handle separation drift (carrier-arm coordination quality).
        if self._taw_initial_handle_sep is not None:
            sep = float(np.linalg.norm(
                sim.data.site_xpos[env.handle_left_site_id][:2]
                - sim.data.site_xpos[env.handle_right_site_id][:2]
            ))
            drift = abs(sep - self._taw_initial_handle_sep)
            if drift > self._taw_max_handle_sep_drift:
                self._taw_max_handle_sep_drift = drift

    def _metrics_on_done(self):
        env = self.env
        sim = env.sim

        tray_xy = sim.data.body_xpos[self._taw_tray_body_id][:2].copy()
        sponge_xy = sim.data.body_xpos[self._taw_sponge_body_id][:2].copy()
        return_xy = sim.data.site_xpos[self._taw_return_pad_site_id][:2].copy()
        sponge_pad_xy = sim.data.site_xpos[self._taw_sponge_pad_site_id][:2].copy()
        tray_to_return_final = float(np.linalg.norm(tray_xy - return_xy))
        sponge_to_pad_final = float(np.linalg.norm(sponge_xy - sponge_pad_xy))
        coverage = float(env._wipe_coverage)
        tray_visited_waiting_pad = (
            self._taw_tray_pad_dist_min < self.TRAY_PAD_VISIT_THRESH
        )
        tray_returned = tray_to_return_final < self.TRAY_RETURN_THRESH
        sponge_home = sponge_to_pad_final < self.SPONGE_HOME_THRESH
        coverage_ok = coverage >= self.WIPE_COVERAGE_THRESH
        tilt_ok = self._taw_max_tilt_deg <= self.MAX_TRAY_TILT_DEG
        handle_drift_ok = self._taw_max_handle_sep_drift <= self.MAX_HANDLE_DRIFT
        motion_quality_ok = bool(tilt_ok and handle_drift_ok)
        task_success = bool(
            tray_visited_waiting_pad
            and tray_returned
            and sponge_home
            and coverage_ok
            and motion_quality_ok
        )
        generator_length_ok = len(self._taw_actions) <= 1600
        generator_acceptance_success = bool(task_success and generator_length_ok)

        drop0 = self._taw_dropped_per_arm[0]
        drop1 = self._taw_dropped_per_arm[1]
        drop_steps = [s for s in self._taw_drop_step_per_arm if s > 0]
        first_drop_step = int(min(drop_steps)) if drop_steps else -1

        return {
            "tray_to_return_final": tray_to_return_final,
            "sponge_to_pad_final": sponge_to_pad_final,
            "tray_to_waiting_pad_min": float(self._taw_tray_pad_dist_min),
            "wipe_coverage_final": coverage,
            "tray_visited_waiting_pad": bool(tray_visited_waiting_pad),
            "tray_returned": bool(tray_returned),
            "sponge_home": bool(sponge_home),
            "coverage_ok": bool(coverage_ok),
            "tray_tilt_ok": bool(tilt_ok),
            "handle_drift_ok": bool(handle_drift_ok),
            "motion_quality_ok": motion_quality_ok,
            "task_success": task_success,
            "wipe_success": task_success,
            "generator_length_ok": bool(generator_length_ok),
            "generator_acceptance_success": generator_acceptance_success,
            "sponge_tray_collision": bool(self._taw_sponge_tray_collided),
            "sponge_tray_collision_steps": int(self._taw_sponge_tray_collision_steps),
            "tray_dropped": bool(drop0 or drop1),
            "tray_dropped_robot0": bool(drop0),
            "tray_dropped_robot1": bool(drop1),
            "tray_drop_step": first_drop_step,
            "tray_drop_step_robot0": int(self._taw_drop_step_per_arm[0]),
            "tray_drop_step_robot1": int(self._taw_drop_step_per_arm[1]),
            "tray_tilt_max_deg": float(self._taw_max_tilt_deg),
            "handle_separation_drift_max": float(self._taw_max_handle_sep_drift),
            "episode_length": len(self._taw_actions),
        }

    def _taw_geoms_for_body(self, body_id):
        model = self.env.sim.model
        return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id]


# ---------------------------------------------------------------------------
# Data-gen acceptance helpers
#
# Mirror the per-task ``task_success`` of the corresponding mixin, but read
# the saved rollout dict + live ``env._wipe_coverage`` instead of running the
# per-step collector. Data-gen evaluates post-hoc (after a deterministic
# waypoint rollout finishes), so it doesn't have the live obs stream the
# mixin expects. Thresholds come from the same mixin class constants, so
# bumping them in one place updates eval AND acceptance together.
# ---------------------------------------------------------------------------

def acceptance_check_twoarm_placewipe(rollout, env, max_rollout_len=1100):
    """Acceptance gate for two-arm placewipe data-gen rollouts.

    Returns ``(ok: bool, msg: str, metrics: dict)``. ``ok`` matches
    ``PlaceWipeMetricsCollectorMixin.task_success``: cube_returned AND
    sponge_home AND cube_visited_dest AND coverage_ok.
    """
    M = PlaceWipeMetricsCollectorMixin
    obs_list = rollout["observations"]
    if not obs_list:
        return False, "no_obs", {}
    if len(obs_list) > max_rollout_len:
        return False, f"too_long ({len(obs_list)} steps)", {}

    cube_pos = np.array([o["handle_cube_pos"] for o in obs_list])
    sponge_pos = np.array([o["sponge_pos"] for o in obs_list])
    dest_pos = np.array([o["destination_pad_pos"] for o in obs_list])
    return_xy = np.asarray(obs_list[-1]["return_pad_pos"], dtype=np.float64)[:2]
    sponge_pad_xy = np.asarray(obs_list[-1]["sponge_pad_pos"], dtype=np.float64)[:2]

    cube_to_return_final = float(np.linalg.norm(cube_pos[-1, :2] - return_xy))
    sponge_to_pad_final = float(np.linalg.norm(sponge_pos[-1, :2] - sponge_pad_xy))
    cube_to_destination_min = float(
        np.min(np.linalg.norm(cube_pos[:, :2] - dest_pos[:, :2], axis=1))
    )
    coverage = float(getattr(env, "_wipe_coverage", 0.0))

    cube_returned = cube_to_return_final < M.CUBE_RETURNED_THRESH
    sponge_home = sponge_to_pad_final < M.SPONGE_HOME_THRESH
    cube_visited_dest = cube_to_destination_min < M.CUBE_VISITED_DEST_THRESH
    coverage_ok = coverage >= M.WIPE_COVERAGE_THRESH
    ok = bool(cube_returned and sponge_home and cube_visited_dest and coverage_ok)

    msg = (
        f"cube_returned={cube_returned} sponge_home={sponge_home} "
        f"cube_visited_dest={cube_visited_dest} coverage_ok={coverage_ok} "
        f"wipe_coverage_final={coverage:.3f} "
        f"cube_to_return_final={cube_to_return_final:.3f} "
        f"cube_to_destination_min={cube_to_destination_min:.3f}"
    )
    metrics = {
        "cube_to_return_final": cube_to_return_final,
        "sponge_to_pad_final": sponge_to_pad_final,
        "cube_to_destination_min": cube_to_destination_min,
        "wipe_coverage_final": coverage,
        "cube_returned": bool(cube_returned),
        "sponge_home": bool(sponge_home),
        "cube_visited_dest": bool(cube_visited_dest),
        "coverage_ok": bool(coverage_ok),
        "task_success": ok,
        "episode_length": len(obs_list),
    }
    return ok, msg, metrics


def acceptance_check_singlearm_placewipe(rollout, env, max_rollout_len=1100):
    """Acceptance gate for single-arm placewipe data-gen rollouts.

    Branches on ``env.task`` like ``SingleArmPlaceWipeMetricsCollectorMixin``.
    Reads ``env._mode`` to recover the place_return direction.
    """
    M = SingleArmPlaceWipeMetricsCollectorMixin
    obs_list = rollout["observations"]
    if not obs_list:
        return False, "no_obs", {}
    if len(obs_list) > max_rollout_len:
        return False, f"too_long ({len(obs_list)} steps)", {}

    task = env.task
    if task == "place_return":
        cube_pos = np.array([o["handle_cube_pos"] for o in obs_list])
        dest_xy = np.asarray(obs_list[-1]["destination_pad_pos"], dtype=np.float64)[:2]
        return_xy = np.asarray(obs_list[-1]["return_pad_pos"], dtype=np.float64)[:2]
        final_xy = cube_pos[-1, :2]
        cube_to_destination_final = float(np.linalg.norm(final_xy - dest_xy))
        cube_to_return_final = float(np.linalg.norm(final_xy - return_xy))
        direction = int(env._mode) // 2  # 0 = forward, 1 = backward
        if direction == 0:
            cube_to_target_final = cube_to_destination_final
            thresh = M.CUBE_ON_DEST_THRESH
        else:
            cube_to_target_final = cube_to_return_final
            thresh = M.CUBE_RETURNED_THRESH
        ok = bool(cube_to_target_final < thresh)
        msg = (
            f"direction={'forward' if direction == 0 else 'backward'} "
            f"cube_to_target_final={cube_to_target_final:.3f}"
        )
        metrics = {
            "subtask": "place_return",
            "direction": direction,
            "cube_to_destination_final": cube_to_destination_final,
            "cube_to_return_final": cube_to_return_final,
            "cube_to_target_final": cube_to_target_final,
            "task_success": ok,
            "episode_length": len(obs_list),
        }
        return ok, msg, metrics

    # wipe
    sponge_pos = np.array([o["sponge_pos"] for o in obs_list])
    sponge_pad_xy = np.asarray(obs_list[-1]["sponge_pad_pos"], dtype=np.float64)[:2]
    sponge_to_pad_final = float(np.linalg.norm(sponge_pos[-1, :2] - sponge_pad_xy))
    coverage = float(getattr(env, "_wipe_coverage", 0.0))
    sponge_home = sponge_to_pad_final < M.SPONGE_HOME_THRESH
    coverage_ok = coverage >= M.WIPE_COVERAGE_THRESH
    ok = bool(sponge_home and coverage_ok)
    msg = (
        f"sponge_home={sponge_home} coverage_ok={coverage_ok} "
        f"wipe_coverage_final={coverage:.3f} "
        f"sponge_to_pad_final={sponge_to_pad_final:.3f}"
    )
    metrics = {
        "subtask": "wipe",
        "sponge_to_pad_final": sponge_to_pad_final,
        "wipe_coverage_final": coverage,
        "sponge_home": bool(sponge_home),
        "coverage_ok": bool(coverage_ok),
        "task_success": ok,
        "episode_length": len(obs_list),
    }
    return ok, msg, metrics

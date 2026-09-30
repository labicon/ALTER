# Two-arm place + wipe environment.
#
# Robot 0 (-Y side, facing +Y):  picks up a HandleCubeObject by its inverted-U
#   handle and places it on a colored destination pad (visual-only site).
# Robot 1 (+Y side, facing -Y):  grasps a sponge from its home pad (visual-only
#   site), wipes the dirt-tile grid at the table center, and returns the sponge
#   to the home pad.
#
# Four modes = 2 destination-pad positions * 2 sponge-pad positions.

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects.primitive.box import BoxObject as Box
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_element
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from envs.arm.data_gen.utils.singlearm_wipe_env import HandleCubeObject


# The handle cube spawns on top of the dirt patch at the table center.
# Robot 0 (-Y side) must lift it off the dirt before Robot 1 (+Y side) can wipe.
CUBE_SPAWN_OFFSET = np.array([0.0, 0.0, 0.0])

# Destination pads sit near Robot 0's base (y < 0), so the lift + place motion
# retreats toward the -Y side of the table. Two x positions give 2 dest modes.
DESTINATION_PAD_OFFSETS = {
    0: np.array([-0.20, -0.25, 0.0]),
    1: np.array([+0.20, -0.25, 0.0]),
}

# Sponge spawn locations (bare table, no pad rendered) near Robot 1's base.
SPONGE_SPAWN_OFFSETS = {
    0: np.array([-0.20, +0.20, 0.0]),
    1: np.array([+0.20, +0.20, 0.0]),
}

# Sponge drop pads: mirrored across x=0 from the spawn so the sweep is one-way.
SPONGE_PAD_OFFSETS = {
    0: np.array([+0.20, +0.20, 0.0]),
    1: np.array([-0.20, +0.20, 0.0]),
}

HANDLE_CUBE_RGBA = (0.3, 0.8, 0.3, 1.0)
DEST_PAD_RGBA = (0.2, 0.4, 0.9, 0.8)
SPONGE_PAD_RGBA = (0.95, 0.6, 0.1, 0.75)
# Return pad lives directly under the dirt tiles so it is hidden before the
# wipe and revealed as the tiles turn transparent.
RETURN_PAD_RGBA = (0.9, 0.25, 0.5, 0.9)


class TwoArmPlaceWipe(ManipulationEnv):
    """Two-arm place-and-wipe task."""

    # Scene constants (match WipeSceneMixin so cross-task data is comparable)
    SPONGE_SIZE = [0.03, 0.05, 0.03]
    DIRT_PATCH_HALF_SIZE = (0.05, 0.08)
    DIRT_GRID_SHAPE = (8, 5)
    DIRT_TILE_RGBA = (0.35, 0.20, 0.08, 0.95)
    DIRT_SPONGE_Z_TOL = 0.04
    SHOULDER_CAM_LOCAL_OFFSET = np.array([0.3, 0.0, 1.0])

    DEST_PAD_HALF_SIZE = (0.06, 0.06)
    SPONGE_PAD_HALF_SIZE = (0.08, 0.06)
    # Small marker at the dirt-patch center so the surrounding dirt tiles
    # remain visible around it.
    RETURN_PAD_HALF_SIZE = (0.015, 0.015)

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.95, 1.4, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=10,
        lite_physics=True,
        horizon=1500,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        self._dest_mode = 0
        self._sponge_mode = 0

        self._dirt_tile_ids = np.empty((0, 0), dtype=np.int64)
        self._dirt_tile_cleaned = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------
    def _check_robot_configuration(self, robots):
        super()._check_robot_configuration(robots)
        robots_list = [robots] if isinstance(robots, str) else list(robots)
        if len(robots_list) != 2:
            raise AssertionError(
                f"TwoArmPlaceWipe requires exactly 2 robots, got {len(robots_list)}"
            )

    def set_mode(self, dest_mode=0, sponge_mode=0):
        if dest_mode not in DESTINATION_PAD_OFFSETS:
            raise ValueError(f"Invalid dest_mode {dest_mode}")
        if sponge_mode not in SPONGE_PAD_OFFSETS:
            raise ValueError(f"Invalid sponge_mode {sponge_mode}")
        self._dest_mode = dest_mode
        self._sponge_mode = sponge_mode
        if hasattr(self, "sponge_placement_initializer"):
            self.sponge_placement_initializer.reference_pos = (
                self.table_offset + SPONGE_SPAWN_OFFSETS[sponge_mode]
            )

    # ------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()

        # Robot placement: opposed -Y / +Y (mirrors robots 0,1 of three-arm).
        rotations = [np.pi / 2, -np.pi / 2]
        for robot, rotation in zip(self.robots, rotations):
            xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
            xpos = np.array(xpos)
            xpos[0] -= 0.2
            rot = np.array((0, 0, rotation))
            xpos = T.euler2mat(rot) @ xpos
            robot.robot_model.set_base_xpos(xpos)
            robot.robot_model.set_base_ori(rot)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self._build_scene_objects()
        self._build_placement_initializers()

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.handle_cube, self.sponge],
        )

        # Pads first so the dirt tiles are appended to worldbody AFTER the
        # return_pad — later-in-XML opaque sites win the render order, so the
        # tiles stay visible on top of the pad until they go transparent.
        self._add_pads()
        self._add_dirt_patch()
        self._add_shoulder_cameras()

    # ------------------------------------------------------------------
    def _build_scene_objects(self):
        self.handle_cube = HandleCubeObject(
            name="handle_cube",
            rgba_handle=HANDLE_CUBE_RGBA,
            handle_thickness=0.012,
        )
        self.sponge = Box(
            name="sponge",
            size=list(self.SPONGE_SIZE),
            rgba=[1.0, 0.9, 0.2, 1.0],
            density=100.0,
        )

    def _build_placement_initializers(self):
        cube_ref = self.table_offset + CUBE_SPAWN_OFFSET
        self.placement_initializer = UniformRandomSampler(
            name="HandleCubeSampler",
            mujoco_objects=self.handle_cube,
            x_range=[-0.02, 0.02],
            y_range=[-0.02, 0.02],
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=cube_ref,
            rotation=0.0,
        )

        self.sponge_placement_initializer = UniformRandomSampler(
            name="SpongeSampler",
            mujoco_objects=self.sponge,
            x_range=[-0.03, 0.03],
            y_range=[-0.03, 0.03],
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset + SPONGE_SPAWN_OFFSETS[self._sponge_mode],
            rotation=0.0,
        )

    # ------------------------------------------------------------------
    def _add_dirt_patch(self):
        table_x, table_y, table_z = self.table_offset
        dx, dy = self.DIRT_PATCH_HALF_SIZE
        nx, ny = self.DIRT_GRID_SHAPE

        bbox_elem = new_element(
            tag="site", name="dirt_patch", type="box",
            pos=array_to_string(np.array([table_x, table_y, table_z + 0.0005])),
            size=array_to_string(np.array([dx, dy, 0.0005])),
            rgba="0 0 0 0",
        )
        self.model.worldbody.append(bbox_elem)

        tile_hx = dx / nx
        tile_hy = dy / ny
        rgba_str = " ".join(f"{v}" for v in self.DIRT_TILE_RGBA)
        for i in range(nx):
            cx = table_x + (2 * i + 1) * tile_hx - dx
            for j in range(ny):
                cy = table_y + (2 * j + 1) * tile_hy - dy
                # Tile z is raised well above the return_pad (which sits at
                # table_z + 0.0002) so tiles are clearly on top of the pad in
                # both XML order and world z — guarantees the dirt is visible
                # before wiping regardless of the renderer's draw-order rule.
                self.model.worldbody.append(new_element(
                    tag="site", name=f"dirt_tile_{i}_{j}", type="box",
                    pos=array_to_string(np.array([cx, cy, table_z + 0.004])),
                    size=array_to_string(np.array([tile_hx * 0.92, tile_hy * 0.92, 0.001])),
                    rgba=rgba_str,
                ))

    def _add_pads(self):
        dest_pos = self.table_offset + DESTINATION_PAD_OFFSETS[self._dest_mode]
        dest_pos = dest_pos.copy()
        dest_pos[2] += 0.0005
        self.model.worldbody.append(new_element(
            tag="site", name="destination_pad", type="box",
            pos=array_to_string(dest_pos),
            size=array_to_string(np.array([
                self.DEST_PAD_HALF_SIZE[0], self.DEST_PAD_HALF_SIZE[1], 0.0005,
            ])),
            rgba=" ".join(f"{v}" for v in DEST_PAD_RGBA),
        ))

        sponge_pos = self.table_offset + SPONGE_PAD_OFFSETS[self._sponge_mode]
        sponge_pos = sponge_pos.copy()
        sponge_pos[2] += 0.0005
        self.model.worldbody.append(new_element(
            tag="site", name="sponge_pad", type="box",
            pos=array_to_string(sponge_pos),
            size=array_to_string(np.array([
                self.SPONGE_PAD_HALF_SIZE[0], self.SPONGE_PAD_HALF_SIZE[1], 0.0005,
            ])),
            rgba=" ".join(f"{v}" for v in SPONGE_PAD_RGBA),
        ))

        # Return pad: sits at the dirt-patch center, under the dirt tiles.
        # Z is placed just above the table but BELOW the dirt tiles (tiles at
        # table_z + 0.001) so the tiles occlude it until they go transparent.
        return_pos = self.table_offset.copy()
        return_pos[2] += 0.0002
        self.model.worldbody.append(new_element(
            tag="site", name="return_pad", type="box",
            pos=array_to_string(return_pos),
            size=array_to_string(np.array([
                self.RETURN_PAD_HALF_SIZE[0], self.RETURN_PAD_HALF_SIZE[1], 0.0002,
            ])),
            rgba=" ".join(f"{v}" for v in RETURN_PAD_RGBA),
        ))

    def _add_shoulder_cameras(self):
        cam_local_offset = self.SHOULDER_CAM_LOCAL_OFFSET
        target_world = np.array([
            self.table_offset[0], self.table_offset[1], self.table_offset[2] + 0.025,
        ])
        for robot_idx in range(len(self.robots)):
            base_name = f"robot{robot_idx}_base"
            base_body = self.model.worldbody.find(f".//body[@name='{base_name}']")
            if base_body is None:
                raise RuntimeError(f"Could not find base body '{base_name}'.")
            base_pos = np.array([float(v) for v in base_body.get("pos", "0 0 0").split()])
            base_quat_wxyz = np.array([float(v) for v in base_body.get("quat", "1 0 0 0").split()])
            base_rot = T.quat2mat(np.roll(base_quat_wxyz, -1))
            cam_world_pos = base_pos + base_rot @ cam_local_offset
            cam_quat = self._camera_lookat_quat(cam_world_pos, target_world)
            self.model.worldbody.append(new_element(
                tag="camera", name=f"robot{robot_idx}_agentview_shoulder",
                pos=array_to_string(cam_world_pos),
                quat=array_to_string(cam_quat),
                fovy="60",
            ))

    @staticmethod
    def _camera_lookat_quat(cam_pos, target_pos):
        fwd = target_pos - cam_pos
        fwd = fwd / np.linalg.norm(fwd)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up); right /= np.linalg.norm(right)
        up = np.cross(right, fwd); up /= np.linalg.norm(up)
        rot = np.column_stack([right, up, -fwd])
        quat_xyzw = T.mat2quat(rot)
        return np.roll(quat_xyzw, 1)

    # ------------------------------------------------------------------
    def _setup_references(self):
        super()._setup_references()
        self.table_top_id = self.sim.model.site_name2id("table_top")

        self.handle_cube_body_id = self.sim.model.body_name2id(self.handle_cube.root_body)
        self.handle_site_id = self.sim.model.site_name2id(
            self.handle_cube.important_sites["handle"]
        )
        self.cube_center_id = self.sim.model.site_name2id(
            self.handle_cube.important_sites["center"]
        )

        self.sponge_body_id = self.sim.model.body_name2id(self.sponge.root_body)

        self.dirt_patch_site_id = self.sim.model.site_name2id("dirt_patch")
        self.destination_pad_site_id = self.sim.model.site_name2id("destination_pad")
        self.sponge_pad_site_id = self.sim.model.site_name2id("sponge_pad")
        self.return_pad_site_id = self.sim.model.site_name2id("return_pad")

        nx, ny = self.DIRT_GRID_SHAPE
        self._dirt_tile_ids = np.array(
            [[self.sim.model.site_name2id(f"dirt_tile_{i}_{j}") for j in range(ny)]
             for i in range(nx)],
            dtype=np.int64,
        )
        self._dirt_tile_rgba_default = np.array(self.DIRT_TILE_RGBA, dtype=np.float32)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        @sensor(modality=modality)
        def handle_cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.handle_cube_body_id])

        @sensor(modality=modality)
        def handle_cube_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.handle_cube_body_id], to="xyzw")

        @sensor(modality=modality)
        def handle_xpos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.handle_site_id])

        @sensor(modality=modality)
        def sponge_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.sponge_body_id])

        @sensor(modality=modality)
        def dirt_patch_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.dirt_patch_site_id])

        @sensor(modality=modality)
        def destination_pad_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.destination_pad_site_id])

        @sensor(modality=modality)
        def sponge_pad_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.sponge_pad_site_id])

        @sensor(modality=modality)
        def return_pad_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.return_pad_site_id])

        sensors = [
            handle_cube_pos, handle_cube_quat, handle_xpos,
            sponge_pos, dirt_patch_pos, destination_pad_pos, sponge_pad_pos,
            return_pad_pos,
        ]
        for s in sensors:
            observables[s.__name__] = Observable(
                name=s.__name__, sensor=s, sampling_rate=self.control_freq,
            )
        return observables

    # ------------------------------------------------------------------
    def _reset_internal(self):
        if hasattr(self, "sponge_placement_initializer"):
            self.sponge_placement_initializer.reference_pos = (
                self.table_offset + SPONGE_SPAWN_OFFSETS[self._sponge_mode]
            )
        super()._reset_internal()
        self._reset_scene()

    def _reset_scene(self):
        if self.deterministic_reset:
            return

        cube_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in cube_placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
            )

        sponge_placements = self.sponge_placement_initializer.sample()
        for obj_pos, obj_quat, obj in sponge_placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
            )

        nx, ny = self.DIRT_GRID_SHAPE
        self._dirt_tile_cleaned = np.zeros((nx, ny), dtype=bool)
        default_rgba = self._dirt_tile_rgba_default
        for i in range(nx):
            for j in range(ny):
                self.sim.model.site_rgba[self._dirt_tile_ids[i, j]] = default_rgba

    # ------------------------------------------------------------------
    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_wipe_coverage()
        return reward, done, info

    def _update_wipe_coverage(self):
        if self._dirt_tile_cleaned is None:
            return
        sponge_pos = self.sim.data.body_xpos[self.sponge_body_id]
        table_z = self.sim.data.site_xpos[self.table_top_id][2]
        if sponge_pos[2] > table_z + self.DIRT_SPONGE_Z_TOL:
            return
        sx_half = self.SPONGE_SIZE[0] + 0.005
        sy_half = self.SPONGE_SIZE[1] + 0.005
        nx, ny = self.DIRT_GRID_SHAPE
        transparent = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                if self._dirt_tile_cleaned[i, j]:
                    continue
                tile_pos = self.sim.data.site_xpos[self._dirt_tile_ids[i, j]]
                if (
                    abs(sponge_pos[0] - tile_pos[0]) < sx_half
                    and abs(sponge_pos[1] - tile_pos[1]) < sy_half
                ):
                    self._dirt_tile_cleaned[i, j] = True
                    self.sim.model.site_rgba[self._dirt_tile_ids[i, j]] = transparent

    @property
    def _wipe_coverage(self):
        if self._dirt_tile_cleaned is None:
            return 0.0
        return float(self._dirt_tile_cleaned.mean())

    # ------------------------------------------------------------------
    def reward(self, action=None):
        reward = 1.0 if self._check_success() else 0.0
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _check_success(self):
        # Cube placed back on the return pad (after dest-pad holding + wipe)?
        cube_xy = self.sim.data.body_xpos[self.handle_cube_body_id][:2]
        return_xy = self.sim.data.site_xpos[self.return_pad_site_id][:2]
        rx, ry = self.RETURN_PAD_HALF_SIZE
        place_ok = (abs(cube_xy[0] - return_xy[0]) < rx
                    and abs(cube_xy[1] - return_xy[1]) < ry)

        # Wipe coverage?
        wipe_ok = self._wipe_coverage >= 0.6

        # Sponge returned home?
        sponge_xy = self.sim.data.body_xpos[self.sponge_body_id][:2]
        pad_xy = self.sim.data.site_xpos[self.sponge_pad_site_id][:2]
        sx, sy = self.SPONGE_PAD_HALF_SIZE
        sponge_home_ok = (abs(sponge_xy[0] - pad_xy[0]) < sx
                         and abs(sponge_xy[1] - pad_xy[1]) < sy)

        return place_ok and wipe_ok and sponge_home_ok


if __name__ == "__main__":
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        robot="Kinova3", controller="envs/arm/data_gen/kinova.json",
    )
    env = TwoArmPlaceWipe(
        robots=["Kinova3", "Kinova3"],
        gripper_types="default",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=500,
    )
    obs = env.reset()
    print(f"Action dim: {env.action_spec[0].shape[0]}")
    print(f"Obs keys: {sorted(obs.keys())}")
    print(f"Cube pos: {env.sim.data.body_xpos[env.handle_cube_body_id]}")
    print(f"Sponge pos: {env.sim.data.body_xpos[env.sponge_body_id]}")
    print(f"Dest pad: {env.sim.data.site_xpos[env.destination_pad_site_id]}")
    print(f"Sponge pad: {env.sim.data.site_xpos[env.sponge_pad_site_id]}")
    for i in range(2):
        base_id = env.sim.model.body_name2id(f"robot{i}_base")
        print(f"Robot{i} base: {env.sim.data.body_xpos[base_id]}")
    for _ in range(100):
        a = np.random.uniform(-0.05, 0.05, env.action_spec[0].shape[0])
        env.step(a)
        env.render()
    env.close()

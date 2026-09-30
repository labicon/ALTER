# Single-arm mirrors of the two-arm place+wipe subtasks.
#
# Two task variants, each uses exactly one Kinova3 and the same scene layout
# as TwoArmPlaceWipe — but with only the objects/pads the active arm touches:
#
#   - "place_return": one arm (-Y, rot +pi/2 = robot0 slot in two-arm) grasps a
#       HandleCubeObject from the dirt-patch center, places it on a colored
#       destination pad (gripper closed, holds there), lifts, and places it on
#       the small return-pad marker at the dirt center. Mode selects one of
#       two destination-pad positions.
#   - "wipe": one arm (+Y, rot -pi/2 = robot1 slot in two-arm) grasps a sponge
#       from the sponge pad, sweeps the dirt-tile grid, and returns the sponge
#       to the pad. Mode selects one of two sponge-pad positions.
#
# This file also provides the `HandleCubeObject` CompositeObject (unchanged
# signature/behavior) that `twoarm_wipe_env.py` imports.

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CompositeObject
from robosuite.models.objects.primitive.box import BoxObject as Box
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import add_to_dict, array_to_string, new_element
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler


# ---------------------------------------------------------------------------
# Scene constants — match twoarm_wipe_env.py exactly so single-arm demos live
# in the same geometric frame as two-arm demos.
# ---------------------------------------------------------------------------

TASK_ROTATIONS = {
    "place_return": np.pi / 2,   # -Y side (robot0 slot in two-arm)
    "wipe": -np.pi / 2,          # +Y side (robot1 slot in two-arm)
}

CUBE_SPAWN_OFFSET = np.array([0.0, 0.0, 0.0])

DESTINATION_PAD_OFFSETS = {
    0: np.array([-0.20, -0.25, 0.0]),
    1: np.array([+0.20, -0.25, 0.0]),
}

# Sponge spawn locations: sponge sits on the bare table (no pad rendered here).
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
RETURN_PAD_RGBA = (0.9, 0.25, 0.5, 0.9)


# ---------------------------------------------------------------------------
# HandleCubeObject — preserved verbatim (imported by twoarm_wipe_env.py).
# ---------------------------------------------------------------------------

class HandleCubeObject(CompositeObject):
    """Small base block with one inverted-U handle on top.

    Mirrors the handle geometry of TrayObject so the grasp policy is identical.
    """

    def __init__(
        self,
        name,
        base_half_size=(0.05, 0.05, 0.02),
        handle_post_height=0.10,
        handle_post_separation=0.10,
        handle_thickness=0.008,
        handle_friction=1.0,
        rgba_base=(0.7, 0.7, 0.7, 1.0),
        rgba_handle=(0.3, 0.8, 0.3, 1.0),
        density=500.0,
        friction=(1.0, 5e-3, 1e-4),
    ):
        self._name = name
        self.base_half_size = np.array(base_half_size)
        self.handle_post_height = handle_post_height
        self.handle_post_separation = handle_post_separation
        self.handle_thickness = handle_thickness
        self.handle_friction = handle_friction
        self.rgba_base = np.array(rgba_base)
        self.rgba_handle = np.array(rgba_handle)
        self._density = density
        self._friction = np.array(friction) if friction is not None else None

        self._handle_geoms = []
        self._important_sites = {}

        super().__init__(**self._get_geom_attrs())

    def _get_geom_attrs(self):
        bx, by, bz = self.base_half_size
        ht = self.handle_thickness / 2.0
        hp = self.handle_post_height
        hs_half = self.handle_post_separation / 2.0
        base_top_z = bz
        post_center_z = base_top_z + hp / 2.0
        bar_center_z = base_top_z + hp - ht

        base_args = {
            "total_size": np.array([bx, by, bz + hp / 2.0]),
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self._density,
        }
        obj_args = {}
        site_attrs = []

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, 0),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=(bx, by, bz),
            geom_names="base",
            geom_rgbas=self.rgba_base,
            geom_materials=None,
            geom_frictions=self._friction,
        )

        for post_sign, post_suffix in [(-1.0, "l"), (1.0, "r")]:
            name = f"handle_post_{post_suffix}"
            self._handle_geoms.append(name)
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(post_sign * hs_half, 0.0, post_center_z),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=(ht, ht, hp / 2.0),
                geom_names=name,
                geom_rgbas=self.rgba_handle,
                geom_materials=None,
                geom_frictions=(self.handle_friction, 0.005, 0.0001),
            )
        bar_name = "handle_bar"
        self._handle_geoms.append(bar_name)
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0.0, 0.0, bar_center_z),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=(hs_half + ht, ht, ht),
            geom_names=bar_name,
            geom_rgbas=self.rgba_handle,
            geom_materials=None,
            geom_frictions=(self.handle_friction, 0.005, 0.0001),
        )

        grasp_site = self.get_site_attrib_template()
        grasp_site.update({
            "name": "handle",
            "pos": array_to_string(np.array([0.0, 0.0, bar_center_z])),
            "size": "0.005",
            "rgba": array_to_string(self.rgba_handle),
        })
        site_attrs.append(grasp_site)
        self._important_sites["handle"] = self.naming_prefix + "handle"

        center_site = self.get_site_attrib_template()
        center_site.update({
            "name": "center",
            "pos": array_to_string(np.array([0, 0, 0])),
            "size": "0.005",
        })
        site_attrs.append(center_site)
        self._important_sites["center"] = self.naming_prefix + "center"

        obj_args.update(base_args)
        obj_args["sites"] = site_attrs
        return obj_args

    @property
    def important_sites(self):
        dic = super().important_sites
        dic.update(self._important_sites)
        return dic

    @property
    def bottom_offset(self):
        return np.array([0.0, 0.0, -self.base_half_size[2]])

    @property
    def top_offset(self):
        return np.array([0.0, 0.0, self.base_half_size[2] + self.handle_post_height])

    @property
    def horizontal_radius(self):
        return float(np.sqrt(self.base_half_size[0] ** 2 + self.base_half_size[1] ** 2))


# ---------------------------------------------------------------------------
# Single-arm env
# ---------------------------------------------------------------------------

class SingleArmPlaceWipe(ManipulationEnv):
    """Single-arm mirror of one subtask from `TwoArmPlaceWipe`.

    Args:
        task (str): one of {"place_return", "wipe"}.
    """

    # Scene constants — identical to TwoArmPlaceWipe.
    SPONGE_SIZE = [0.03, 0.05, 0.03]
    DIRT_PATCH_HALF_SIZE = (0.05, 0.08)
    DIRT_GRID_SHAPE = (8, 5)
    DIRT_TILE_RGBA = (0.35, 0.20, 0.08, 0.95)
    DIRT_SPONGE_Z_TOL = 0.04
    SHOULDER_CAM_LOCAL_OFFSET = np.array([0.3, 0.0, 1.0])

    DEST_PAD_HALF_SIZE = (0.06, 0.06)
    SPONGE_PAD_HALF_SIZE = (0.08, 0.06)
    RETURN_PAD_HALF_SIZE = (0.015, 0.015)

    def __init__(
        self,
        robots,
        task="place_return",
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
        if task not in TASK_ROTATIONS:
            raise ValueError(f"Invalid task '{task}'. Must be one of {list(TASK_ROTATIONS)}")
        self.task = task

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        self._mode = 0  # mode index (dest pad index for place_return, sponge pad index for wipe)

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
        if len(robots_list) != 1:
            raise AssertionError(
                f"SingleArmPlaceWipe requires exactly 1 robot, got {len(robots_list)}"
            )

    def set_mode(self, mode=0):
        if self.task == "place_return":
            # Modes 0/1 = forward (cube spawns at center, placed on dest pad `mode`).
            # Modes 2/3 = backward (cube spawns on dest pad `mode - 2`, placed on return_pad).
            if mode not in (0, 1, 2, 3):
                raise ValueError(f"Invalid place_return mode {mode}")
        else:  # wipe
            if mode not in SPONGE_PAD_OFFSETS:
                raise ValueError(f"Invalid sponge mode {mode}")
        self._mode = mode
        if self.task == "wipe" and hasattr(self, "sponge_placement_initializer"):
            self.sponge_placement_initializer.reference_pos = (
                self.table_offset + SPONGE_SPAWN_OFFSETS[mode]
            )
        if self.task == "place_return" and hasattr(self, "placement_initializer"):
            self.placement_initializer.reference_pos = (
                self.table_offset + self._place_return_cube_spawn_offset()
            )

    # ------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()

        rotation = TASK_ROTATIONS[self.task]
        robot = self.robots[0]
        xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos = np.array(xpos)
        xpos[0] -= 0.2  # match twoarm_wipe_env.py:175
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
            mujoco_objects=self._scene_mujoco_objects(),
        )

        # Pads before tiles so tiles win the draw-order tie on the return_pad.
        # The dirt patch is only relevant to the wipe task; place_return omits
        # it entirely (the return_pad marker is the only cue at the table center).
        self._add_pads()
        if self.task == "wipe":
            self._add_dirt_patch()
        self._add_shoulder_cameras()

    # ------------------------------------------------------------------
    def _build_scene_objects(self):
        if self.task == "place_return":
            self.handle_cube = HandleCubeObject(
                name="handle_cube",
                rgba_handle=HANDLE_CUBE_RGBA,
                handle_thickness=0.012,
            )
            self.sponge = None
        else:  # wipe
            self.handle_cube = None
            self.sponge = Box(
                name="sponge",
                size=list(self.SPONGE_SIZE),
                rgba=[1.0, 0.9, 0.2, 1.0],
                density=100.0,
            )

    def _scene_mujoco_objects(self):
        if self.task == "place_return":
            return [self.handle_cube]
        return [self.sponge]

    def _place_return_cube_spawn_offset(self):
        """Center for modes 0/1 (forward); on dest pad for modes 2/3 (backward)."""
        direction = self._mode // 2
        dest_idx = self._mode % 2
        if direction == 0:
            return CUBE_SPAWN_OFFSET
        return DESTINATION_PAD_OFFSETS[dest_idx]

    def _build_placement_initializers(self):
        if self.task == "place_return":
            self.placement_initializer = UniformRandomSampler(
                name="HandleCubeSampler",
                mujoco_objects=self.handle_cube,
                x_range=[-0.02, 0.02],
                y_range=[-0.02, 0.02],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset + self._place_return_cube_spawn_offset(),
                rotation=0.0,
            )
        else:  # wipe
            self.sponge_placement_initializer = UniformRandomSampler(
                name="SpongeSampler",
                mujoco_objects=self.sponge,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset + SPONGE_SPAWN_OFFSETS[self._mode],
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
                self.model.worldbody.append(new_element(
                    tag="site", name=f"dirt_tile_{i}_{j}", type="box",
                    pos=array_to_string(np.array([cx, cy, table_z + 0.004])),
                    size=array_to_string(np.array([tile_hx * 0.92, tile_hy * 0.92, 0.001])),
                    rgba=rgba_str,
                ))

    def _add_pads(self):
        if self.task == "place_return":
            dest_pos = self.table_offset + DESTINATION_PAD_OFFSETS[self._mode % 2]
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
        else:  # wipe
            sponge_pos = self.table_offset + SPONGE_PAD_OFFSETS[self._mode]
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

    def _add_shoulder_cameras(self):
        cam_local_offset = self.SHOULDER_CAM_LOCAL_OFFSET
        target_world = np.array([
            self.table_offset[0], self.table_offset[1], self.table_offset[2] + 0.025,
        ])
        base_name = "robot0_base"
        base_body = self.model.worldbody.find(f".//body[@name='{base_name}']")
        if base_body is None:
            raise RuntimeError(f"Could not find base body '{base_name}'.")
        base_pos = np.array([float(v) for v in base_body.get("pos", "0 0 0").split()])
        base_quat_wxyz = np.array([float(v) for v in base_body.get("quat", "1 0 0 0").split()])
        base_rot = T.quat2mat(np.roll(base_quat_wxyz, -1))
        cam_world_pos = base_pos + base_rot @ cam_local_offset
        cam_quat = self._camera_lookat_quat(cam_world_pos, target_world)
        self.model.worldbody.append(new_element(
            tag="camera", name="robot0_agentview_shoulder",
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

        if self.task == "place_return":
            self.handle_cube_body_id = self.sim.model.body_name2id(self.handle_cube.root_body)
            self.handle_site_id = self.sim.model.site_name2id(
                self.handle_cube.important_sites["handle"]
            )
            self.cube_center_id = self.sim.model.site_name2id(
                self.handle_cube.important_sites["center"]
            )
            self.destination_pad_site_id = self.sim.model.site_name2id("destination_pad")
            self.return_pad_site_id = self.sim.model.site_name2id("return_pad")
        else:  # wipe
            self.sponge_body_id = self.sim.model.body_name2id(self.sponge.root_body)
            self.sponge_pad_site_id = self.sim.model.site_name2id("sponge_pad")
            self.dirt_patch_site_id = self.sim.model.site_name2id("dirt_patch")
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

        sensors = []

        if self.task == "place_return":
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
            def destination_pad_pos(obs_cache):
                return np.array(self.sim.data.site_xpos[self.destination_pad_site_id])

            @sensor(modality=modality)
            def return_pad_pos(obs_cache):
                return np.array(self.sim.data.site_xpos[self.return_pad_site_id])

            sensors += [handle_cube_pos, handle_cube_quat, handle_xpos,
                        destination_pad_pos, return_pad_pos]
        else:  # wipe
            @sensor(modality=modality)
            def sponge_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.sponge_body_id])

            @sensor(modality=modality)
            def sponge_pad_pos(obs_cache):
                return np.array(self.sim.data.site_xpos[self.sponge_pad_site_id])

            @sensor(modality=modality)
            def dirt_patch_pos(obs_cache):
                return np.array(self.sim.data.site_xpos[self.dirt_patch_site_id])

            sensors += [sponge_pos, sponge_pad_pos, dirt_patch_pos]

        for s in sensors:
            observables[s.__name__] = Observable(
                name=s.__name__, sensor=s, sampling_rate=self.control_freq,
            )
        return observables

    # ------------------------------------------------------------------
    def _reset_internal(self):
        if self.task == "wipe" and hasattr(self, "sponge_placement_initializer"):
            self.sponge_placement_initializer.reference_pos = (
                self.table_offset + SPONGE_SPAWN_OFFSETS[self._mode]
            )
        if self.task == "place_return" and hasattr(self, "placement_initializer"):
            self.placement_initializer.reference_pos = (
                self.table_offset + self._place_return_cube_spawn_offset()
            )
        super()._reset_internal()
        self._reset_scene()

    def _reset_scene(self):
        if self.deterministic_reset:
            return

        if self.task == "place_return":
            cube_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in cube_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )
        else:  # wipe
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
        if self.task == "wipe":
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
        if self.task == "place_return":
            cube_xy = self.sim.data.body_xpos[self.handle_cube_body_id][:2]
            direction = self._mode // 2
            if direction == 0:
                # Forward: cube should end on the dest pad.
                dest_xy = self.sim.data.site_xpos[self.destination_pad_site_id][:2]
                dx, dy = self.DEST_PAD_HALF_SIZE
                return (abs(cube_xy[0] - dest_xy[0]) < dx
                        and abs(cube_xy[1] - dest_xy[1]) < dy)
            # Backward: cube should end on the pink return_pad at center.
            return_xy = self.sim.data.site_xpos[self.return_pad_site_id][:2]
            rx, ry = self.RETURN_PAD_HALF_SIZE
            return (abs(cube_xy[0] - return_xy[0]) < rx
                    and abs(cube_xy[1] - return_xy[1]) < ry)
        else:  # wipe
            sponge_xy = self.sim.data.body_xpos[self.sponge_body_id][:2]
            pad_xy = self.sim.data.site_xpos[self.sponge_pad_site_id][:2]
            sx, sy = self.SPONGE_PAD_HALF_SIZE
            sponge_home_ok = (abs(sponge_xy[0] - pad_xy[0]) < sx
                              and abs(sponge_xy[1] - pad_xy[1]) < sy)
            return self._wipe_coverage >= 0.6 and sponge_home_ok

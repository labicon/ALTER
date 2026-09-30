"""Hard-variant single-arm place/wipe env.

This keeps the same task semantics as ``singlearm_wipe_env.py`` but widens the
seed-conditioned geometry distribution used for expert demo generation.
"""

import numpy as np

from robosuite.models.objects.primitive.box import BoxObject as Box
from robosuite.utils.mjcf_utils import array_to_string, new_element
from robosuite.utils.placement_samplers import UniformRandomSampler

from envs.arm.data_gen.utils.singlearm_wipe_env import (
    CUBE_SPAWN_OFFSET,
    DESTINATION_PAD_OFFSETS,
    DEST_PAD_RGBA,
    HANDLE_CUBE_RGBA,
    HandleCubeObject,
    RETURN_PAD_RGBA,
    SPONGE_PAD_OFFSETS,
    SPONGE_PAD_RGBA,
    SPONGE_SPAWN_OFFSETS,
    SingleArmPlaceWipe,
)


class SingleArmPlaceWipeHard(SingleArmPlaceWipe):
    """Single-arm place/wipe env with hard-v1 geometry randomization."""

    CUBE_XY_RANGE = (-0.05, 0.05)
    SPONGE_XY_RANGE = (-0.06, 0.06)
    DEST_PAD_JITTER = 0.04
    SPONGE_PAD_JITTER = 0.04
    DIRT_CENTER_JITTER = 0.035

    def _sample_hard_offsets(self):
        self._hard_return_offset = CUBE_SPAWN_OFFSET.copy()
        self._hard_return_offset[:2] += np.random.uniform(
            -self.DIRT_CENTER_JITTER, self.DIRT_CENTER_JITTER, size=2
        )

        self._hard_dest_offsets = {}
        for mode, offset in DESTINATION_PAD_OFFSETS.items():
            v = offset.copy()
            v[:2] += np.random.uniform(-self.DEST_PAD_JITTER, self.DEST_PAD_JITTER, size=2)
            self._hard_dest_offsets[mode] = v

        self._hard_sponge_spawn_offsets = {}
        for mode, offset in SPONGE_SPAWN_OFFSETS.items():
            self._hard_sponge_spawn_offsets[mode] = offset.copy()

        self._hard_sponge_pad_offsets = {}
        for mode, offset in SPONGE_PAD_OFFSETS.items():
            v = offset.copy()
            v[:2] += np.random.uniform(
                -self.SPONGE_PAD_JITTER, self.SPONGE_PAD_JITTER, size=2
            )
            self._hard_sponge_pad_offsets[mode] = v

    def _load_model(self):
        self._sample_hard_offsets()
        super()._load_model()

    def _place_return_cube_spawn_offset(self):
        direction = self._mode // 2
        dest_idx = self._mode % 2
        if direction == 0:
            return self._hard_return_offset
        return self._hard_dest_offsets[dest_idx]

    def _build_scene_objects(self):
        if self.task == "place_return":
            self.handle_cube = HandleCubeObject(
                name="handle_cube",
                rgba_handle=HANDLE_CUBE_RGBA,
                handle_thickness=0.012,
            )
            self.sponge = None
        else:
            self.handle_cube = None
            self.sponge = Box(
                name="sponge",
                size=list(self.SPONGE_SIZE),
                rgba=[1.0, 0.9, 0.2, 1.0],
                density=100.0,
            )

    def _build_placement_initializers(self):
        if self.task == "place_return":
            self.placement_initializer = UniformRandomSampler(
                name="HandleCubeSamplerHard",
                mujoco_objects=self.handle_cube,
                x_range=list(self.CUBE_XY_RANGE),
                y_range=list(self.CUBE_XY_RANGE),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset + self._place_return_cube_spawn_offset(),
                rotation=0.0,
            )
        else:
            self.sponge_placement_initializer = UniformRandomSampler(
                name="SpongeSamplerHard",
                mujoco_objects=self.sponge,
                x_range=list(self.SPONGE_XY_RANGE),
                y_range=list(self.SPONGE_XY_RANGE),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset + self._hard_sponge_spawn_offsets[self._mode],
                rotation=0.0,
            )

    def _add_dirt_patch(self):
        dirt_center = self.table_offset + self._hard_return_offset
        table_x, table_y = dirt_center[:2]
        table_z = self.table_offset[2]
        dx, dy = self.DIRT_PATCH_HALF_SIZE
        nx, ny = self.DIRT_GRID_SHAPE

        self.model.worldbody.append(new_element(
            tag="site", name="dirt_patch", type="box",
            pos=array_to_string(np.array([table_x, table_y, table_z + 0.0005])),
            size=array_to_string(np.array([dx, dy, 0.0005])),
            rgba="0 0 0 0",
        ))

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
            dest_pos = self.table_offset + self._hard_dest_offsets[self._mode % 2]
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

            return_pos = self.table_offset + self._hard_return_offset
            return_pos = return_pos.copy()
            return_pos[2] += 0.0002
            self.model.worldbody.append(new_element(
                tag="site", name="return_pad", type="box",
                pos=array_to_string(return_pos),
                size=array_to_string(np.array([
                    self.RETURN_PAD_HALF_SIZE[0], self.RETURN_PAD_HALF_SIZE[1], 0.0002,
                ])),
                rgba=" ".join(f"{v}" for v in RETURN_PAD_RGBA),
            ))
        else:
            sponge_pos = self.table_offset + self._hard_sponge_pad_offsets[self._mode]
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

    def _reset_internal(self):
        if self.task == "wipe" and hasattr(self, "sponge_placement_initializer"):
            self.sponge_placement_initializer.reference_pos = (
                self.table_offset + self._hard_sponge_spawn_offsets[self._mode]
            )
        if self.task == "place_return" and hasattr(self, "placement_initializer"):
            self.placement_initializer.reference_pos = (
                self.table_offset + self._place_return_cube_spawn_offset()
            )
        super(SingleArmPlaceWipe, self)._reset_internal()
        self._reset_scene()

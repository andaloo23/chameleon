# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
Tensor-based grasp detection for Isaac Lab.

Detects grasp when both fingertips are inside their assigned OBB face zones
for frames_to_grasp consecutive frames.
"""

from __future__ import annotations

import torch
from torch import Tensor


class GraspDetectorTensor:
    """
    GPU-batched grasp detection based on fingertip zone occupancy.

    Detects grasp when:
    - Left fingertip is inside the left OBB face zone
    - Right fingertip is inside the right OBB face zone
    - Both conditions hold for frames_to_grasp consecutive frames

    Detects drop when:
    - Either fingertip leaves its zone for frames_to_drop consecutive frames
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        frames_to_grasp: int = 3,
        frames_to_drop: int = 5,
        lift_threshold: float = 0.025,
        # Legacy params kept for API compatibility (unused in new logic)
        history_len: int = 10,
        stall_frames: int = 5,
        stall_threshold: float = 0.001,
        following_threshold: float = 0.03,
        near_cube_threshold: float = 0.12,
        close_command_threshold: float = 0.6,
    ):
        self.num_envs = num_envs
        self.device = device
        self.frames_to_grasp = frames_to_grasp
        self.frames_to_drop = frames_to_drop
        self.lift_threshold = lift_threshold

        # Persistent state tensors
        self._init_state_tensors()

    def _init_state_tensors(self):
        """Initialize all state tracking tensors."""
        # Frame counters for temporal filtering
        self.in_zone_frames = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.out_of_zone_frames = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Main grasp state
        self.is_grasped = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        # Droppable and in-cup detection (for reward computation)
        self.is_droppable = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.is_in_cup = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def reset(self, env_ids: Tensor | None = None):
        """Reset state for specified environments."""
        if env_ids is None:
            self._init_state_tensors()
        else:
            self.in_zone_frames[env_ids] = 0
            self.out_of_zone_frames[env_ids] = 0
            self.is_grasped[env_ids] = False
            self.is_droppable[env_ids] = False
            self.is_in_cup[env_ids] = False

    def update(
        self,
        left_in_zone: Tensor,
        right_in_zone: Tensor,
        cube_pos: Tensor,
        cup_pos: Tensor,
        cup_height: Tensor,
        cup_inner_radius: float,
        cube_half_size: Tensor,
        droppable_min_height: float = 0.005,
        in_cup_height_margin: float = 0.02,
        droppable_xy_radius: float | None = None,
        # Legacy params kept for API compatibility (unused)
        gripper_value: Tensor | None = None,
        target_gripper: Tensor | None = None,
        gripper_pos: Tensor | None = None,
        jaw_pos: Tensor | None = None,
    ) -> Tensor:
        """
        Update grasp detection state for all environments.

        Grasp condition: both fingertips inside their OBB face zones for N frames.
        Drop condition:  either fingertip leaves its zone for M frames.

        Args:
            left_in_zone:        Bool [num_envs] — fixed jaw tip inside left face zone
            right_in_zone:       Bool [num_envs] — moving jaw tip inside right face zone
            cube_pos:            [num_envs, 3]
            cup_pos:             [num_envs, 3]
            cup_height:          [num_envs] per-env cup height
            cup_inner_radius:    scalar — physical cup opening (used for is_in_cup XY check)
            cube_half_size:      scalar
            droppable_min_height: scalar
            in_cup_height_margin: scalar
            droppable_xy_radius: scalar or None — XY tolerance for is_droppable (learning milestone).
                                 Defaults to cup_inner_radius if None. Set larger than cup_inner_radius
                                 so "droppable" fires before the precise in-cup XY is reached.

        Returns:
            is_grasped: Boolean tensor [num_envs]
        """
        cube_z = cube_pos[:, 2]

        # --- Droppable / in-cup detection ---
        cube_xy = cube_pos[:, :2]
        cup_xy = cup_pos[:, :2]
        cube_cup_xy_dist = torch.norm(cube_xy - cup_xy, dim=1)
        cup_top_z = cup_pos[:, 2] + cup_height
        cube_bottom_z = cube_z - cube_half_size

        # is_droppable: learning milestone — cube is above cup height, within a generous XY radius.
        # Uses droppable_xy_radius (wider) so the robot gets rewarded for lifting high
        # before it achieves precise XY alignment.
        _droppable_xy_r = droppable_xy_radius if droppable_xy_radius is not None else cup_inner_radius
        droppable_xy_ok = cube_cup_xy_dist <= _droppable_xy_r
        above_cup = cube_bottom_z >= (cup_top_z + droppable_min_height)
        self.is_droppable = droppable_xy_ok & above_cup

        # is_in_cup: success condition — cube physically inside the cup (tight XY = cup_inner_radius).
        cup_bottom_z = cup_pos[:, 2]
        xy_in_cup = cube_cup_xy_dist <= cup_inner_radius
        cube_inside_height = (cube_bottom_z >= (cup_bottom_z - in_cup_height_margin)) & (cube_bottom_z <= cup_top_z)
        self.is_in_cup = xy_in_cup & cube_inside_height

        # --- Grasp entry: both tips in zone for frames_to_grasp frames ---
        both_in_zone = left_in_zone & right_in_zone

        self.in_zone_frames = torch.where(
            ~self.is_grasped & both_in_zone,
            self.in_zone_frames + 1,
            torch.zeros_like(self.in_zone_frames),
        )
        new_grasps = ~self.is_grasped & (self.in_zone_frames >= self.frames_to_grasp)
        self.is_grasped = self.is_grasped | new_grasps
        self.in_zone_frames = torch.where(new_grasps, torch.zeros_like(self.in_zone_frames), self.in_zone_frames)

        # --- Drop: either tip leaves zone for frames_to_drop frames ---
        lost_zone = ~both_in_zone
        self.out_of_zone_frames = torch.where(
            self.is_grasped & lost_zone,
            self.out_of_zone_frames + 1,
            torch.zeros_like(self.out_of_zone_frames),
        )
        drops = self.is_grasped & (self.out_of_zone_frames >= self.frames_to_drop)
        self.is_grasped = self.is_grasped & ~drops
        self.out_of_zone_frames = torch.where(drops, torch.zeros_like(self.out_of_zone_frames), self.out_of_zone_frames)

        return self.is_grasped

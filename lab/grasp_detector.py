# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
Tensor-based grasp detection for Isaac Lab.

Ports the behavioral grasp detection from gripper.py to batched tensor operations
for parallel environment execution.
"""

from __future__ import annotations

import torch
from torch import Tensor


class GraspDetectorTensor:
    """
    GPU-batched grasp detection based on behavioral signals.
    
    Detects grasp when:
    - Gripper is actively closing and stalled (blocked by object)
    - Cube is lifted above ground threshold
    - Cube distance to gripper is stable (following) for N frames
    
    Detects drop when:
    - Cube distance is no longer stable for M frames
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        history_len: int = 10,
        stall_frames: int = 5,
        frames_to_grasp: int = 15,
        frames_to_drop: int = 30,
        stall_threshold: float = 0.001,
        following_threshold: float = 0.0005,
        lift_threshold: float = 0.025,
    ):
        """
        Initialize the grasp detector.
        
        Args:
            num_envs: Number of parallel environments
            device: Torch device ("cuda" or "cpu")
            history_len: Number of frames to track for following detection
            stall_frames: Number of frames to consider gripper stalled
            frames_to_grasp: Consecutive following frames to confirm grasp
            frames_to_drop: Consecutive not-following frames to confirm drop
            stall_threshold: Max joint position change to be considered stalled
            following_threshold: Max distance variation to be considered following
            lift_threshold: Height above which cube is considered lifted
        """
        self.num_envs = num_envs
        self.device = device
        self.history_len = history_len
        self.stall_frames = stall_frames
        self.frames_to_grasp = frames_to_grasp
        self.frames_to_drop = frames_to_drop
        self.stall_threshold = stall_threshold
        self.following_threshold = following_threshold
        self.lift_threshold = lift_threshold
        
        # Persistent state tensors
        self._init_state_tensors()
    
    def _init_state_tensors(self):
        """Initialize all state tracking tensors."""
        # Distance history for following detection: [num_envs, history_len]
        self.dist_history = torch.zeros(
            self.num_envs, self.history_len, device=self.device, dtype=torch.float32
        )
        self.dist_history_idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.dist_history_filled = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        
        # Gripper value history for stall detection: [num_envs, stall_frames]
        self.gripper_history = torch.zeros(
            self.num_envs, self.stall_frames, device=self.device, dtype=torch.float32
        )
        self.gripper_history_idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.gripper_history_filled = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        
        # Frame counters for temporal filtering
        self.following_frames = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.not_following_frames = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        
        # Previous target for closing intent detection
        self.prev_target_gripper = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.closing_intent = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
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
            self.dist_history[env_ids] = 0.0
            self.dist_history_idx[env_ids] = 0
            self.dist_history_filled[env_ids] = False
            self.gripper_history[env_ids] = 0.0
            self.gripper_history_idx[env_ids] = 0
            self.gripper_history_filled[env_ids] = False
            self.following_frames[env_ids] = 0
            self.not_following_frames[env_ids] = 0
            self.prev_target_gripper[env_ids] = 0.0
            self.closing_intent[env_ids] = False
            self.is_grasped[env_ids] = False
            self.is_droppable[env_ids] = False
            self.is_in_cup[env_ids] = False

    def update(
        self,
        gripper_value: Tensor,
        target_gripper: Tensor,
        gripper_pos: Tensor,
        cube_pos: Tensor,
        cup_pos: Tensor,
        cup_height: float,
        cup_inner_radius: float,
        cube_half_size: float,
        droppable_min_height: float = 0.005,
        in_cup_height_margin: float = 0.02,
    ) -> Tensor:
        """
        Update grasp detection state for all environments.
        
        Args:
            gripper_value: Current gripper joint position [num_envs]
            target_gripper: Target gripper position from action [num_envs]
            gripper_pos: Gripper world position [num_envs, 3]
            cube_pos: Cube world position [num_envs, 3]
            cup_pos: Cup world position [num_envs, 3]
            cup_height: Cup height (scalar)
            cup_inner_radius: Cup inner radius (scalar)
            cube_half_size: Half of cube side length (scalar)
            droppable_min_height: Min height above cup to be droppable (scalar)
            in_cup_height_margin: Height tolerance for in-cup detection (scalar)
        
        Returns:
            is_grasped: Boolean tensor [num_envs] indicating grasp state
        """
        # 1. Update closing intent based on target changes
        target_decreasing = target_gripper < (self.prev_target_gripper - 1e-4)
        target_increasing = target_gripper > (self.prev_target_gripper + 1e-4)
        self.closing_intent = torch.where(target_decreasing, torch.ones_like(self.closing_intent), self.closing_intent)
        self.closing_intent = torch.where(target_increasing, torch.zeros_like(self.closing_intent), self.closing_intent)
        self.prev_target_gripper = target_gripper.clone()
        
        # 2. Update gripper history and check stall
        idx = self.gripper_history_idx % self.stall_frames
        self.gripper_history.scatter_(1, idx.unsqueeze(1), gripper_value.unsqueeze(1))
        self.gripper_history_idx = (self.gripper_history_idx + 1) % self.stall_frames
        self.gripper_history_filled = self.gripper_history_filled | (self.gripper_history_idx == 0)
        
        # Stalled = max - min < threshold (only if history is filled)
        gripper_max = self.gripper_history.max(dim=1).values
        gripper_min = self.gripper_history.min(dim=1).values
        stalled = self.gripper_history_filled & ((gripper_max - gripper_min) < self.stall_threshold)
        
        # Closed = actively closing AND stalled
        closed = self.closing_intent & stalled
        
        # 3. Check if cube is lifted
        cube_z = cube_pos[:, 2]
        lifted = cube_z > self.lift_threshold
        
        # 4. Update distance history and check following
        curr_dist = torch.norm(gripper_pos - cube_pos, dim=1)
        idx = self.dist_history_idx % self.history_len
        self.dist_history.scatter_(1, idx.unsqueeze(1), curr_dist.unsqueeze(1))
        self.dist_history_idx = (self.dist_history_idx + 1) % self.history_len
        self.dist_history_filled = self.dist_history_filled | (self.dist_history_idx == 0)
        
        # Following = max - min distance < threshold (only if history is filled)
        dist_max = self.dist_history.max(dim=1).values
        dist_min = self.dist_history.min(dim=1).values
        following = self.dist_history_filled & ((dist_max - dist_min) < self.following_threshold)
        
        # 5. Droppable detection: cube is above cup and aligned XY
        cube_xy = cube_pos[:, :2]
        cup_xy = cup_pos[:, :2]
        cube_cup_xy_dist = torch.norm(cube_xy - cup_xy, dim=1)
        cup_top_z = cup_pos[:, 2] + cup_height
        cube_bottom_z = cube_z - cube_half_size
        
        xy_in_range = cube_cup_xy_dist <= cup_inner_radius
        above_cup = cube_bottom_z >= (cup_top_z + droppable_min_height)
        self.is_droppable = xy_in_range & above_cup
        
        # 6. In-cup detection: cube XY aligned AND cube bottom inside cup height
        cup_bottom_z = cup_pos[:, 2]
        cube_inside_height = (cube_bottom_z >= (cup_bottom_z - in_cup_height_margin)) & (cube_bottom_z <= cup_top_z)
        self.is_in_cup = xy_in_range & cube_inside_height
        
        # 7. Apply grasp detection logic
        # If not grasped: need actively closing + lifted + following for N frames
        grasp_condition = closed & lifted & following
        
        # Increment following_frames where condition met, reset otherwise
        self.following_frames = torch.where(
            ~self.is_grasped & grasp_condition,
            self.following_frames + 1,
            torch.zeros_like(self.following_frames)
        )
        
        # Transition to grasped when following_frames >= frames_to_grasp
        new_grasps = ~self.is_grasped & (self.following_frames >= self.frames_to_grasp)
        self.is_grasped = self.is_grasped | new_grasps
        self.following_frames = torch.where(new_grasps, torch.zeros_like(self.following_frames), self.following_frames)
        
        # If grasped: lose grasp if actively opening OR (not following AND not lifted)
        # This prevents toggling "Released" messages if the cube slips during transport
        opening_intent = target_increasing
        lost_grasp_condition = opening_intent | (~following & ~lifted)
        
        self.not_following_frames = torch.where(
            self.is_grasped & lost_grasp_condition,
            self.not_following_frames + 1,
            torch.zeros_like(self.not_following_frames)
        )
        
        # Transition to not grasped when not_following_frames >= frames_to_drop
        drops = self.is_grasped & (self.not_following_frames >= self.frames_to_drop)
        self.is_grasped = self.is_grasped & ~drops
        self.not_following_frames = torch.where(drops, torch.zeros_like(self.not_following_frames), self.not_following_frames)
        
        return self.is_grasped

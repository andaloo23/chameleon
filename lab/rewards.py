# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
JIT-compiled reward functions for Isaac Lab pick-and-place task.

Ports the 5-stage reward system from reward_engine.py to torch.jit.script
functions for GPU-accelerated parallel computation.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.jit.script
def compute_approach_reward(
    gripper_pos: Tensor,
    cube_pos: Tensor,
    prev_gripper_cube_dist: Tensor,
    approach_weight: float,
) -> tuple[Tensor, Tensor]:
    """
    Compute delta-based approach shaping reward.
    
    Reward = weight * max(0, prev_dist - curr_dist)
    Only rewards getting closer, no penalty for moving away.
    
    Args:
        gripper_pos: [num_envs, 3] gripper world position
        cube_pos: [num_envs, 3] cube world position
        prev_gripper_cube_dist: [num_envs] previous distance
        approach_weight: Reward weight per meter closer
    
    Returns:
        reward: [num_envs] approach shaping reward
        curr_dist: [num_envs] current distance (to cache for next step)
    """
    curr_dist = torch.norm(gripper_pos - cube_pos, dim=1)
    delta = prev_gripper_cube_dist - curr_dist
    reward = approach_weight * torch.clamp(delta, min=0.0)
    return reward, curr_dist


@torch.jit.script
def compute_transport_reward(
    cube_pos: Tensor,
    cup_pos: Tensor,
    is_grasped: Tensor,
    transport_weight: float,
    transport_distance_max: float,
) -> Tensor:
    """
    Compute transport shaping reward (distance to cup).
    
    Only active after grasping. Reward increases as cube XY distance to cup decreases.
    
    Args:
        cube_pos: [num_envs, 3] cube world position
        cup_pos: [num_envs, 3] cup world position
        is_grasped: [num_envs] boolean grasp state
        transport_weight: Base reward weight
        transport_distance_max: Max distance for normalization
    
    Returns:
        reward: [num_envs] transport shaping reward
    """
    cube_xy = cube_pos[:, :2]
    cup_xy = cup_pos[:, :2]
    dist_xy = torch.norm(cube_xy - cup_xy, dim=1)
    
    # Normalize: 0 at max distance, 1 at cup
    ratio = 1.0 - torch.clamp(dist_xy / transport_distance_max, max=1.0)
    reward = transport_weight * ratio * is_grasped.float()
    return reward


@torch.jit.script
def compute_one_time_bonuses(
    is_grasped: Tensor,
    stage_grasped: Tensor,
    is_lifted: Tensor,
    stage_lifted: Tensor,
    is_droppable: Tensor,
    stage_droppable: Tensor,
    is_in_cup: Tensor,
    stage_success: Tensor,
    grasp_bonus: float,
    lift_bonus: float,
    droppable_bonus: float,
    success_bonus: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute one-time bonuses for stage transitions with latching logic.
    
    Returns:
        grasp_reward, lift_reward, droppable_reward, success_reward: [num_envs] bonuses
        new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success: [num_envs] updated flags
    """
    # Award bonus ONLY if condition is met AND it hasn't been awarded before
    new_grasp = is_grasped & ~stage_grasped
    new_lift = is_lifted & ~stage_lifted
    new_droppable = is_droppable & ~stage_droppable
    new_success = is_in_cup & ~stage_success
    
    grasp_reward = grasp_bonus * new_grasp.float()
    lift_reward = lift_bonus * new_lift.float()
    droppable_reward = droppable_bonus * new_droppable.float()
    success_reward = success_bonus * new_success.float()
    
    # Update latched flags
    new_stage_grasped = stage_grasped | is_grasped
    new_stage_lifted = stage_lifted | is_lifted
    new_stage_droppable = stage_droppable | is_droppable
    new_stage_success = stage_success | is_in_cup
    
    return (grasp_reward, lift_reward, droppable_reward, success_reward,
            new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success)


@torch.jit.script
def compute_penalties(
    joint_vel: Tensor,
    cube_pos: Tensor,
    is_grasped: Tensor,
    stage_grasped: Tensor,
    is_in_cup: Tensor,
    cube_half_size: float,
    action_cost_weight: float,
    drop_penalty: float,
    stage_dropped: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute penalty terms.
    
    Args:
        joint_vel: [num_envs, num_joints] joint velocities
        cube_pos: [num_envs, 3] cube world position
        is_grasped: [num_envs] current grasp state
        stage_grasped: [num_envs] latched flag - was ever grasped this episode
        is_in_cup: [num_envs] current in-cup state
        cube_half_size: Half of cube side length
        action_cost_weight: Weight for action cost penalty
        drop_penalty: One-time penalty for dropping cube not in cup
        stage_dropped: [num_envs] latched flag - was already penalized for drop
    
    Returns:
        action_cost: [num_envs] action cost penalty
        drop_penalty_reward: [num_envs] drop penalty (only on transition)
        new_stage_dropped: [num_envs] updated latched drop flag
    """
    # Action cost: L2 norm of joint velocities
    action_cost = -action_cost_weight * torch.norm(joint_vel, dim=1)
    
    # Drop detection: was grasped, cube low, not currently grasped, not in cup
    cube_z = cube_pos[:, 2]
    low_height = cube_z <= cube_half_size * 1.5  # On ground threshold
    dropped = stage_grasped & ~is_grasped & low_height & ~is_in_cup
    
    # Only apply penalty if not already penalized this episode
    apply_drop_penalty = dropped & ~stage_dropped
    drop_penalty_reward = drop_penalty * apply_drop_penalty.float()
    
    new_stage_dropped = stage_dropped | dropped
    
    return action_cost, drop_penalty_reward, new_stage_dropped


@torch.jit.script
def compute_pick_place_rewards(
    gripper_pos: Tensor,
    cube_pos: Tensor,
    cup_pos: Tensor,
    joint_vel: Tensor,
    prev_gripper_cube_dist: Tensor,
    is_grasped: Tensor,
    is_droppable: Tensor,
    is_in_cup: Tensor,
    stage_grasped: Tensor,
    stage_lifted: Tensor,
    stage_droppable: Tensor,
    stage_success: Tensor,
    stage_dropped: Tensor,
    cube_half_size: float,
    # Reward weights
    approach_weight: float,
    grasp_bonus: float,
    transport_weight: float,
    transport_distance_max: float,
    lift_bonus: float,
    droppable_bonus: float,
    success_bonus: float,
    action_cost_weight: float,
    drop_penalty: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute total reward for pick-and-place task.
    
    Returns:
        total_reward: [num_envs] total reward
        curr_gripper_cube_dist: [num_envs] to cache for next step
        new_stage_grasped: [num_envs] updated latched grasp flag
        action_cost: [num_envs] action cost penalty
        drop_penalty: [num_envs] drop penalty
        new_stage_grasped: [num_envs]
        new_stage_lifted: [num_envs]
        new_stage_droppable: [num_envs]
        new_stage_success: [num_envs]
        new_stage_dropped: [num_envs]
    """
    # Stage 1: Approach shaping
    approach_reward, curr_dist = compute_approach_reward(
        gripper_pos, cube_pos, prev_gripper_cube_dist, approach_weight
    )
    
    # Stage 3: Transport shaping
    transport_reward = compute_transport_reward(
        cube_pos, cup_pos, is_grasped, transport_weight, transport_distance_max
    )
    
    # Stages 2, 4, 5, 6: One-time bonuses
    cube_z = cube_pos[:, 2]
    is_lifted = is_grasped & (cube_z > 0.03)  # Table height threshold
    
    (grasp_reward, lift_reward, droppable_reward, success_reward,
     new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success) = compute_one_time_bonuses(
        is_grasped, stage_grasped,
        is_lifted, stage_lifted,
        is_droppable, stage_droppable,
        is_in_cup, stage_success,
        grasp_bonus, lift_bonus, droppable_bonus, success_bonus
    )
    
    # Penalties
    action_cost, drop_penalty_reward, new_stage_dropped = compute_penalties(
        joint_vel, cube_pos, is_grasped, stage_grasped,
        is_in_cup, cube_half_size, action_cost_weight, drop_penalty,
        stage_dropped
    )
    
    # Total reward
    total_reward = (
        approach_reward +
        grasp_reward +
        lift_reward +
        transport_reward +
        droppable_reward +
        success_reward +
        action_cost +
        drop_penalty_reward
    )
    
    return (total_reward, curr_dist, 
            new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success, new_stage_dropped,
            action_cost, drop_penalty_reward)

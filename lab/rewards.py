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
    stage_grasped: Tensor,
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
    
    # Only award if NOT yet grasped
    reward = approach_weight * torch.clamp(delta, min=0.0) * (~stage_grasped).float()
    return reward, curr_dist


@torch.jit.script
def compute_lift_shaping_delta(
    cube_height: Tensor,
    prev_cube_height: Tensor,
    is_grasped: Tensor,
    lift_weight: float,
) -> tuple[Tensor, Tensor]:
    """
    Compute delta-based lift shaping reward.
    
    Reward = weight * max(0, curr_height - prev_height)
    Only active after grasping.
    """
    delta = cube_height - prev_cube_height
    reward = lift_weight * torch.clamp(delta, min=0.0) * is_grasped.float()
    return reward, cube_height


@torch.jit.script
def compute_transport_shaping_3d(
    cube_pos: Tensor,
    cup_pos: Tensor,
    cup_height: float,
    cube_half_size: float,
    prev_transport_dist: Tensor,
    is_grasped: Tensor,
    transport_weight: float,
) -> tuple[Tensor, Tensor]:
    """
    Compute 3D transport shaping reward (delta-based).
    
    Target Z is cup_top + 2cm. Cube Z is based on bottom face.
    Distance d = sqrt(dx^2 + dy^2 + 0.3 * dz^2)
    Reward = w * (d_prev - d_curr)
    """
    # Target Z: top of cup + 2cm margin
    z_target = cup_pos[:, 2] + cup_height + 0.02
    z_bottom = cube_pos[:, 2] - cube_half_size
    
    dx = cube_pos[:, 0] - cup_pos[:, 0]
    dy = cube_pos[:, 1] - cup_pos[:, 1]
    dz = z_bottom - z_target
    
    # 3D distance with Z-weighting (0.3)
    curr_dist = torch.sqrt(dx**2 + dy**2 + 0.3 * dz**2)
    
    # Delta-based reward: positive if getting closer
    delta = prev_transport_dist - curr_dist
    
    # Only award if CURRENTLY grasped
    reward = transport_weight * torch.clamp(delta, min=0.0) * is_grasped.float()
    
    return reward, curr_dist




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
def compute_fingertip_obb_reach_reward(
    gripper_tip_pos: Tensor,
    jaw_tip_pos: Tensor,
    cube_pos: Tensor,
    cube_half_size: float,
    zone_margin: float,
    prev_right_tip_dist: Tensor,
    prev_left_tip_dist: Tensor,
    stage_grasped: Tensor,
    cube_quat_w: Tensor,
    use_x: Tensor,
    fingertip_obb_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the pre-grasp fingertip reaching reward (delta-based).

    For each fingertip, measure how far outside its NEAREST cube-face
    zone slab it is (0 when inside/touching). Use delta shaping:
      reward += w * max(d_prev - d_curr, 0)  per fingertip

    Each tip independently targets whichever face zone is closer,
    so the reward works regardless of wrist orientation.

    Zero once stage_grasped latches.

    Returns:
        reach_reward:       [num_envs]
        new_right_tip_dist: [num_envs] (cache for next step)
        new_left_tip_dist:  [num_envs] (cache for next step)
    """
    not_grasped = (~stage_grasped).float()

    # --- Build rotation matrix R from cube quaternion (w, x, y, z) ---
    w = cube_quat_w[:, 0]
    x = cube_quat_w[:, 1]
    y = cube_quat_w[:, 2]
    z = cube_quat_w[:, 3]
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ], dim=-1),
        torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ], dim=-1),
        torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=-1),
    ], dim=1)  # [N, 3, 3]
    R_inv = R.transpose(1, 2)  # [N, 3, 3]

    # --- Local axis from use_x ---
    axis_local = torch.zeros_like(cube_pos)           # [N, 3]
    axis_local[:, 0] = use_x.float()                  # x=1 if use_x
    axis_local[:, 1] = (~use_x).float()               # y=1 if not use_x

    # --- Zone slab centers for +side and -side faces ---
    t = zone_margin                                    # zone protrusion thickness
    face_offset = cube_half_size + 0.5 * t             # center of zone slab

    r_local_pos = face_offset * axis_local             # [N, 3] center of +face zone
    r_local_neg = -face_offset * axis_local            # [N, 3] center of -face zone

    # --- Half-extents: tangent dirs = cube_half_size, normal dir = t/2 ---
    h = torch.full_like(axis_local, cube_half_size)    # [N, 3]
    half_t = t / 2.0
    h[:, 0] = torch.where(use_x,  torch.tensor(half_t, device=h.device), h[:, 0])
    h[:, 1] = torch.where(~use_x, torch.tensor(half_t, device=h.device), h[:, 1])

    # --- Transform tips to cube-local space ---
    q_gripper = torch.bmm(R_inv, (gripper_tip_pos - cube_pos).unsqueeze(-1)).squeeze(-1)  # [N, 3]
    q_jaw     = torch.bmm(R_inv, (jaw_tip_pos     - cube_pos).unsqueeze(-1)).squeeze(-1)  # [N, 3]

    # --- Distance from each tip to BOTH face zones, take the min ---
    d_gripper_pos = torch.linalg.norm(torch.clamp((q_gripper - r_local_pos).abs() - h, min=0.0), dim=1)
    d_gripper_neg = torch.linalg.norm(torch.clamp((q_gripper - r_local_neg).abs() - h, min=0.0), dim=1)
    d_left = torch.minimum(d_gripper_pos, d_gripper_neg)  # gripper tip → nearest face

    d_jaw_pos = torch.linalg.norm(torch.clamp((q_jaw - r_local_pos).abs() - h, min=0.0), dim=1)
    d_jaw_neg = torch.linalg.norm(torch.clamp((q_jaw - r_local_neg).abs() - h, min=0.0), dim=1)
    d_right = torch.minimum(d_jaw_pos, d_jaw_neg)  # jaw tip → nearest face

    # Delta reward: reward for closing the gap, zero penalty for moving away
    delta_left  = torch.clamp(prev_left_tip_dist  - d_left,  min=0.0)
    delta_right = torch.clamp(prev_right_tip_dist - d_right, min=0.0)

    reach_reward = fingertip_obb_weight * (delta_left + delta_right) * not_grasped

    return reach_reward, d_right, d_left


@torch.jit.script
def compute_pick_place_rewards(
    gripper_pos: Tensor,
    cube_pos: Tensor,
    cup_pos: Tensor,
    joint_vel: Tensor,
    prev_gripper_cube_dist: Tensor,
    prev_transport_dist: Tensor,
    prev_cube_z: Tensor,
    is_grasped: Tensor,
    is_droppable: Tensor,
    is_in_cup: Tensor,
    stage_grasped: Tensor,
    stage_lifted: Tensor,
    stage_droppable: Tensor,
    stage_success: Tensor,
    stage_dropped: Tensor,
    cup_height: float,
    cube_half_size: float,
    zone_margin: float,
    # Fingertip OBB inputs
    gripper_tip_pos: Tensor,
    jaw_tip_pos: Tensor,
    cube_quat_w: Tensor,
    use_x: Tensor,
    prev_right_tip_dist: Tensor,
    prev_left_tip_dist: Tensor,
    # Reward weights
    approach_weight: float,
    grasp_bonus: float,
    transport_weight: float,
    lift_bonus: float,
    droppable_bonus: float,
    success_bonus: float,
    lift_shaping_weight: float,
    action_cost_weight: float,
    drop_penalty: float,
    fingertip_obb_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute total reward for pick-and-place task.

    Returns:
        total_reward:         [num_envs]
        curr_gripper_cube_dist, curr_transport_dist, curr_cube_z: cached distances
        new_stage_*:          updated latched flags
        action_cost, drop_penalty_reward: per-env penalty tensors
        new_right_tip_dist, new_left_tip_dist: cached fingertip OBB distances
    """
    # Stage 1: Approach shaping
    approach_reward, curr_dist = compute_approach_reward(
        gripper_pos, cube_pos, prev_gripper_cube_dist, stage_grasped, approach_weight
    )

    # Pre-grasp: Fingertip OBB reaching (delta-based)
    fingertip_reach_reward, new_right_tip_dist, new_left_tip_dist = compute_fingertip_obb_reach_reward(
        gripper_tip_pos=gripper_tip_pos,
        jaw_tip_pos=jaw_tip_pos,
        cube_pos=cube_pos,
        cube_half_size=cube_half_size,
        zone_margin=zone_margin,
        prev_right_tip_dist=prev_right_tip_dist,
        prev_left_tip_dist=prev_left_tip_dist,
        stage_grasped=stage_grasped,
        cube_quat_w=cube_quat_w,
        use_x=use_x,
        fingertip_obb_weight=fingertip_obb_weight,
    )

    # Stage 3: Unified 3D Transport shaping
    transport_reward, curr_transport_dist = compute_transport_shaping_3d(
        cube_pos, cup_pos, cup_height, cube_half_size,
        prev_transport_dist, is_grasped, transport_weight
    )

    # Stage 3: Lift shaping (delta-based)
    cube_z = cube_pos[:, 2]
    lift_shaping_reward, curr_cube_z = compute_lift_shaping_delta(
        cube_z, prev_cube_z, is_grasped, lift_shaping_weight
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
        fingertip_reach_reward +
        grasp_reward +
        lift_reward +
        lift_shaping_reward +
        transport_reward +
        droppable_reward +
        success_reward +
        action_cost +
        drop_penalty_reward
    )

    return (total_reward, curr_dist, curr_transport_dist, curr_cube_z,
            new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success, new_stage_dropped,
            action_cost, drop_penalty_reward,
            new_right_tip_dist, new_left_tip_dist)


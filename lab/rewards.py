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
    best_axis: Tensor,
    left_is_positive: Tensor,
    cube_half_size: float,
    zone_margin: float,
    reach_dist: Tensor,
    prev_right_tip_dist: Tensor,
    prev_left_tip_dist: Tensor,
    stage_grasped: Tensor,
    cube_quat_w: Tensor,
    fingertip_obb_weight: float,
    straddle_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the pre-grasp fingertip reaching reward.

    Two components, both zero once stage_grasped latches:

    1. OBB distance reward:
       For each fingertip, measure how far outside its designated cube-face
       slab it is (0 when inside/touching). Use delta shaping:
         reward += w * max(d_prev - d_curr, 0)  per fingertip

    2. Straddle reward:
       Project fingertips onto the pinch axis. When the span between them
       matches the cube diameter, reward = straddle_weight; tapers with
       a Gaussian (sigma = cube_half_size * 0.75).

    Left fingertip = jaw_tip  (moving jaw)
    Right fingertip = gripper_tip  (fixed jaw)

    Returns:
        reach_reward:       [num_envs]
        new_right_tip_dist: [num_envs] (cache for next step)
        new_left_tip_dist:  [num_envs] (cache for next step)
        reach_gate:         [num_envs] (for metrics)
        d_right:            [num_envs] (for metrics)
        d_left:             [num_envs] (for metrics)
    """
    not_grasped = (~stage_grasped).float()

    # --- Per-fingertip face-slab distance ---
    # best_axis points from cube center toward the +normal face.
    # left_is_positive == True  => left face is +normal side
    #                              right face is -normal side
    #   left (gripper) target: left face center
    #   right (jaw) target: right face center

    # Signed projection of each tip onto the pinch axis, relative to cube center
    tip_to_cube_gripper = gripper_tip_pos - cube_pos  # [N,3] -> Left tip
    tip_to_cube_jaw     = jaw_tip_pos     - cube_pos  # [N,3] -> Right tip

    proj_gripper = (tip_to_cube_gripper * best_axis).sum(dim=-1)  # [N]
    proj_jaw     = (tip_to_cube_jaw     * best_axis).sum(dim=-1)  # [N]

    # Target projection for each fingertip:
    #   left (gripper) tip  -> left face  -> +half if left_is_positive, else -half
    #   right (jaw) -> right face -> -half if left_is_positive, else +half
    sign_left  = torch.where(left_is_positive,
                             torch.ones_like(proj_gripper), -torch.ones_like(proj_gripper))
    sign_right = -sign_left

    target_proj_gripper = sign_left  * cube_half_size  # [N]
    target_proj_jaw     = sign_right * cube_half_size  # [N]

    # --- Build rotation matrix R from cube quaternion (w, x, y, z) ---
    w = cube_quat_w[:, 0]  # [N]
    x = cube_quat_w[:, 1]
    y = cube_quat_w[:, 2]
    z = cube_quat_w[:, 3]
    # Rotation matrix columns (cube local -> world)
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ], dim=-1),
        torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ], dim=-1),
        torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=-1),
    ], dim=1)  # [N, 3, 3]

    # R^T transforms world -> cube local
    R_inv = R.transpose(1, 2)  # [N, 3, 3]

    # --- Determine face patch center and half-extents in cube local frame ---
    # best_axis in world frame tells us which local axis is the pinch axis.
    # We recover the local pinch direction by rotating best_axis back to local.
    local_pinch = torch.bmm(R_inv, best_axis.unsqueeze(-1)).squeeze(-1)  # [N, 3]
    # The dominant component of local_pinch tells us which axis (X or Y)
    # Use sign_left to determine the direction

    # Left face center in local: sign_left * cube_half_size along the pinch axis
    r_local_left  = sign_left.unsqueeze(-1) * cube_half_size * local_pinch   # [N, 3]
    r_local_right = sign_right.unsqueeze(-1) * cube_half_size * local_pinch  # [N, 3]

    # Face half-extents: full cube_half_size on the two tangent axes, 0 on the normal axis
    # h = cube_half_size * (1 - |local_pinch|)  (0 on the pinch axis, half_size on others)
    abs_pinch = local_pinch.abs()
    # Clamp to create a clean mask: 1 on pinch axis, 0 on tangent axes
    pinch_mask = (abs_pinch > 0.5).float()
    h = cube_half_size * (1.0 - pinch_mask)  # [N, 3]

    # --- Box-distance formula for each fingertip ---
    # Transform fingertip to cube local frame
    q_left  = torch.bmm(R_inv, (gripper_tip_pos - cube_pos).unsqueeze(-1)).squeeze(-1)  # [N, 3]
    q_right = torch.bmm(R_inv, (jaw_tip_pos     - cube_pos).unsqueeze(-1)).squeeze(-1)  # [N, 3]

    # Signed distance to patch, clamped to 0 inside
    d_left_3d  = torch.clamp((q_left  - r_local_left).abs()  - h, min=0.0)  # [N, 3]
    d_right_3d = torch.clamp((q_right - r_local_right).abs() - h, min=0.0)  # [N, 3]

    d_left  = torch.linalg.norm(d_left_3d,  dim=1)  # [N]
    d_right = torch.linalg.norm(d_right_3d, dim=1)  # [N]

    # Delta reward: reward only for closing the gap
    delta_left  = torch.clamp(prev_left_tip_dist  - d_left,  min=0.0)
    delta_right = torch.clamp(prev_right_tip_dist - d_right, min=0.0)

    # Base shaping components
    obb_reward_base = fingertip_obb_weight * (delta_left + delta_right)

    # Absolute distance reward base
    w_abs = 1.0
    sigma_abs = 0.01
    d = d_left + d_right
    abs_reward_base = w_abs * torch.exp(-d / sigma_abs)
    
    # Reach gating
    sigma_reach = 0.10
    reach_gate = torch.exp(-reach_dist / sigma_reach)
    
    # Gated rewards
    obb_reward = obb_reward_base * reach_gate * not_grasped
    abs_reward = abs_reward_base * reach_gate * not_grasped

    # --- Straddle reward ---
    # Measure signed span of fingertips along the pinch axis
    # Positive when jaw is on left side (+normal) and gripper on right (-normal)
    span = sign_left * (proj_jaw - proj_gripper)  # [N]
    target_width = 2.0 * (cube_half_size + zone_margin)
    sigma = cube_half_size * 0.75
    straddle_base = straddle_weight * torch.exp(
        -0.5 * ((span - target_width) / sigma) ** 2
    )
    
    near = torch.exp(-d / sigma_abs)
    straddle = straddle_base * near * not_grasped

    reach_reward = obb_reward + abs_reward + straddle

    return reach_reward, d_right, d_left, reach_gate


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
    reach_dist: Tensor,
    # New fingertip OBB inputs
    gripper_tip_pos: Tensor,
    jaw_tip_pos: Tensor,
    cube_quat_w: Tensor,
    best_axis: Tensor,
    left_is_positive: Tensor,
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
    straddle_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute total reward for pick-and-place task.

    Returns:
        total_reward:         [num_envs]
        curr_gripper_cube_dist, curr_transport_dist, curr_cube_z: cached distances
        new_stage_*:          updated latched flags
        action_cost, drop_penalty_reward: per-env penalty tensors
        new_right_tip_dist, new_left_tip_dist: cached fingertip OBB distances
        reach_gate:           [num_envs] (for metrics)
        d_right:              [num_envs] (for metrics)
        d_left:               [num_envs] (for metrics)
    """
    # Stage 1: Approach shaping
    approach_reward, curr_dist = compute_approach_reward(
        gripper_pos, cube_pos, prev_gripper_cube_dist, stage_grasped, approach_weight
    )

    # Pre-grasp: Fingertip OBB reaching + straddle
    fingertip_reach_reward, new_right_tip_dist, new_left_tip_dist, reach_gate = compute_fingertip_obb_reach_reward(
        gripper_tip_pos=gripper_tip_pos,
        jaw_tip_pos=jaw_tip_pos,
        cube_pos=cube_pos,
        best_axis=best_axis,
        left_is_positive=left_is_positive,
        cube_half_size=cube_half_size,
        zone_margin=zone_margin,
        reach_dist=reach_dist,
        prev_right_tip_dist=prev_right_tip_dist,
        prev_left_tip_dist=prev_left_tip_dist,
        stage_grasped=stage_grasped,
        cube_quat_w=cube_quat_w,
        fingertip_obb_weight=fingertip_obb_weight,
        straddle_weight=straddle_weight,
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
            new_right_tip_dist, new_left_tip_dist,
            reach_gate)

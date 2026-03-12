# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
JIT-compiled reward functions for Isaac Lab pick-and-place task.
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
    """Delta-based approach shaping (kept for reference, not used in main reward)."""
    curr_dist = torch.norm(gripper_pos - cube_pos, dim=1)
    delta = prev_gripper_cube_dist - curr_dist
    reward = approach_weight * torch.clamp(delta, min=0.0) * (~stage_grasped).float()
    return reward, curr_dist


@torch.jit.script
def compute_lift_shaping_delta(
    cube_height: Tensor,
    prev_cube_height: Tensor,
    is_grasped: Tensor,
    lift_weight: float,
    stage_grasped: Tensor,
    d_avg: Tensor,
    zone_radius: float = 0.06,
) -> tuple[Tensor, Tensor]:
    """Delta-based lift shaping. Active whenever cube was ever grasped this episode,
    or if fingertips are very close to cube (pre-grasp lift shaping).

    Uses stage_grasped (latched) instead of is_grasped so brief zone-exit events
    during lifting (from cube rotation) don't zero out the lift signal.
    """
    delta = cube_height - prev_cube_height
    # Enable lift shaping if latched grasp OR fingertips are very near cube
    is_active = stage_grasped | (d_avg < zone_radius)
    reward = lift_weight * torch.clamp(delta, min=0.0) * is_active.float()
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
    """3D transport shaping (delta-based, post-grasp)."""
    z_target = cup_pos[:, 2] + cup_height + 0.02
    z_bottom = cube_pos[:, 2] - cube_half_size
    dx = cube_pos[:, 0] - cup_pos[:, 0]
    dy = cube_pos[:, 1] - cup_pos[:, 1]
    dz = z_bottom - z_target
    curr_dist = torch.sqrt(dx**2 + dy**2 + 0.3 * dz**2)
    delta = prev_transport_dist - curr_dist
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
    """One-time bonuses for stage transitions with latching logic."""
    new_grasp     = is_grasped    & ~stage_grasped
    new_lift      = is_lifted     & ~stage_lifted
    new_droppable = is_droppable  & ~stage_droppable
    new_success   = is_in_cup     & ~stage_success

    grasp_reward     = grasp_bonus     * new_grasp.float()
    lift_reward      = lift_bonus      * new_lift.float()
    droppable_reward = droppable_bonus * new_droppable.float()
    success_reward   = success_bonus   * new_success.float()

    new_stage_grasped   = stage_grasped   | is_grasped
    new_stage_lifted    = stage_lifted    | is_lifted
    new_stage_droppable = stage_droppable | is_droppable
    new_stage_success   = stage_success   | is_in_cup

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
    """Penalty terms."""
    action_cost = -action_cost_weight * torch.norm(joint_vel, dim=1)
    cube_z = cube_pos[:, 2]
    low_height = cube_z <= cube_half_size * 1.5
    dropped = stage_grasped & ~is_grasped & low_height & ~is_in_cup
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
    stage_grasped: Tensor,
    cube_quat_w: Tensor,
    use_x: Tensor,
    gripper_value: Tensor,
    gripper_close_threshold: float,
    prev_d_avg: Tensor,
    approach_weight: float,
    sigma: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Pre-grasp fingertip reaching reward using delta-based potential shaping.

    Potential function: Φ(d) = exp(-d / σ)
    Reward per step: weight * clamp(Φ(d_old) - Φ(d_new), min=0)

    This only rewards *closing* distance — standing still or retreating gives zero.
    When stage_grasped becomes true, d is set to 0 to pay out remaining potential.

    A per-step gripper-closing bonus is added when very close to encourage grasping.

    Returns:
        reach_reward:   [num_envs] (dense per-step delta reward)
        d_R, d_L:       [num_envs] raw Euclidean distances (for display)
        d_L_pos, d_L_neg, d_R_pos, d_R_neg: [num_envs] debug per-face distances
        new_d_avg:      [num_envs] current d_avg for state tracking
    """

    # --- Rotation matrix from cube quaternion (w, x, y, z) ---
    w = cube_quat_w[:, 0]
    x = cube_quat_w[:, 1]
    y = cube_quat_w[:, 2]
    z = cube_quat_w[:, 3]
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ], dim=-1),
        torch.stack([2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ], dim=-1),
        torch.stack([2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)], dim=-1),
    ], dim=1)
    R_inv = R.transpose(1, 2)

    # --- Local face axis ---
    axis_local = torch.zeros_like(cube_pos)
    axis_local[:, 0] = use_x.float()
    axis_local[:, 1] = (~use_x).float()

    # --- Zone slab centers in cube-local space ---
    t = zone_margin
    face_offset = cube_half_size + 0.5 * t
    r_local_pos = face_offset * axis_local
    r_local_neg = -face_offset * axis_local

    # --- Tips in cube-local space ---
    q_L = torch.bmm(R_inv, (gripper_tip_pos - cube_pos).unsqueeze(-1)).squeeze(-1)
    q_R = torch.bmm(R_inv, (jaw_tip_pos     - cube_pos).unsqueeze(-1)).squeeze(-1)

    # --- Euclidean distances to each face zone center ---
    d_L_pos = torch.norm(q_L - r_local_pos, dim=1)
    d_L_neg = torch.norm(q_L - r_local_neg, dim=1)
    d_R_pos = torch.norm(q_R - r_local_pos, dim=1)
    d_R_neg = torch.norm(q_R - r_local_neg, dim=1)

    # --- Optimal assignment (min total distance) ---
    sumA = d_L_pos + d_R_neg   # Case A: L->+face, R->-face
    sumB = d_L_neg + d_R_pos   # Case B: L->-face, R->+face
    use_A = (sumA <= sumB)
    d_L = torch.where(use_A, d_L_pos, d_L_neg)
    d_R = torch.where(use_A, d_R_neg, d_R_pos)

    # --- Compute d_avg ---
    d_avg_real = 0.5 * (d_L + d_R)

    # If grasped, pretend the fingers are perfectly touching (d = 0.0)
    # This pays out remaining potential and ensures Grasping > Hovering.
    d_avg = torch.where(stage_grasped, torch.zeros_like(d_avg_real), d_avg_real)

    # --- Delta-based potential shaping (unclamped) ---
    # Φ(d) = exp(-d / σ), higher when closer
    phi_old = torch.exp(-prev_d_avg / sigma)
    phi_new = torch.exp(-d_avg / sigma)
    # Raw delta: positive for approach, negative for retreat (proper shaping)
    delta_phi = phi_new - phi_old
    r_approach = approach_weight * delta_phi

    # --- Gripper closing near cube (per-step incentive) ---
    gripper_closing = (gripper_value < gripper_close_threshold).float()
    r_grip_close = 2.0 * (d_avg < 0.05).float() * gripper_closing

    reach_reward = r_approach + r_grip_close

    return reach_reward, d_R, d_L, d_L_pos, d_L_neg, d_R_pos, d_R_neg, d_avg


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
    # Fingertip inputs
    gripper_tip_pos: Tensor,
    jaw_tip_pos: Tensor,
    cube_quat_w: Tensor,
    use_x: Tensor,
    gripper_value: Tensor,
    gripper_close_threshold: float,
    # Delta-based approach state
    prev_d_avg: Tensor,
    approach_weight: float,
    approach_sigma: float,
    # Reward weights
    grasp_bonus: float,
    transport_weight: float,
    lift_bonus: float,
    droppable_bonus: float,
    success_bonus: float,
    lift_shaping_weight: float,
    action_cost_weight: float,
    drop_penalty: float,
    grasp_hold_weight: float,
    height_bonus_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Total reward for pick-and-place.

    Pre-grasp: delta-based potential approach shaping + gripper-closing bonus.
    Post-grasp: grasp hold + lift shaping + height bonus + transport shaping + one-time bonuses.

    Returns (18 tensors):
        total_reward
        curr_dist, curr_transport_dist, curr_cube_z
        new_stage_grasped/lifted/droppable/success/dropped
        action_cost, drop_penalty_reward
        new_right_tip_dist, new_left_tip_dist  (d_R, d_L raw distances)
        d_L_pos, d_L_neg, d_R_pos, d_R_neg    (debug)
        new_d_avg                               (state for next step)
    """
    curr_dist = torch.norm(gripper_pos - cube_pos, dim=1)  # diagnostic only

    # Pre-grasp: delta-based potential approach shaping
    (
        fingertip_reach_reward,
        new_right_tip_dist, new_left_tip_dist,
        d_L_pos, d_L_neg, d_R_pos, d_R_neg,
        new_d_avg,
    ) = compute_fingertip_obb_reach_reward(
        gripper_tip_pos=gripper_tip_pos,
        jaw_tip_pos=jaw_tip_pos,
        cube_pos=cube_pos,
        cube_half_size=cube_half_size,
        zone_margin=zone_margin,
        stage_grasped=stage_grasped,
        cube_quat_w=cube_quat_w,
        use_x=use_x,
        gripper_value=gripper_value,
        gripper_close_threshold=gripper_close_threshold,
        prev_d_avg=prev_d_avg,
        approach_weight=approach_weight,
        sigma=approach_sigma,
    )

    # Post-grasp shaping
    transport_reward, curr_transport_dist = compute_transport_shaping_3d(
        cube_pos, cup_pos, cup_height, cube_half_size,
        prev_transport_dist, is_grasped, transport_weight
    )
    cube_z = cube_pos[:, 2]
    d_avg = 0.5 * (new_right_tip_dist + new_left_tip_dist)
    lift_shaping_reward, curr_cube_z = compute_lift_shaping_delta(
        cube_z, prev_cube_z, is_grasped, lift_shaping_weight, stage_grasped, d_avg
    )

    # Per-step grasp hold reward: incentivizes maintaining grasp, not just triggering it
    grasp_hold_reward = grasp_hold_weight * is_grasped.float()

    # Per-step height bonus while grasped: direct gradient for lifting the cube
    # Dead zone at 2cm (0.5cm above rest) to ignore physics jitter at table level
    height_above_rest = torch.clamp(cube_z - 0.02, min=0.0)
    height_reward = height_bonus_weight * height_above_rest * is_grasped.float()

    # One-time bonuses
    # Use stage_grasped (latched) for is_lifted so brief zone-exit during
    # lifting (cube rotation) doesn't prevent the lift bonus from firing
    is_lifted = stage_grasped & (cube_z > 0.03)
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

    total_reward = (
        fingertip_reach_reward +
        grasp_reward +
        grasp_hold_reward +
        lift_reward +
        lift_shaping_reward +
        height_reward +
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
            d_L_pos, d_L_neg, d_R_pos, d_R_neg,
            new_d_avg)


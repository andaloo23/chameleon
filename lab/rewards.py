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
    """
    Compute delta-based approach shaping reward (kept for reference, not used in main reward).

    Reward = weight * max(0, prev_dist - curr_dist)
    Only rewards getting closer, no penalty for moving away.
    """
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
    """
    Compute one-time bonuses for stage transitions with latching logic.
    """
    new_grasp    = is_grasped    & ~stage_grasped
    new_lift     = is_lifted     & ~stage_lifted
    new_droppable = is_droppable & ~stage_droppable
    new_success  = is_in_cup     & ~stage_success

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
    """
    Compute penalty terms.
    """
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
    prev_right_tip_dist: Tensor,  # best Phi achieved so far this episode (HWM), init 0.0
    prev_left_tip_dist: Tensor,   # best Phi achieved so far this episode (HWM), init 0.0
    stage_grasped: Tensor,
    cube_quat_w: Tensor,
    use_x: Tensor,
    fingertip_obb_weight: float,
    sigma: float,                  # exponential distance scale (meters)
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the pre-grasp fingertip reaching reward.

    Reward = weight * (Phi_L + Phi_R + dPhi_L + dPhi_R) * not_grasped

    Where:
      Phi(d) = exp(-d / sigma)       -- dense exponential proximity (absolute, non-farmable)
      dPhi   = max(0, Phi - best_Phi) -- HWM progress bonus (only on new records, anti-farming)
      best_Phi = max(best_Phi, Phi) each step

    The 'prev' tensors store the best Phi (HIGH-water-mark) seen this episode,
    initialized to 0.0 at episode reset (equivalent to d = +infinity best distance).

    Symmetric Straddle Assignment:
    We evaluate two pairings and pick the one minimising total distance:
      Case A: Fixed Tip -> +Normal Face, Moving Tip -> -Normal Face
      Case B: Fixed Tip -> -Normal Face, Moving Tip -> +Normal Face

    Returns:
        reach_reward:              [num_envs]
        d_R, d_L:                  [num_envs] raw Euclidean distances to face zone centers
        new_right_hw, new_left_hw: [num_envs] updated HWM Phi scores (stored as next prev_*)
        d_L_pos, d_L_neg, d_R_pos, d_R_neg: [num_envs] debug raw distances to each face
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
    axis_local = torch.zeros_like(cube_pos)    # [N, 3]
    axis_local[:, 0] = use_x.float()           # x=1 if use_x
    axis_local[:, 1] = (~use_x).float()        # y=1 if not use_x

    # --- Zone slab centers in cube-local space ---
    t = zone_margin
    face_offset = cube_half_size + 0.5 * t    # center of zone slab

    r_local_pos = face_offset * axis_local     # [N, 3] center of +face zone
    r_local_neg = -face_offset * axis_local    # [N, 3] center of -face zone

    # --- Transform tips to cube-local space ---
    # Fixed Jaw = L, Moving Jaw = R
    q_L = torch.bmm(R_inv, (gripper_tip_pos - cube_pos).unsqueeze(-1)).squeeze(-1)  # [N, 3]
    q_R = torch.bmm(R_inv, (jaw_tip_pos     - cube_pos).unsqueeze(-1)).squeeze(-1)  # [N, 3]

    # --- Euclidean distance from each tip to BOTH face zone centers ---
    # Computed in cube-local space (isometric to world): local norm == world norm.
    d_L_pos = torch.norm(q_L - r_local_pos, dim=1)
    d_L_neg = torch.norm(q_L - r_local_neg, dim=1)
    d_R_pos = torch.norm(q_R - r_local_pos, dim=1)
    d_R_neg = torch.norm(q_R - r_local_neg, dim=1)

    # --- Optimal Assignment (minimise total distance) ---
    # Case A: L -> +face, R -> -face
    sumA = d_L_pos + d_R_neg
    # Case B: L -> -face, R -> +face
    sumB = d_L_neg + d_R_pos

    use_A = (sumA <= sumB)
    d_L = torch.where(use_A, d_L_pos, d_L_neg)   # raw distance for L tip
    d_R = torch.where(use_A, d_R_neg, d_R_pos)   # raw distance for R tip

    # --- Exponential proximity scores: Phi in (0, 1], -> 1 as d -> 0 ---
    Phi_L = torch.exp(-d_L / sigma)
    Phi_R = torch.exp(-d_R / sigma)

    # --- High-water-mark progress: only reward NEW proximity records ---
    # prev_* = best Phi achieved this episode (init 0.0 at reset)
    delta_L = torch.clamp(Phi_L - prev_left_tip_dist,  min=0.0)
    delta_R = torch.clamp(Phi_R - prev_right_tip_dist, min=0.0)

    # Update HWM: best Phi = max Phi = closest distance achieved
    new_left_hw  = torch.max(prev_left_tip_dist,  Phi_L)
    new_right_hw = torch.max(prev_right_tip_dist, Phi_R)

    # --- Combined reward ---
    # Dense proximity (Phi): always-on, absolute position -> non-farmable by oscillation
    # Progress (delta):       HWM-gated -> non-farmable (only fires on new records)
    reach_reward = fingertip_obb_weight * (Phi_L + Phi_R + delta_L + delta_R) * not_grasped

    return reach_reward, d_R, d_L, new_right_hw, new_left_hw, d_L_pos, d_L_neg, d_R_pos, d_R_neg


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
    prev_right_tip_dist: Tensor,
    prev_left_tip_dist: Tensor,
    # Reward weights
    grasp_bonus: float,
    transport_weight: float,
    lift_bonus: float,
    droppable_bonus: float,
    success_bonus: float,
    lift_shaping_weight: float,
    action_cost_weight: float,
    drop_penalty: float,
    fingertip_obb_weight: float,
    fingertip_sigma: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute total reward for pick-and-place task.

    Pre-grasp signal is exclusively fingertip-based (exponential proximity + HWM progress).
    Approach reward removed - the fingertip exponential gradient handles the whole approach.

    Returns (19 tensors):
        total_reward
        curr_dist, curr_transport_dist, curr_cube_z
        new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success, new_stage_dropped
        action_cost, drop_penalty_reward
        new_right_tip_dist, new_left_tip_dist  (raw Euclidean distances, for display)
        new_right_hw, new_left_hw              (HWM Phi scores, stored as next prev_*)
        d_L_pos, d_L_neg, d_R_pos, d_R_neg    (debug)
    """
    # Gripper-to-cube distance for diagnostics only (not used in reward)
    curr_dist = torch.norm(gripper_pos - cube_pos, dim=1)

    # Pre-grasp: exponential proximity + HWM progress
    (
        fingertip_reach_reward,
        new_right_tip_dist, new_left_tip_dist,
        new_right_hw, new_left_hw,
        d_L_pos, d_L_neg, d_R_pos, d_R_neg,
    ) = compute_fingertip_obb_reach_reward(
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
        sigma=fingertip_sigma,
    )

    # Transport shaping (post-grasp)
    transport_reward, curr_transport_dist = compute_transport_shaping_3d(
        cube_pos, cup_pos, cup_height, cube_half_size,
        prev_transport_dist, is_grasped, transport_weight
    )

    # Lift shaping (post-grasp)
    cube_z = cube_pos[:, 2]
    lift_shaping_reward, curr_cube_z = compute_lift_shaping_delta(
        cube_z, prev_cube_z, is_grasped, lift_shaping_weight
    )

    # One-time bonuses
    cube_z = cube_pos[:, 2]
    is_lifted = is_grasped & (cube_z > 0.03)

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

    # Total reward (no approach reward - fingertip reward handles all of pre-grasp)
    total_reward = (
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
            new_right_hw, new_left_hw,
            d_L_pos, d_L_neg, d_R_pos, d_R_neg)

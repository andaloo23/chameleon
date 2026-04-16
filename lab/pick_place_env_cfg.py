# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
Configuration for the SO-100 Pick-and-Place environment in Isaac Lab.

This defines the environment settings, robot articulation, scene objects,
observation/action spaces, and reward weights.
"""

from __future__ import annotations

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


# Get path to URDF relative to this file
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_URDF_PATH = os.path.join(_CURRENT_DIR, "..", "so100.urdf")


@configclass
class PickPlaceEnvCfg(DirectRLEnvCfg):
    """Configuration for the SO-100 pick-and-place environment."""

    # ===== Environment Settings =====
    decimation = 8  # Reduced for 500Hz physics (500/8 = 62.5Hz RL step)
    episode_length_s = 8.0
    
    # Action and observation space dimensions
    action_space = 6
    observation_space = 29  # joint(12) + rel_cube(3) + rel_cup(3) + cup_height(1) + left_tip(3) + right_tip(3) + gripper_width(1) + cube_dims(3)
    state_space = 0

    # ===== Simulation Settings =====
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 500.0,  # 500Hz for better performance on VMs
        render_interval=8,  # Match decimation
        physx=PhysxCfg(
            solver_type=1,  # TGS solver
            enable_ccd=True, # Re-enabled to prevent tunneling during "soft" contact
            min_position_iteration_count=64, # Increased for better stability
            min_velocity_iteration_count=32, # Better energy dissipation
            gpu_found_lost_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**22,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            bounce_threshold_velocity=0.5, # Slightly higher to prevent jitter
        ),
    )

    # ===== Scene Configuration =====
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # Default parallel environments
        env_spacing=2.0,  # Must contain workspace (0.42m) + park slots (max 0.93m from origin)
        replicate_physics=True,
    )

    # ===== Robot Articulation =====
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UrdfFileCfg(
            asset_path=_URDF_PATH,
            fix_base=True,
            self_collision=False,
            collider_type="convexDecomposition",
            joint_drive=UrdfFileCfg.JointDriveCfg(
                gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None,  # Will use actuator config
                    damping=None,
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 1.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                stiffness=2000.0, # Balanced for payload and movement
                damping=200.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                stiffness=1000.0, # Boosted for faster response
                damping=10.0, # Significantly reduced for max speed
            ),
        },
    )
    
    # Joint names for indexing
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    # Joint limits (radians)
    joint_limits = {
        "shoulder_pan": (-1.5708, 1.5708),  # ±90°
        "shoulder_lift": (0.0, 3.5),
        "elbow_flex": (-3.14159, 0.0),
        "wrist_flex": (-2.5, 1.2),
        "wrist_roll": (-3.14159, 3.14159),
        "gripper": (-0.5, 1.5),
    }

    # Physics material for higher friction
    high_friction_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, # Normalized to prevent "sticking" to floor
        dynamic_friction=1.0,
        restitution=0.0,
    )

    # Cube properties
    cube_mass = 0.05
    # Cube size variants: each tuple is (half_x, half_y, half_z) in meters
    # Full dimensions = 2x each half-size; range 2.5–4.5 cm
    cube_size_variants: tuple = (
        # Symmetric cubes
        (0.0125, 0.0125, 0.0125),  # 2.5 cm
        (0.0150, 0.0150, 0.0150),  # 3.0 cm
        (0.0175, 0.0175, 0.0175),  # 3.5 cm
        (0.0200, 0.0200, 0.0200),  # 4.0 cm
        (0.0225, 0.0225, 0.0225),  # 4.5 cm
        # Rectangular prisms
        (0.0125, 0.0200, 0.0175),  # narrow x, wide y
        (0.0200, 0.0125, 0.0175),  # wide x, narrow y
        (0.0150, 0.0225, 0.0125),  # narrow x, widest y, flat
        (0.0225, 0.0150, 0.0125),  # widest x, narrow y, flat
        (0.0175, 0.0175, 0.0225),  # square base, tall
    )
    cube_color = (1.0, 0.0, 0.0)

    # ===== Cup (Target Container) =====
    cup_height = 0.075  # 7.5cm — default / mean; actual per-episode height drawn from cup_height_variants
    cup_height_variants: tuple = (0.055, 0.065, 0.075, 0.085, 0.095)  # heights to randomly select from at each episode reset
    cup_outer_radius_top = 0.057  # 5.7cm
    cup_outer_radius_bottom = 0.045  # 4.5cm
    cup_wall_thickness = 0.005  # 5mm
    cup_inner_radius_top = 0.052  # cup_outer_radius_top - cup_wall_thickness
    cup_inner_radius_bottom = 0.040
    cup_bottom_thickness = 0.008  # 8mm
    cup_mass = 0.20  # 200g
    cup_color = (0.8, 0.3, 0.2)
    # Note: Cup is created manually with hollow mesh in _setup_scene

    # ===== Workspace Bounds =====
    # Sampling range for cube and cup positions (matches workspace.py)
    workspace_radius_range = (0.20, 0.42)  # 200mm to 420mm from robot base
    workspace_angle_range = (-70.0, 70.0)  # ±70° from -Y axis (cube)
    workspace_angle_range_cup = (-70.0, 70.0)  # ±70° for cup (same as cube)
    cup_cube_min_distance = 0.10  # Minimum separation between cube and cup

    # ===== Action Scaling =====
    action_scale = 0.2  # 2x speed increase for faster transport

    # ===== Action Smoothing =====
    # EMA on joint targets: smoothed = alpha * prev_smoothed + (1-alpha) * new_target
    # Reduces high-frequency jitter that prevents stable grasp zone contact.
    # alpha=0.0 disables smoothing, alpha→1.0 = frozen (use 0.5–0.7 range).
    action_smooth_alpha = 0.6

    # ===== Reward Weights =====
    # Stage 1: Approach cube (delta-based shaping)
    rew_approach_delta_weight = 50.0  # Coarse gripper-to-cube approach (pre-grasp delta shaping)
    
    # Stage 2: Grasp cube (one-time bonus)
    rew_grasp_bonus = 500.0

    # Stage 2b: Per-step reward for maintaining grasp. Must satisfy:
    #   weight × episode_length < droppable_bonus  →  1.5 × 500 = 750 < 800 ✓ (no hold-still optimum)
    # Too low (0.5) causes grasp instability during aggressive transport exploration.
    rew_grasp_hold_weight = 1.5

    # Stage 3: Lift cube (one-time bonus)
    rew_lift_bonus = 800.0

    # Stage 3: Lift cube (dense delta shaping per step)
    rew_lift_shaping_weight = 500.0

    # Stage 3b: Per-step height bonus while grasped — absolute gradient to maintain height.
    # Set to 0: at 80% lift rate this is no longer needed for teaching, and a positive value
    # creates a hover local optimum post-lift (robot earns dense reward for staying at max height
    # instead of navigating toward the cup).
    rew_height_bonus_weight = 0.0

    # Stage 4: Transport to cup (3D delta-based shaping)
    # Distance metric: sqrt(transport_xy_weight*(dx²+dy²) + transport_z_weight*dz²)
    rew_transport_weight = 800.0
    transport_xy_weight = 1.0
    transport_z_weight = 1.0  # equal weight: pure Euclidean distance to target point above cup
    transport_z_clearance = 0.03  # cube bottom target height above cup rim (must be within droppable window [0.01, 0.06])
    # Per-step exponential potential toward cup target during transport (gated on is_grasped &
    # stage_lifted & ~stage_droppable). Must be large enough to beat the hold-still optimum
    # (grasp_hold × 500 steps). At 0.3m: 35 * exp(-0.3/0.35) ≈ 14/step > hold-still signal.
    transport_potential_weight = 35.0
    transport_potential_sigma = 0.35   # distance scale (m): larger sigma gives gradient from ~0.5m out
    
    # Stage 4: Aligned above cup (one-time bonus — cube reached target point above cup center)
    rew_droppable_bonus = 800.0
    
    # Stage 5: Success (one-time bonus)
    rew_success_bonus = 3000.0
    
    # Penalties
    rew_action_cost_weight = 0.025  # Halved alongside action_scale 2x to keep cost-per-distance constant
    rew_drop_penalty = 0.0      # Disabled: was canceling grasp_bonus (+100 grasp - 100 drop = 0 net)
    rew_cup_collision_penalty = -0.5
    # Per-step reward for cube falling downward while above the cup after droppable milestone.
    # Rewards clean releases: if the wrist blocks the cube it won't fall freely.
    # NOTE: keep small — 2000 * fall_distance created a height-proportional exploit where the
    # policy learned to fly as high as possible before releasing to maximise fall reward.
    rew_drop_guidance_weight = 50.0

    # Pre-grasp fingertip reaching rewards
    rew_fingertip_obb_weight     = 200.0   # weight for potential-based delta and close bonus
    rew_grip_open_near           = 1.0     # per-step reward for open gripper (>0.5) within 10cm of cube
    rew_grip_close               = 150.0   # delta-based: reward per rad of closing in contact range (0.20–0.60) within 5cm; ~100 total per committed close
    fingertip_sigma              = 0.20   # exp scale (meters): wider gradient gives signal across full approach range
    fingertip_close_threshold    = 0.025  # avg fingertip distance below which close_bonus fires
    fingertip_close_bonus        = 0.5    # per-step bonus when avg d < close_threshold

    # ===== Grasp Detection Thresholds =====
    grasp_min_contact_pos = 0.20         # gripper must be > this to count as blocked-on-cube (excludes initial pos ≈ 0.0)
    grasp_close_command_threshold = 0.6  # gripper must be < this (open ≈ 1.0–1.5, blocked-on-cube ≈ 0.25–0.55)
    grasp_stall_threshold = 1.0  # rad/s — gripper velocity below this = stalled/blocked by cube
    grasp_lift_threshold = 0.022 # Detect lift earlier
    grasp_following_threshold = 0.03 # 3cm tolerance: handles contact-phase noise during gripper snap-close
    grasp_near_cube_threshold = 0.06  # Max gripper-to-cube dist to allow grasp registration (blocks air-grasp farming)
    grasp_frames_to_grasp = 4 # Require ~65ms of sustained contact before registering grasp
    grasp_frames_to_drop = 20 # Hysteresis: ~320ms at 62.5Hz
    grasp_history_len = 5 # More robust following checks
    grasp_stall_frames = 5 # More robust stall detection
    # Euclidean radius for zone-entry: fingertip is "in zone" when its distance to
    # the assigned face zone center (= fcL or fcR) is below this threshold.
    # Tune based on keyboard_control_lab fcL/fcR values at successful contact.
    grasp_zone_entry_radius = 0.03  # 3cm — tighter zone to reduce false positives
    
    # Aligned / in-cup detection
    # Aligned = cube bottom above cup rim AND cube XY within cup_inner_radius_top of cup center.
    # This is the "target point" — cube is directly above the cup opening, ready to drop.
    aligned_min_height_above_rim = 0.01  # cube bottom must clear cup rim by at least 1cm
    aligned_max_height_above_rim = 0.06  # cube bottom must be no more than 6cm above cup rim
    in_cup_height_margin = 0.02

    # ===== Camera Configuration =====
    # Set enable_cameras=True for dataset collection; leave False during RL training (saves overhead).
    enable_cameras: bool = False
    camera_width: int = 224   # π0.5 / most VLAs expect 224×224
    camera_height: int = 224

    # Third-person camera: position (x,y,z) and look-at target in env-local frame.
    # Robot is at origin; workspace extends 0.2–0.42 m in the –Y direction.
    # Tune these in Isaac Sim viewer → Viewport → Camera if the framing is off.
    camera_third_person_eye: tuple = (0.0, -0.10, 0.70)
    camera_third_person_target: tuple = (0.0, -0.25, 0.0)

    # Wrist camera: translation offset (x,y,z) in gripper-link–local frame.
    # Gripper looks along its –Y axis (tip_offset_gripper ≈ [0.01, –0.102, 0]).
    # Positive Y offset = behind fingertips; positive Z = above centre.
    camera_wrist_pos: tuple = (0.0, 0.04, 0.03)

    # Domain randomization ranges for lighting (sim-to-real).
    # Randomised once per reset call — provides diverse illumination in the dataset.
    light_intensity_range: tuple = (800.0, 3000.0)
    # RGB colour range: ((r_min,g_min,b_min), (r_max,g_max,b_max))
    light_color_range: tuple = ((0.65, 0.62, 0.55), (1.0, 0.98, 0.90))

    # ===== Reset Configuration =====
    # Randomization ranges for cube/cup positions are defined in workspace bounds above
    initial_joint_noise = 0.0  # No noise on joint positions at reset
    # Reset pose: default upright (pan, lift, elbow, wrist, roll, gripper)
    reset_joint_pos = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    # ===== Termination Conditions =====
    max_episode_steps = 500  # Fallback; episode_length_s takes precedence

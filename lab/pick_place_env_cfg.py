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
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
    decimation = 2  # Number of physics steps per RL step
    episode_length_s = 8.0  # ~500 steps at 60Hz RL rate
    
    # Action and observation space dimensions
    action_space = 6  # Delta joint positions for 6 joints
    observation_space = 21  # joint_pos(6) + joint_vel(6) + gripper_pos(3) + cube_pos(3) + cup_pos(3)
    state_space = 0  # No asymmetric critic

    # ===== Simulation Settings =====
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,  # 120Hz physics
        render_interval=2,  # Render every 2 physics steps (60Hz)
        physx=PhysxCfg(
            solver_type=1,  # TGS solver
            enable_ccd=True,  # Continuous collision detection
            min_position_iteration=8,
            min_velocity_iteration=1,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
            bounce_threshold_velocity=0.2,
        ),
    )

    # ===== Scene Configuration =====
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # Default parallel environments
        env_spacing=1.5,  # Space between environment clones
        replicate_physics=True,
    )

    # ===== Robot Articulation =====
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UrdfFileCfg(
            asset_path=_URDF_PATH,
            fix_base=True,
            self_collision=False,
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
                stiffness=1e6,
                damping=1e4,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                stiffness=6000.0,
                damping=400.0,
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
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )

    # ===== Cube (Object to Pick) =====
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.001,
            ),
            physics_material=high_friction_material,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.15, 0.02),  # Default position (will be randomized)
        ),
    )
    
    # Cube properties
    cube_scale = (0.04, 0.04, 0.04)
    cube_mass = 0.05

    # ===== Cup (Target Container) =====
    cup_height = 0.075  # 7.5cm
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
    workspace_radius_range = (0.20, 0.45)  # 200mm to 450mm from robot base
    workspace_angle_range = (-80.0, 80.0)  # ±80° from -Y axis (cube)
    workspace_angle_range_cup = (-80.0, 80.0)  # ±80° for cup (same as cube)
    cup_cube_min_distance = 0.10  # Minimum separation between cube and cup

    # ===== Action Scaling =====
    action_scale = 0.05  # Delta joint position scale (radians)

    # ===== Reward Weights =====
    # Stage 1: Approach cube (delta-based shaping)
    rew_approach_delta_weight = 20.0  # Reward per meter closer to cube
    
    # Stage 2: Grasp cube (one-time bonus)
    rew_grasp_bonus = 2.0
    
    # Stage 3: Transport to cup (dense shaping)
    rew_transport_distance_max = 0.5  # Max XY distance for shaping
    rew_transport_weight = 2.0
    
    # Stage 4: Droppable range (one-time bonus)
    rew_droppable_bonus = 1.5
    
    # Stage 5: Success (one-time bonus)
    rew_success_bonus = 10.0
    
    # Penalties
    rew_action_cost_weight = 0.0002
    rew_drop_penalty = -3.0
    rew_cup_collision_penalty = -0.5

    # ===== Grasp Detection Thresholds =====
    grasp_close_command_threshold = 0.1
    grasp_stall_threshold = 0.001
    grasp_lift_threshold = 0.025
    grasp_following_threshold = 0.0005
    grasp_frames_to_grasp = 15
    grasp_frames_to_drop = 30
    grasp_history_len = 10
    
    # Droppable/In-cup detection
    droppable_xy_margin = 1.0
    droppable_min_height = 0.005
    in_cup_xy_margin = 1.0
    in_cup_height_margin = 0.02

    # ===== Reset Configuration =====
    # Randomization ranges for cube/cup positions are defined in workspace bounds above
    initial_joint_noise = 0.0  # No noise on joint positions at reset

    # ===== Termination Conditions =====
    max_episode_steps = 500  # Fallback; episode_length_s takes precedence

# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
Isaac Lab DirectRLEnv implementation for SO-100 pick-and-place task.

This is the main environment class that ports the Isaac Sim implementation
to Isaac Lab's batched, GPU-accelerated framework.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply


from .pick_place_env_cfg import PickPlaceEnvCfg
from .grasp_detector import GraspDetectorTensor
from .rewards import compute_pick_place_rewards


class PickPlaceEnv(DirectRLEnv):
    """
    Isaac Lab environment for SO-100 robot pick-and-place task.
    
    The task is to pick up a cube and place it into a cup.
    
    Observation space (21-dim):
        - joint_positions (6)
        - joint_velocities (6)
        - gripper_pos (3)
        - cube_pos (3)
        - cup_pos (3)
    
    Action space (6-dim):
        - Delta joint position commands scaled by action_scale
    """

    cfg: PickPlaceEnvCfg

    def __init__(self, cfg: PickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Find joint indices
        self._joint_indices = {}
        for name in self.cfg.joint_names:
            idx, _ = self.robot.find_joints(name)
            self._joint_indices[name] = idx[0]
        
        self._gripper_joint_idx = self._joint_indices["gripper"]
        self._arm_joint_indices = [self._joint_indices[n] for n in self.cfg.joint_names[:5]]
        
        # Find gripper link index for position queries
        self._gripper_link_name = "gripper"
        self._jaw_link_name = "jaw"
        
        self._gripper_body_idx, _ = self.robot.find_bodies(self._gripper_link_name)
        self._jaw_body_idx, _ = self.robot.find_bodies(self._jaw_link_name)
        
        # Fallback if names don't match exactly
        if len(self._gripper_body_idx) == 0:
            for i, body_name in enumerate(self.robot.body_names):
                if "gripper" in body_name.lower() or "fixed_jaw" in body_name.lower():
                    self._gripper_body_idx = [i]
                    break
        
        if len(self._jaw_body_idx) == 0:
            for i, body_name in enumerate(self.robot.body_names):
                if "jaw" in body_name.lower() or "moving_jaw" in body_name.lower():
                    self._jaw_body_idx = [i]
                    break
        
        # Cache joint data views
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # Initialize grasp detector
        self.grasp_detector = GraspDetectorTensor(
            num_envs=self.num_envs,
            device=self.device,
            history_len=self.cfg.grasp_history_len,
            stall_frames=self.cfg.grasp_stall_frames,
            frames_to_grasp=self.cfg.grasp_frames_to_grasp,
            frames_to_drop=self.cfg.grasp_frames_to_drop,
            stall_threshold=self.cfg.grasp_stall_threshold,
            following_threshold=self.cfg.grasp_following_threshold,
            lift_threshold=self.cfg.grasp_lift_threshold,
            near_cube_threshold=self.cfg.grasp_near_cube_threshold,
            close_command_threshold=self.cfg.grasp_close_command_threshold,
        )
        
        # Persistent state tensors for rewards
        self._prev_gripper_cube_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_transport_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_cube_z = torch.zeros(self.num_envs, device=self.device)
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_droppable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_in_cup = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage_droppable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stage_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Current action buffer
        self.actions = torch.zeros(self.num_envs, 6, device=self.device)
        
        # Current joint targets
        self._joint_targets = torch.zeros(self.num_envs, 6, device=self.device)

        # EMA-smoothed joint targets (reduces per-step jitter)
        self._smoothed_joint_targets = torch.zeros(self.num_envs, 6, device=self.device)
        
        # Debug sphere offsets (mutable for keyboard tuning)
        # Fixed (Green): offset in gripper's local frame
        self.tip_offset_gripper = torch.tensor([0.0100, -0.1020, 0.0], device=self.device)
        # Moving (Red): offset in jaw's local frame (180° Y-rotated from gripper)
        self.tip_offset_jaw = torch.tensor([-0.0100, -0.0780, 0.0], device=self.device)

        
        # Cached axis selection per env (set once per episode, used every frame)
        self._use_x = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._left_is_positive = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._zone_margin = 0.025  # 2.5cm protrusion (depth slab + tangential buffer)
        self._fixed_tip_in_left_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._moving_tip_in_right_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Cached fingertip world positions (computed in _get_observations, reused in _get_rewards)
        self._gripper_tip_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._jaw_tip_pos     = torch.zeros(self.num_envs, 3, device=self.device)

        # Cup positions, height, and active variant index (will be set during reset)
        self._cup_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._cup_height = torch.full((self.num_envs,), self.cfg.cup_height, device=self.device)
        self._active_cup_variant = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Per-env cube dimension tensors (half-sizes in meters, set at reset)
        self._cube_dims = torch.zeros(self.num_envs, 3, device=self.device)  # [N, 3] (half_x, half_y, half_z)
        self._cube_half_size_z = torch.zeros(self.num_envs, device=self.device)  # [N]
        self._cube_half_size_grasp = torch.zeros(self.num_envs, device=self.device)  # [N] approach-axis half-size
        self._active_cube_variant = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Previous fingertip-to-face distances for OBB reach reward (pre-grasp)
        self._prev_left_fingertip_dist  = torch.zeros(self.num_envs, device=self.device)
        self._prev_right_fingertip_dist = torch.zeros(self.num_envs, device=self.device)

        # Delta-based approach: previous d_avg for potential shaping
        self._prev_d_avg = torch.ones(self.num_envs, device=self.device)

        # Previous gripper joint value for delta-based grip-close reward
        self._prev_gripper_value = torch.zeros(self.num_envs, device=self.device)
        # Ratchet: tracks the minimum (most-closed) gripper value seen this episode.
        # Only rewards closing beyond this — prevents oscillation farming.
        self._min_gripper_value = torch.full((self.num_envs,), fill_value=2.0, device=self.device)

        self._cum_left_obb_reward   = torch.zeros(self.num_envs, device=self.device)
        self._cum_right_obb_reward  = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """Create the scene with robot, cube, cup, and ground."""
        # Create robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)
        
        from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema, Gf
        import isaaclab.sim.utils.stage as stage_utils
        stage = stage_utils.get_current_stage()
        
        # Spawn the high friction material once globally
        material_path = "/World/Materials/HighFrictionMaterial"
        if not stage.GetPrimAtPath(material_path):
            self.cfg.high_friction_material.func(material_path, self.cfg.high_friction_material)
        
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if robot_prim:
            for prim in Usd.PrimRange(robot_prim):
                prim_path = prim.GetPath().pathString
                
                # Apply high friction material to all robot links
                sim_utils.bind_physics_material(prim.GetPath(), material_path)
                
                # Set contact offsets for all collision prims (1mm for precision)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    physx_col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    # Stronger but controlled depenetration
                    physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    physx_rb_api.CreateMaxDepenetrationVelocityAttr().Set(0.5)
                    
                    # Ensure rest offset is zero for safety
                    physx_col_api.CreateRestOffsetAttr().Set(0.0)
                    physx_col_api.CreateContactOffsetAttr().Set(0.002)
        
        # Spawn each cube size variant at env_0 (will be cloned to all envs)
        for i, (hx, hy, hz) in enumerate(self.cfg.cube_size_variants):
            cube_spawn_cfg = sim_utils.CuboidCfg(
                size=(hx * 2, hy * 2, hz * 2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.5,
                    angular_damping=0.5,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.cube_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                physics_material=self.cfg.high_friction_material,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.cfg.cube_color,
                ),
            )
            cube_spawn_cfg.func(
                f"/World/envs/env_0/Cube_{i}", cube_spawn_cfg,
                translation=(0.0, 0.15 + i * 0.15, hz),
            )

        # Add ground plane and ensure it has a small contact offset
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        ground_prim = stage.GetPrimAtPath("/World/ground/Plane") # Typical path for PlaneCfg
        if not ground_prim: # Fallback search
            for prim in stage.TraverseAll():
                if "ground" in prim.GetPath().pathString and prim.IsA(UsdGeom.Mesh):
                    ground_prim = prim
                    break
        if ground_prim:
            ground_physx = PhysxSchema.PhysxCollisionAPI.Apply(ground_prim)
            ground_physx.CreateContactOffsetAttr().Set(0.001)
            ground_physx.CreateRestOffsetAttr().Set(0.0)
            # Static colliders don't have maxDepenetrationVelocity attributes that we need to set manually here
        
        # Create one cup prim per height variant at env_0 (will be cloned)
        for i, h in enumerate(self.cfg.cup_height_variants):
            self._create_cup_prim(f"/World/envs/env_0/Cup_{i}", (0.0, -0.3 - i * 0.5, 0.0), height=h)
        
        # Clone environments AFTER all assets are added to env_0
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Create RigidObject wrappers for each cup height variant after cloning
        from isaaclab.assets import RigidObjectCfg
        self.cups = []
        for i, h in enumerate(self.cfg.cup_height_variants):
            cup_wrapper_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Cup_{i}",
                spawn=None,  # Already spawned
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, -0.3 - i * 0.5, 0.0),
                ),
            )
            self.cups.append(RigidObject(cup_wrapper_cfg))

        # Create RigidObject wrappers for each cube size variant after cloning
        from isaaclab.assets import RigidObjectCfg as _RigidObjectCfg
        self.cubes = []
        for i, (hx, hy, hz) in enumerate(self.cfg.cube_size_variants):
            cube_wrapper_cfg = _RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Cube_{i}",
                spawn=None,
                init_state=_RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.15 + i * 0.15, hz)),
            )
            self.cubes.append(RigidObject(cube_wrapper_cfg))

        # Add assets to scene
        self.scene.articulations["robot"] = self.robot
        for i, cube in enumerate(self.cubes):
            self.scene.rigid_objects[f"cube_{i}"] = cube
        for i, cup in enumerate(self.cups):
            self.scene.rigid_objects[f"cup_{i}"] = cup
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)




    def _create_cup_prim(self, prim_path: str, position: tuple, height: float | None = None):
        """Create a hollow cup mesh at the given prim path."""
        from pxr import Gf, UsdGeom, UsdPhysics, Usd, PhysxSchema
        import isaaclab.sim.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()

        # Cup dimensions from config
        outer_r_top = self.cfg.cup_outer_radius_top
        outer_r_bot = self.cfg.cup_outer_radius_bottom
        inner_r_top = self.cfg.cup_inner_radius_top
        inner_r_bot = self.cfg.cup_inner_radius_bottom
        height = height if height is not None else self.cfg.cup_height
        bottom_thick = self.cfg.cup_bottom_thickness
        color = self.cfg.cup_color
        
        # Build cup mesh
        points, face_counts, face_indices = self._build_cup_mesh(
            outer_r_top, outer_r_bot, height, inner_r_top, inner_r_bot, bottom_thick
        )
        
        # Create xform for cup
        xform = UsdGeom.Xform.Define(stage, prim_path)
        UsdGeom.XformCommonAPI(xform).SetTranslate(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )
        
        # Create mesh
        mesh_path = f"{prim_path}/CupMesh"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        
        # Apply physics
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        physx_col_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        physx_col_api.CreateContactOffsetAttr().Set(0.001)
        physx_col_api.CreateRestOffsetAttr().Set(0.0)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexDecomposition")
        
        # Apply high friction material
        import isaaclab.sim as sim_utils
        material_path = "/World/Materials/HighFrictionMaterial"
        sim_utils.bind_physics_material(mesh_path, material_path)
        
        xform_prim = xform.GetPrim()
        rb_api = UsdPhysics.RigidBodyAPI.Apply(xform_prim)
        rb_api.CreateKinematicEnabledAttr().Set(True)  # Cup never moves; kinematic = no gravity/contacts overhead
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(xform_prim)
        physx_rb_api.CreateMaxDepenetrationVelocityAttr().Set(0.5)

        mass_api = UsdPhysics.MassAPI.Apply(xform_prim)
        mass_api.CreateMassAttr().Set(10.0)
    
    def _build_cup_mesh(self, outer_r_top, outer_r_bot, height, inner_r_top, inner_r_bot, bottom_thick, segments=32):
        """Build hollow cup mesh geometry."""
        from pxr import Gf
        import numpy as np
        
        bottom_thick = min(bottom_thick, height * 0.4)
        points = []
        face_counts = []
        face_indices = []
        
        def angle(i):
            return (2.0 * np.pi * i) / segments
        
        # Outer bottom ring
        for i in range(segments):
            ang = angle(i)
            points.append(Gf.Vec3f(outer_r_bot * np.cos(ang), outer_r_bot * np.sin(ang), 0.0))
        
        # Outer top ring
        outer_top_offset = len(points)
        for i in range(segments):
            ang = angle(i)
            points.append(Gf.Vec3f(outer_r_top * np.cos(ang), outer_r_top * np.sin(ang), height))
        
        # Inner top ring
        inner_top_offset = len(points)
        for i in range(segments):
            ang = angle(i)
            points.append(Gf.Vec3f(inner_r_top * np.cos(ang), inner_r_top * np.sin(ang), height))
        
        # Inner bottom ring (at bottom_thick height)
        inner_bottom_offset = len(points)
        for i in range(segments):
            ang = angle(i)
            points.append(Gf.Vec3f(inner_r_bot * np.cos(ang), inner_r_bot * np.sin(ang), bottom_thick))
        
        # Center points for bottom
        bottom_center_top_idx = len(points)
        points.append(Gf.Vec3f(0.0, 0.0, bottom_thick))
        bottom_center_bottom_idx = len(points)
        points.append(Gf.Vec3f(0.0, 0.0, 0.0))
        
        def add_tri(a, b, c):
            face_counts.append(3)
            face_indices.extend([a, b, c])
        
        # Build faces
        for i in range(segments):
            next_i = (i + 1) % segments
            # Outer wall
            add_tri(i, next_i, outer_top_offset + i)
            add_tri(outer_top_offset + i, next_i, outer_top_offset + next_i)
            # Inner wall
            add_tri(inner_bottom_offset + i, inner_top_offset + i, inner_bottom_offset + next_i)
            add_tri(inner_top_offset + i, inner_top_offset + next_i, inner_bottom_offset + next_i)
            # Bottom outer to inner
            add_tri(i, inner_bottom_offset + i, next_i)
            add_tri(next_i, inner_bottom_offset + i, inner_bottom_offset + next_i)
            # Inner bottom fan
            add_tri(inner_bottom_offset + i, inner_bottom_offset + next_i, bottom_center_top_idx)
            # Outer bottom fan
            add_tri(i, bottom_center_bottom_idx, next_i)
            # Top rim
            add_tri(outer_top_offset + i, inner_top_offset + i, outer_top_offset + next_i)
            add_tri(inner_top_offset + i, inner_top_offset + next_i, outer_top_offset + next_i)
        
        return points, face_counts, face_indices

    def _get_active_cube_state(self) -> tuple[Tensor, Tensor]:
        """Return world-space (pos, quat) of the active cube for each env."""
        all_pos  = torch.stack([c.data.root_pos_w  for c in self.cubes], dim=0)  # [V, N, 3]
        all_quat = torch.stack([c.data.root_quat_w for c in self.cubes], dim=0)  # [V, N, 4]
        idx = self._active_cube_variant  # [N]
        arange = torch.arange(self.num_envs, device=self.device)
        pos  = all_pos[idx, arange]   # [N, 3]
        quat = all_quat[idx, arange]  # [N, 4]
        return pos, quat

    def _pre_physics_step(self, actions: Tensor) -> None:
        """Process actions before physics step."""
        # Clip actions to [-1, 1]
        self.actions = torch.clamp(actions, -1.0, 1.0)

        # Convert delta actions to absolute joint targets
        delta = self.actions * self.cfg.action_scale
        current_pos = self.joint_pos.clone()
        raw_targets = current_pos + delta

        # Clamp to joint limits
        for i, name in enumerate(self.cfg.joint_names):
            lower, upper = self.cfg.joint_limits[name]
            raw_targets[:, i] = torch.clamp(raw_targets[:, i], lower, upper)

        # EMA smoothing: blend toward new target to reduce per-step jitter.
        # Stable grasp detection needs fingertips in zone for several consecutive frames;
        # high-frequency oscillation breaks this even when the policy is "close enough".
        alpha = self.cfg.action_smooth_alpha
        self._smoothed_joint_targets = alpha * self._smoothed_joint_targets + (1.0 - alpha) * raw_targets
        self._joint_targets = self._smoothed_joint_targets

        # Rule: if droppable (cube above cup, within XY tolerance), force gripper fully open.
        # Applied after EMA so the release is immediate and not damped out.
        gripper_open_pos = self.cfg.joint_limits["gripper"][1]  # upper limit = fully open
        droppable = self.grasp_detector.is_droppable
        self._joint_targets[droppable, self._gripper_joint_idx] = gripper_open_pos
        self._smoothed_joint_targets[droppable, self._gripper_joint_idx] = gripper_open_pos

    def _apply_action(self) -> None:
        """Apply joint position targets to robot."""
        self.robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict:
        """Compute observations for all environments."""
        # Update cached joint data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # Get gripper and jaw world positions / orientations
        gripper_pos  = self.robot.data.body_pos_w[:, self._gripper_body_idx[0], :]
        gripper_quat = self.robot.data.body_quat_w[:, self._gripper_body_idx[0], :]
        jaw_pos      = self.robot.data.body_pos_w[:, self._jaw_body_idx[0], :]
        jaw_quat     = self.robot.data.body_quat_w[:, self._jaw_body_idx[0], :]

        # Compute and cache fingertip world positions (reused in _get_rewards)
        n = gripper_pos.shape[0]
        self._gripper_tip_pos = gripper_pos + quat_apply(gripper_quat, self.tip_offset_gripper.unsqueeze(0).expand(n, -1))
        self._jaw_tip_pos     = jaw_pos     + quat_apply(jaw_quat,     self.tip_offset_jaw.unsqueeze(0).expand(n, -1))
        
        cube_pos, _ = self._get_active_cube_state()

        # Cube-relative fingertip positions (translation-invariant across workspace)
        gripper_tip_rel = self._gripper_tip_pos - cube_pos  # [num_envs, 3]
        jaw_tip_rel     = self._jaw_tip_pos     - cube_pos  # [num_envs, 3]
        
        # Compute new relational vectors
        cube_rel_gripper = cube_pos - gripper_pos
        cup_rel_cube = self._cup_pos - cube_pos
        
        # Get gripper width (unsqueeze to [num_envs, 1] for concatenation)
        gripper_width = self.joint_pos[:, self._gripper_joint_idx].unsqueeze(-1)

        # Concatenate observation
        obs = torch.cat([
            self.joint_pos,                        # [num_envs, 6]
            self.joint_vel,                        # [num_envs, 6]
            cube_rel_gripper,                      # [num_envs, 3]
            cup_rel_cube,                          # [num_envs, 3]
            self._cup_height.unsqueeze(-1),        # [num_envs, 1]
            gripper_tip_rel,                       # [num_envs, 3]
            jaw_tip_rel,                           # [num_envs, 3]
            gripper_width,                         # [num_envs, 1]
            self._cube_dims,                       # [num_envs, 3]
        ], dim=1)  # total: 29
        
        return {"policy": obs}

    def _get_rewards(self) -> Tensor:
        """Compute rewards for all environments."""
        # Get gripper and jaw world positions and orientations
        gripper_pos  = self.robot.data.body_pos_w[:, self._gripper_body_idx[0], :]
        gripper_quat = self.robot.data.body_quat_w[:, self._gripper_body_idx[0], :]
        jaw_pos      = self.robot.data.body_pos_w[:, self._jaw_body_idx[0], :]
        jaw_quat     = self.robot.data.body_quat_w[:, self._jaw_body_idx[0], :]
        
        # Reuse fingertip positions cached by _get_observations this step
        gripper_tip_pos = self._gripper_tip_pos
        jaw_tip_pos     = self._jaw_tip_pos

        cube_pos, cube_quat_cached = self._get_active_cube_state()

        # Select which cube face axis to use + left/right assignment (once per episode at frame 1)
        first_frame_mask = self.episode_length_buf == 1
        if first_frame_mask.any():
            cube_quat_w = cube_quat_cached  # [num_envs, 4]
            n_envs = cube_quat_w.shape[0]
            local_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(n_envs, -1)
            local_y = torch.tensor([0.0, 1.0, 0.0], device=self.device).unsqueeze(0).expand(n_envs, -1)
            cube_x_world = quat_apply(cube_quat_w, local_x)
            cube_y_world = quat_apply(cube_quat_w, local_y)
            # Approach direction: robot base (origin) toward cube, XY only
            approach_dir = cube_pos.clone()
            approach_dir[:, 2] = 0.0
            approach_dir = approach_dir / (approach_dir.norm(dim=-1, keepdim=True) + 1e-8)
            # Pick whichever cube axis is most perpendicular to approach
            dot_x = torch.sum(approach_dir * cube_x_world, dim=-1).abs()
            dot_y = torch.sum(approach_dir * cube_y_world, dim=-1).abs()
            self._use_x = dot_x <= dot_y  # cached per-env boolean
            
            # Determine left/right face assignment via cross product.
            # cross(approach, face_normal).z > 0 means +normal face is on the LEFT.
            best_axis_init = torch.where(self._use_x.unsqueeze(-1), cube_x_world, cube_y_world)
            cross_z = approach_dir[:, 0] * best_axis_init[:, 1] - approach_dir[:, 1] * best_axis_init[:, 0]
            self._left_is_positive = cross_z > 0  # True: +n face = left, -n face = right
            
            pass  # axis/side assignment handled above


        
        # Recompute best_axis for zone checks (every frame, from live pose)
        cube_quat_w = cube_quat_cached
        n_envs = cube_quat_w.shape[0]
        local_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(n_envs, -1)
        local_y = torch.tensor([0.0, 1.0, 0.0], device=self.device).unsqueeze(0).expand(n_envs, -1)
        cube_x_world = quat_apply(cube_quat_w, local_x)
        cube_y_world = quat_apply(cube_quat_w, local_y)
        best_axis = torch.where(self._use_x.unsqueeze(-1), cube_x_world, cube_y_world)
        best_axis = best_axis / best_axis.norm(dim=-1, keepdim=True)
        
        # --- Zone-entry check: Euclidean distance to assigned face zone center ---
        # Face zone center in cube-local space:
        #   face_offset = half_size + 0.5 * zone_margin  (same as rewards.py)
        #   r_pos = +face_offset along axis, r_neg = -face_offset along axis
        # A fingertip is "in zone" when its 3D Euclidean distance to the assigned
        # face center < grasp_zone_entry_radius.  This is directly comparable to
        # the fcL / fcR values printed in the keyboard script.
        margin = self._zone_margin
        is_local_x = self._use_x
        cube_quat_inv = cube_quat_w.clone()
        cube_quat_inv[:, 1:] *= -1.0  # conjugate [w, -x, -y, -z]
        fixed_local  = quat_apply(cube_quat_inv, gripper_tip_pos - cube_pos)
        moving_local = quat_apply(cube_quat_inv, jaw_tip_pos     - cube_pos)

        face_offset = self._cube_half_size_grasp + 0.5 * margin  # [N]
        axis_local_vec = torch.zeros_like(cube_pos)
        axis_local_vec[:, 0] = is_local_x.float()
        axis_local_vec[:, 1] = (~is_local_x).float()
        r_pos = face_offset.unsqueeze(-1) * axis_local_vec   # +face center (cube-local)
        r_neg = -face_offset.unsqueeze(-1) * axis_local_vec  # -face center (cube-local)

        d_fixed_pos  = torch.norm(fixed_local  - r_pos, dim=1)
        d_fixed_neg  = torch.norm(fixed_local  - r_neg, dim=1)
        d_moving_pos = torch.norm(moving_local - r_pos, dim=1)
        d_moving_neg = torch.norm(moving_local - r_neg, dim=1)

        # Optimal assignment: same logic as compute_fingertip_obb_reach_reward
        sum_A = d_fixed_pos + d_moving_neg   # fixed→+face, moving→-face
        sum_B = d_fixed_neg + d_moving_pos   # fixed→-face, moving→+face
        use_A = sum_A <= sum_B
        _d_fixed_to_face  = torch.where(use_A, d_fixed_pos,  d_fixed_neg)   # == fcL
        _d_moving_to_face = torch.where(use_A, d_moving_neg, d_moving_pos)  # == fcR

        # Get gripper joint value and target
        gripper_value = self.joint_pos[:, self._gripper_joint_idx]
        target_gripper = self._joint_targets[:, self._gripper_joint_idx]

        zone_r = self.cfg.grasp_zone_entry_radius
        # Require tips in zone AND gripper is in the blocked-on-cube range AND stalled.
        # Initial gripper pos ≈ 0.0 (fingers closed together, not gripping anything).
        # Blocked-on-cube pos ≈ 0.25–0.65 (fingers spread apart by the cube).
        # Requiring pos > grasp_min_contact_pos excludes the idle/initial closed state.
        gripper_vel = self.joint_vel[:, self._gripper_joint_idx]
        gripper_in_contact_range = (gripper_value > self.cfg.grasp_min_contact_pos) & \
                                   (gripper_value < self.cfg.grasp_close_command_threshold)
        gripper_stalled   = gripper_vel.abs() < self.cfg.grasp_stall_threshold
        contact_confirmed = gripper_in_contact_range & gripper_stalled
        self._fixed_tip_in_left_zone   = (_d_fixed_to_face  < zone_r) & contact_confirmed
        self._moving_tip_in_right_zone = (_d_moving_to_face < zone_r) & contact_confirmed

        # Physical grasp fallback: cube clearly airborne + fingertip right next to it + gripper closed.
        # Catches genuine grasps where fingertips miss the exact OBB face zones.
        cube_bottom_z = cube_pos[:, 2] - self._cube_half_size_z
        gripper_tip_dist_to_cube = torch.norm(gripper_tip_pos - cube_pos, dim=1)
        physical_grasp = (
            (cube_bottom_z > 0.05) &
            (gripper_tip_dist_to_cube < 0.08) &
            (gripper_value > 0.10) &
            (gripper_value < 0.85)
        )
        left_in_zone_final  = self._fixed_tip_in_left_zone  | physical_grasp
        right_in_zone_final = self._moving_tip_in_right_zone | physical_grasp

        # Calculate Local Tip-to-Cube vectors (stationary when cube is held)
        # Transform world-space delta into gripper's local frame
        # Inverse rotation = rotate by conjugate q* = (-x, -y, -z, w)
        gripper_quat_inv = gripper_quat.clone()
        gripper_quat_inv[:, 1:] *= -1.0  # conjugate of [w,x,y,z] = [w,-x,-y,-z]

        gripper_tip_local_dist = quat_apply(gripper_quat_inv, cube_pos - gripper_tip_pos)
        jaw_tip_local_dist = quat_apply(gripper_quat_inv, cube_pos - jaw_tip_pos)

        # Update grasp detector with fingertip zone occupancy
        self.grasp_detector.update(
            left_in_zone=left_in_zone_final,
            right_in_zone=right_in_zone_final,
            cube_pos=cube_pos,
            cup_pos=self._cup_pos,
            cup_height=self._cup_height,
            cup_inner_radius=self.cfg.cup_inner_radius_top,
            cube_half_size=self._cube_half_size_z,
            droppable_min_height=self.cfg.aligned_min_height_above_rim,
            droppable_max_height=self.cfg.aligned_max_height_above_rim,
            in_cup_height_margin=self.cfg.in_cup_height_margin,
            droppable_xy_radius=self.cfg.cup_inner_radius_top * 0.5,  # tighter: cube must be well-centered above cup to release
        )
        
        # Compute 2D reach distance (diagnostic only)
        gripper_xy = gripper_pos[:, :2]
        cube_xy = cube_pos[:, :2]
        reach_dist = torch.norm(gripper_xy - cube_xy, dim=1)



        # Compute rewards
        (
            total_reward, new_dist, new_transport_dist, new_cube_z,
            new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success, new_stage_dropped,
            action_cost, drop_penalty,
            new_right_tip_dist, new_left_tip_dist,
            d_L_pos, d_L_neg, d_R_pos, d_R_neg,
            new_d_avg,
        ) = compute_pick_place_rewards(
            gripper_pos=gripper_pos,
            cube_pos=cube_pos,
            cup_pos=self._cup_pos,
            joint_vel=self.joint_vel,
            prev_gripper_cube_dist=self._prev_gripper_cube_dist,
            prev_transport_dist=self._prev_transport_dist,
            prev_cube_z=self._prev_cube_z,
            is_grasped=self.grasp_detector.is_grasped,
            is_droppable=self.grasp_detector.is_droppable,
            is_in_cup=self.grasp_detector.is_in_cup,
            stage_grasped=self._stage_grasped,
            stage_lifted=self._stage_lifted,
            stage_droppable=self._stage_droppable,
            stage_success=self._stage_success,
            stage_dropped=self._stage_dropped,
            cup_height=self._cup_height,
            cube_half_size_z=self._cube_half_size_z,
            cube_half_grasp=self._cube_half_size_grasp,
            zone_margin=self._zone_margin,
            # Fingertip inputs
            gripper_tip_pos=gripper_tip_pos,
            jaw_tip_pos=jaw_tip_pos,
            cube_quat_w=cube_quat_cached,
            use_x=self._use_x,
            gripper_value=gripper_value,
            gripper_close_threshold=self.cfg.grasp_close_command_threshold,
            # Delta-based approach state
            prev_d_avg=self._prev_d_avg,
            approach_weight=self.cfg.rew_fingertip_obb_weight,
            approach_sigma=self.cfg.fingertip_sigma,
            # Reward weights
            grasp_bonus=self.cfg.rew_grasp_bonus,
            transport_weight=self.cfg.rew_transport_weight,
            lift_bonus=self.cfg.rew_lift_bonus,
            droppable_bonus=self.cfg.rew_droppable_bonus,
            success_bonus=self.cfg.rew_success_bonus,
            lift_shaping_weight=self.cfg.rew_lift_shaping_weight,
            action_cost_weight=self.cfg.rew_action_cost_weight,
            drop_penalty=self.cfg.rew_drop_penalty,
            grasp_hold_weight=self.cfg.rew_grasp_hold_weight,
            height_bonus_weight=self.cfg.rew_height_bonus_weight,
            approach_delta_weight=self.cfg.rew_approach_delta_weight,
            transport_xy_weight=self.cfg.transport_xy_weight,
            transport_z_weight=self.cfg.transport_z_weight,
            transport_potential_weight=self.cfg.transport_potential_weight,
            transport_potential_sigma=self.cfg.transport_potential_sigma,
            transport_z_clearance=self.cfg.transport_z_clearance,
        )
        


        # Ratchet grip-close reward: only fires when gripper reaches a new minimum this episode.
        # Oscillating open→close earns nothing after the first close — un-farmable.
        # Total possible reward per episode: rew_grip_close × (close_threshold - min_contact) ≈ 60.
        gripper_in_contact_range = (
            (gripper_value > self.cfg.grasp_min_contact_pos) &
            (gripper_value < self.cfg.grasp_close_command_threshold)
        ).float()
        gripper_ratchet_delta = torch.clamp(self._min_gripper_value - gripper_value, min=0.0)
        self._min_gripper_value = torch.minimum(self._min_gripper_value, gripper_value)
        r_grip_close_delta = (
            self.cfg.rew_grip_close
            * (new_d_avg < 0.05).float()
            * gripper_in_contact_range
            * gripper_ratchet_delta
            * (~new_stage_grasped).float()
        )
        total_reward = total_reward + r_grip_close_delta

        # Drop guidance reward: once the droppable milestone is reached, reward the cube
        # falling downward while above the cup. Directly rewards clean releases — if the
        # wrist blocks the cube it won't fall freely and this reward won't fire.
        cube_falling = torch.clamp(self._prev_cube_z - cube_pos[:, 2], min=0.0)
        above_cup_xy = torch.norm(cube_pos[:, :2] - self._cup_pos[:, :2], dim=1) < self.cfg.cup_inner_radius_top * 1.5
        r_drop_guidance = (
            self.cfg.rew_drop_guidance_weight
            * cube_falling
            * above_cup_xy.float()
            * new_stage_droppable.float()
        )
        total_reward = total_reward + r_drop_guidance

        # Accumulate per-fingertip debug metrics BEFORE updating cached values.
        not_grasped_mask = (~self._stage_grasped).float()
        
        # d_L and d_R directly from current step
        d_avg = 0.5 * (new_left_tip_dist + new_right_tip_dist)
        _close_mask = (d_avg < self.cfg.fingertip_close_threshold).float()
        
        # We no longer track HWM or compute cumulative phi deltas. 
        # (Could restore direct cumulative reward tracking here later if desired)
        self._cum_left_obb_reward  += 0.0
        self._cum_right_obb_reward += 0.0
        
        # Step reward computation for debug display (matches new delta-based reward)
        phi_old = torch.exp(-self._prev_d_avg / self.cfg.fingertip_sigma)
        phi_new = torch.exp(-new_d_avg / self.cfg.fingertip_sigma)
        delta_phi = torch.clamp(phi_new - phi_old, min=0.0)
        r_approach_dbg = self.cfg.rew_fingertip_obb_weight * delta_phi
        # Reward 1: gripper open (pos > 0.5) while tips are near the cube.
        # Teaches the policy to approach with an open gripper, enabling the grasp.
        gripper_open = (gripper_value > 0.50).float()
        r_open_near = self.cfg.rew_grip_open_near * (new_d_avg < 0.10).float() * gripper_open
        # Reward 2: gripper in contact range (0.20–0.60) while tips are very close.
        # Rewards the closing-onto-cube transition. Uses contact range, not just pos<0.6,
        # so a fully-closed idle gripper (pos≈0.0) does NOT get this reward.
        gripper_in_range = ((gripper_value > self.cfg.grasp_min_contact_pos) &
                            (gripper_value < self.cfg.grasp_close_command_threshold)).float()
        r_grip_close = self.cfg.rew_grip_close * (new_d_avg < 0.10).float() * gripper_in_range

        _step_fingertip_rew = (r_approach_dbg + r_open_near + r_grip_close) * not_grasped_mask
        
        # Removed pre-grasp averages and minimums

        # Update state for next step
        self._prev_gripper_cube_dist = new_dist
        self._prev_transport_dist = new_transport_dist
        self._prev_cube_z = new_cube_z
        self._prev_d_avg = new_d_avg
        self._prev_gripper_value = gripper_value.clone()
        self._was_grasped = self.grasp_detector.is_grasped.clone()
        self._was_droppable = self.grasp_detector.is_droppable.clone()
        self._was_in_cup = self.grasp_detector.is_in_cup.clone()
        self._stage_grasped = new_stage_grasped
        self._stage_lifted = new_stage_lifted
        self._stage_droppable = new_stage_droppable
        self._stage_success = new_stage_success
        self._stage_dropped = new_stage_dropped
        
        # Add debug metrics to extras (for training script monitoring)
        # Return per-env tensors so training script can track all environments
        gripper_cube_dist = torch.norm(gripper_pos - cube_pos, dim=-1)
        gripper_value = self.joint_pos[:, self._gripper_joint_idx]
        
        # --- Debug: compute cube-local tip positions for keyboard script ---
        w_q = cube_quat_w[:, 0]
        x_q = cube_quat_w[:, 1]
        y_q = cube_quat_w[:, 2]
        z_q = cube_quat_w[:, 3]
        R_dbg = torch.stack([
            torch.stack([1 - 2*(y_q*y_q + z_q*z_q), 2*(x_q*y_q - w_q*z_q), 2*(x_q*z_q + w_q*y_q)], dim=-1),
            torch.stack([2*(x_q*y_q + w_q*z_q), 1 - 2*(x_q*x_q + z_q*z_q), 2*(y_q*z_q - w_q*x_q)], dim=-1),
            torch.stack([2*(x_q*z_q - w_q*y_q), 2*(y_q*z_q + w_q*x_q), 1 - 2*(x_q*x_q + y_q*y_q)], dim=-1),
        ], dim=1)
        R_inv_dbg = R_dbg.transpose(1, 2)
        dbg_q_left  = torch.bmm(R_inv_dbg, (gripper_tip_pos - cube_pos).unsqueeze(-1)).squeeze(-1)
        dbg_q_right = torch.bmm(R_inv_dbg, (jaw_tip_pos     - cube_pos).unsqueeze(-1)).squeeze(-1)
        dbg_axis_local = torch.zeros_like(cube_pos)
        dbg_axis_local[:, 0] = self._use_x.float()
        dbg_axis_local[:, 1] = (~self._use_x).float()
        dbg_sign_left = torch.where(self._left_is_positive,
                                    torch.ones(self.num_envs, device=self.device), -torch.ones(self.num_envs, device=self.device))
        
        self.extras["task_state"] = {
            "gripper_cube_distance": gripper_cube_dist,  # [num_envs] tensor
            "gripper_width": gripper_value,  # [num_envs] tensor
            "is_grasped": self.grasp_detector.is_grasped,  # [num_envs] bool tensor
            "is_droppable": self.grasp_detector.is_droppable,  # [num_envs] bool tensor
            "is_in_cup": self.grasp_detector.is_in_cup,  # [num_envs] bool tensor
            "gripper_pos": gripper_pos,  # Fixed jaw frame origin
            "jaw_pos": jaw_pos,          # Moving jaw frame origin
            "gripper_tip_pos": gripper_tip_pos, # Fixed jaw physical tip (Left)
            "jaw_tip_pos": jaw_tip_pos,         # Moving jaw physical tip (Right)
            "left_zone_ok": self._fixed_tip_in_left_zone,    # fixed jaw in left face zone
            "right_zone_ok": self._moving_tip_in_right_zone,   # moving jaw in right face zone
            "cube_pos": cube_pos,
            # Fingertip OBB face distances (Euclidean to face zone center)
            "d_left": new_left_tip_dist,
            "d_right": new_right_tip_dist,
            "fingertip_step_reward": _step_fingertip_rew,
            "d_L_pos": d_L_pos,
            "d_L_neg": d_L_neg,
            "d_R_pos": d_R_pos,
            "d_R_neg": d_R_neg,
            "cum_left_obb_reward":  self._cum_left_obb_reward,
            "cum_right_obb_reward": self._cum_right_obb_reward,
            # Debug: cube-local tip positions for keyboard script
            "dbg_q_left": dbg_q_left,
            "dbg_q_right": dbg_q_right,
            "dbg_axis_local": dbg_axis_local,
            "dbg_sign_left": dbg_sign_left,
            # Zone debug: direct Euclidean distance to assigned face center (== fcL, fcR)
            "dbg_d_fixed": _d_fixed_to_face,    # same as fcL; zone fires when < grasp_zone_entry_radius
            "dbg_d_moving": _d_moving_to_face,   # same as fcR
            "dbg_zone_radius": torch.full((self.num_envs,), self.cfg.grasp_zone_entry_radius, device=self.device),
            "penalties": {
                "action_cost": action_cost,
                "drop_penalty": drop_penalty,
                "cup_collision": torch.zeros_like(action_cost),
                "self_collision": torch.zeros_like(action_cost),
            }
        }
        
        # Expose latched flags for PPO metrics — clone to survive auto-reset
        # (_reset_idx clears _stage_* in-place before training script reads extras)
        self.extras["milestone_flags"] = {
            "lifted": self._stage_lifted.clone(),
            "droppable": self._stage_droppable.clone(),
            "success": self._stage_success.clone(),
            "grasped": self._stage_grasped.clone(),
        }
        
        return total_reward

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        """Compute termination conditions."""
        # Success: cube is in cup
        terminated = self.grasp_detector.is_in_cup
        
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Convert to tensor if needed
        if not isinstance(env_ids, Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_reset = len(env_ids)
        
        # Reset robot to workspace-ready pose with optional noise
        # Start from default (robot-internal ordering), then overwrite with our reset pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        for i, name in enumerate(self.cfg.joint_names):
            idx = self._joint_indices[name]
            joint_pos[:, idx] = self.cfg.reset_joint_pos[i]
        if self.cfg.initial_joint_noise > 0.0:
            noise = sample_uniform(
                -self.cfg.initial_joint_noise, self.cfg.initial_joint_noise,
                joint_pos.shape, self.device
            )
            joint_pos = joint_pos + noise
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        cube_xy = self._sample_workspace_xy(num_reset)
        
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Cube variant selection
        cube_variant_idx = torch.randint(0, len(self.cfg.cube_size_variants), (num_reset,), device=self.device)
        self._active_cube_variant[env_ids] = cube_variant_idx
        variants_dims = torch.tensor(self.cfg.cube_size_variants, device=self.device)  # [V, 3]
        self._cube_dims[env_ids] = variants_dims[cube_variant_idx]  # [num_reset, 3]
        self._cube_half_size_z[env_ids] = self._cube_dims[env_ids, 2]

        # Cube position
        cube_z = self._cube_half_size_z[env_ids]  # per-env height above table
        cube_pos = torch.stack([cube_xy[:, 0], cube_xy[:, 1], cube_z], dim=1)
        cube_pos += self.scene.env_origins[env_ids]

        # Random yaw rotation for cube
        random_yaw = sample_uniform(-math.pi, math.pi, (num_reset,), self.device)
        cube_quat = torch.zeros(num_reset, 4, device=self.device)
        cube_quat[:, 0] = torch.cos(random_yaw / 2.0)  # w
        cube_quat[:, 3] = torch.sin(random_yaw / 2.0)  # z

        # Write each cube variant: active variant at workspace, others at unique park slots.
        # Park slots are BEHIND the robot (+Y), which is never part of the workspace
        # (workspace angle range ±70° from -Y axis), so they never interfere with active sim.
        # 2-row grid: variants 0-4 at y=+0.75, variants 5-9 at y=+0.90; x evenly spread.
        # Max park distance from origin: 0.93m < env_spacing/2 = 1.0m — safe.
        zero_vel_cube = torch.zeros(num_reset, 6, device=self.device)
        for k, cube_obj in enumerate(self.cubes):
            is_active_k = (cube_variant_idx == k)
            # Use both +X and -X sides, with 0.1m spacing, at Y=+0.3 (within env bounds)
            # Arrange: k=0..4 on -X, k=5..9 on +X
            if k < 5:
                park_px = -0.40 + k * 0.1  # -0.40, -0.30, -0.20, -0.10, 0.0
                park_py = 0.3
            else:
                park_px = 0.10 + (k-5) * 0.1  # 0.10, 0.20, 0.30, 0.40, 0.50
                park_py = 0.3
            park_hz = self.cfg.cube_size_variants[k][2]  # rest height for this variant
            active_pos = torch.stack([cube_xy[:, 0], cube_xy[:, 1], cube_z], dim=1)
            park_pos   = torch.stack([
                torch.full((num_reset,), park_px, device=self.device),
                torch.full((num_reset,), park_py, device=self.device),
                torch.full((num_reset,), park_hz, device=self.device),
            ], dim=1)
            pos_k = torch.where(is_active_k.unsqueeze(-1), active_pos, park_pos)
            pos_k += self.scene.env_origins[env_ids]
            cube_obj.write_root_pose_to_sim(torch.cat([pos_k, cube_quat], dim=1), env_ids)
            cube_obj.write_root_velocity_to_sim(zero_vel_cube, env_ids)
        
        # Initialize axis selection immediately so frame 0 has correct OBB distances.
        # This must happen AFTER writing cube pose so quat_apply uses the new orientation.
        local_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(num_reset, -1)
        local_y = torch.tensor([0.0, 1.0, 0.0], device=self.device).unsqueeze(0).expand(num_reset, -1)
        cube_x_world = quat_apply(cube_quat, local_x)
        cube_y_world = quat_apply(cube_quat, local_y)
        # Approach direction: robot base (origin) toward cube, XY only
        cube_xy_dir = cube_xy.clone()
        cube_xy_3d = torch.cat([cube_xy_dir, torch.zeros(num_reset, 1, device=self.device)], dim=1)
        cube_xy_3d = cube_xy_3d / (cube_xy_3d.norm(dim=-1, keepdim=True) + 1e-8)
        # Pick whichever cube axis is most perpendicular to approach
        dot_x = torch.sum(cube_xy_3d * cube_x_world, dim=-1).abs()
        dot_y = torch.sum(cube_xy_3d * cube_y_world, dim=-1).abs()
        self._use_x[env_ids] = dot_x <= dot_y
        # Determine left/right face assignment
        best_axis_init = torch.where(self._use_x[env_ids].unsqueeze(-1), cube_x_world, cube_y_world)
        cross_z = cube_xy_3d[:, 0] * best_axis_init[:, 1] - cube_xy_3d[:, 1] * best_axis_init[:, 0]
        self._left_is_positive[env_ids] = cross_z > 0

        # Set per-env approach-axis half-size (depends on _use_x which was just set)
        use_x_mask = self._use_x[env_ids]
        self._cube_half_size_grasp[env_ids] = torch.where(
            use_x_mask,
            self._cube_dims[env_ids, 0],  # half_x when approaching from X faces
            self._cube_dims[env_ids, 1],  # half_y when approaching from Y faces
        )

        # Randomize cup XY position and height variant
        cup_xy = self._sample_workspace_xy(num_reset, existing_xy=cube_xy)
        if hasattr(self, '_debug_force_cup_variant') and self._debug_force_cup_variant is not None:
            variant_idx = torch.full((num_reset,), self._debug_force_cup_variant, dtype=torch.long, device=self.device)
        else:
            variant_idx = torch.randint(0, len(self.cfg.cup_height_variants), (num_reset,), device=self.device)
        variants_tensor = torch.tensor(self.cfg.cup_height_variants, device=self.device)
        self._cup_height[env_ids] = variants_tensor[variant_idx]
        self._active_cup_variant[env_ids] = variant_idx

        # Update internal cup base position tracking (z=0 at table level in local coords)
        cup_local = torch.stack([cup_xy[:, 0], cup_xy[:, 1], torch.zeros(num_reset, device=self.device)], dim=1)
        self._cup_pos[env_ids] = cup_local + self.scene.env_origins[env_ids]

        # Place the active variant at the workspace position; park all others underground
        cup_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_reset, 4)
        zero_vel = torch.zeros(num_reset, 6, device=self.device)
        for k, (cup_obj, h) in enumerate(zip(self.cups, self.cfg.cup_height_variants)):
            is_active = (variant_idx == k)
            # Use both +X and -X sides, with 0.12m spacing (accounts for max cup radius 0.057m), at Y=+0.6 (within env bounds)
            if k < 3:
                park_px = -0.24 + k * 0.12  # -0.24, -0.12, 0.0
                park_py = 0.6
            else:
                park_px = 0.12 + (k-3) * 0.12  # 0.12, 0.24
                park_py = 0.6
            active_pos = torch.stack([cup_xy[:, 0], cup_xy[:, 1], torch.full((num_reset,), h / 2.0, device=self.device)], dim=1)
            park_pos   = torch.stack([
                torch.full((num_reset,), park_px, device=self.device),
                torch.full((num_reset,), park_py, device=self.device),
                torch.full((num_reset,), -10.0, device=self.device),
            ], dim=1)
            pos_world = torch.where(is_active.unsqueeze(-1), active_pos, park_pos) + self.scene.env_origins[env_ids]
            cup_obj.write_root_pose_to_sim(torch.cat([pos_world, cup_quat], dim=1), env_ids)
            cup_obj.write_root_velocity_to_sim(zero_vel, env_ids)

        # Toggle kinematic per env: parked cups stay underground, active cup responds to physics
        from pxr import UsdPhysics
        import isaaclab.sim.utils.stage as stage_utils
        stage = stage_utils.get_current_stage()
        for env_idx, active_k in zip(env_ids.tolist(), variant_idx.tolist()):
            for k in range(len(self.cfg.cup_height_variants)):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Cup_{k}")
                UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath()).GetKinematicEnabledAttr().Set(k != active_k)
        

        # Reset grasp detector state
        self.grasp_detector.reset(env_ids)
        
        # Reset reward state
        self._prev_gripper_cube_dist[env_ids] = torch.norm(
            self.robot.data.body_pos_w[env_ids, self._gripper_body_idx[0], :] - cube_pos,
            dim=1
        )
        self._prev_cube_z[env_ids] = cube_pos[:, 2]
        # Initialize transport dist to cube-cup distance at episode start (world coords, correct now)
        cube_cup_dx = cube_pos[:, 0] - self._cup_pos[env_ids, 0]
        cube_cup_dy = cube_pos[:, 1] - self._cup_pos[env_ids, 1]
        cube_bottom_z = cube_pos[:, 2] - self._cube_half_size_z[env_ids]
        cup_z_target = self._cup_pos[env_ids, 2] + self._cup_height[env_ids] + self.cfg.transport_z_clearance
        cube_cup_dz = cube_bottom_z - cup_z_target
        self._prev_transport_dist[env_ids] = torch.sqrt(cube_cup_dx**2 + cube_cup_dy**2 + cube_cup_dz**2)
        # Reset fingertip HWM Phi scores to 0.0 (no best yet = d=inf equivalent).
        # With sigma=0.10 the first-step spike is only ~1.77 — too small to farm but
        # provides a useful early-training gradient. The proxy init (gripper-body-dist)
        # overshoots the actual OBB Phi (fingertips farther than body), causing zero
        # reward on every step and killing the approach signal.
        self._prev_left_fingertip_dist[env_ids]  = 0.0
        self._prev_right_fingertip_dist[env_ids] = 0.0
        self._cum_left_obb_reward[env_ids]   = 0.0
        self._cum_right_obb_reward[env_ids]  = 0.0
        # Initialize prev_d_avg to typical starting distance (~0.5m).
        # With unclamped delta, this avoids a large penalty on the first step.
        self._prev_d_avg[env_ids] = 0.5
        self._prev_gripper_value[env_ids] = 0.0
        self._min_gripper_value[env_ids] = 2.0
        self._was_grasped[env_ids] = False
        self._was_droppable[env_ids] = False
        self._was_in_cup[env_ids] = False
        self._stage_grasped[env_ids] = False
        self._stage_lifted[env_ids] = False
        self._stage_droppable[env_ids] = False
        self._stage_success[env_ids] = False
        self._stage_dropped[env_ids] = False
        
        # Reset joint targets and smoothed targets
        self._joint_targets[env_ids] = joint_pos
        self._smoothed_joint_targets[env_ids] = joint_pos

    def _sample_workspace_xy(
        self,
        num_samples: int,
        existing_xy: Tensor | None = None,
    ) -> Tensor:
        """
        Sample XY positions within workspace.
        
        Args:
            num_samples: Number of positions to sample
            existing_xy: Optional existing positions to maintain distance from
        
        Returns:
            xy: Tensor of shape [num_samples, 2]
        """
        # Sample radius within workspace range
        r_min, r_max = self.cfg.workspace_radius_range
        radius = sample_uniform(r_min, r_max, (num_samples,), self.device)
        
        # Sample angle within workspace range
        if existing_xy is not None:
            # Use narrower angle range for cup
            a_min, a_max = self.cfg.workspace_angle_range_cup
        else:
            a_min, a_max = self.cfg.workspace_angle_range
        
        angle_rad = sample_uniform(
            math.radians(a_min),
            math.radians(a_max),
            (num_samples,),
            self.device,
        )
        
        # Convert to XY (angle is from -Y axis)
        x = radius * torch.sin(angle_rad)
        y = -radius * torch.cos(angle_rad)
        xy = torch.stack([x, y], dim=1)
        
        # If existing positions provided, resample any that are too close
        if existing_xy is not None:
            for _ in range(10):  # Max 10 attempts
                dist = torch.norm(xy - existing_xy, dim=1)
                too_close = dist < self.cfg.cup_cube_min_distance
                if not too_close.any():
                    break
                
                # Resample for those too close
                n_resample = too_close.sum().item()
                new_radius = sample_uniform(r_min, r_max, (n_resample,), self.device)
                new_angle = sample_uniform(
                    math.radians(a_min),
                    math.radians(a_max),
                    (n_resample,),
                    self.device,
                )
                xy[too_close, 0] = new_radius * torch.sin(new_angle)
                xy[too_close, 1] = -new_radius * torch.cos(new_angle)
        
        return xy

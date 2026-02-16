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
        
        # Debug sphere offsets (mutable for keyboard tuning)
        # Fixed (Green): offset in gripper's local frame
        self.tip_offset_gripper = torch.tensor([0.0100, -0.1020, 0.0], device=self.device)
        # Moving (Red): offset in jaw's local frame (180° Y-rotated from gripper)
        self.tip_offset_jaw = torch.tensor([-0.0100, -0.0780, 0.0], device=self.device)

        
        # Cached axis selection per env (set once per episode, used every frame)
        self._use_x = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._left_is_positive = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._zone_margin = 0.01  # 1cm protrusion
        self._moving_tip_in_left_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fixed_tip_in_right_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Cup positions (will be set during reset)
        self._cup_pos = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_scene(self):
        """Create the scene with robot, cube, cup, and ground."""
        # Create robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)
        
        from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
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
        
        # Create cube rigid object
        self.cube = RigidObject(self.cfg.cube_cfg)
        
        # Don't apply convexDecomposition to cube - let it use default collision
        
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
        
        # Create hollow cup at env_0 (will be cloned)
        self._create_cup_prim("/World/envs/env_0/Cup", (0.0, -0.3, 0.0))
        
        # Clone environments AFTER all assets are added to env_0
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Create RigidObject wrapper for cup after cloning
        from isaaclab.assets import RigidObjectCfg
        cup_wrapper_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cup",
            spawn=None,  # Already spawned
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, -0.3, 0.0),
            ),
        )
        self.cup = RigidObject(cup_wrapper_cfg)
        
        # Add assets to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.rigid_objects["cup"] = self.cup
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Cube face marker spheres (yellow, visual-only, parented to cube)
        # Created as children of the cube prim so they move with it.
        # No CollisionAPI applied — the gripper passes right through.
        cube_prim_path = "/World/envs/env_0/Cube"
        half = self.cfg.cube_scale[0] / 2.0
        for i, sign in enumerate([1.0, -1.0]):
            sphere_path = f"{cube_prim_path}/FaceMarker_{i}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.CreateRadiusAttr(0.008)
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 0.0)])
            # Default local position along Y (updated when axis is determined)
            UsdGeom.XformCommonAPI(sphere).SetTranslate(
                Gf.Vec3d(0.0, sign * float(half), 0.0)
            )
    def _create_cup_prim(self, prim_path: str, position: tuple):
        """Create a hollow cup mesh at the given prim path."""
        from pxr import Gf, UsdGeom, UsdPhysics, Usd, PhysxSchema
        import isaaclab.sim.utils.stage as stage_utils
        
        stage = stage_utils.get_current_stage()
        
        # Cup dimensions from config
        outer_r_top = self.cfg.cup_outer_radius_top
        outer_r_bot = self.cfg.cup_outer_radius_bottom
        inner_r_top = self.cfg.cup_inner_radius_top
        inner_r_bot = self.cfg.cup_inner_radius_bottom
        height = self.cfg.cup_height
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
        UsdPhysics.RigidBodyAPI.Apply(xform_prim)
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(xform_prim)
        physx_rb_api.CreateMaxDepenetrationVelocityAttr().Set(0.5)
        
        mass_api = UsdPhysics.MassAPI.Apply(xform_prim)
        mass_api.CreateMassAttr().Set(10.0)  # Heavy so it doesn't move
    
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

    def _pre_physics_step(self, actions: Tensor) -> None:
        """Process actions before physics step."""
        # Clip actions to [-1, 1]
        self.actions = torch.clamp(actions, -1.0, 1.0)
        
        # Convert delta actions to absolute joint targets
        delta = self.actions * self.cfg.action_scale
        current_pos = self.joint_pos.clone()
        self._joint_targets = current_pos + delta
        
        # Clamp to joint limits
        for i, name in enumerate(self.cfg.joint_names):
            lower, upper = self.cfg.joint_limits[name]
            self._joint_targets[:, i] = torch.clamp(self._joint_targets[:, i], lower, upper)
        
        # Note: Removed gripper override logic that forced gripper open before grasp.
        # This allows direct gripper control for testing and training.

    def _apply_action(self) -> None:
        """Apply joint position targets to robot."""
        self.robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict:
        """Compute observations for all environments."""
        # Update cached joint data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # Get gripper world position
        gripper_pos = self.robot.data.body_pos_w[:, self._gripper_body_idx[0], :]
        
        # Get cube position
        cube_pos = self.cube.data.root_pos_w
        
        # Concatenate observation
        obs = torch.cat([
            self.joint_pos,                    # [num_envs, 6]
            self.joint_vel,                    # [num_envs, 6]
            gripper_pos,                       # [num_envs, 3]
            cube_pos,                          # [num_envs, 3]
            self._cup_pos,                     # [num_envs, 3]
        ], dim=1)
        
        return {"policy": obs}

    def _get_rewards(self) -> Tensor:
        """Compute rewards for all environments."""
        # Get gripper and jaw world positions and orientations
        gripper_pos = self.robot.data.body_pos_w[:, self._gripper_body_idx[0], :]
        gripper_quat = self.robot.data.body_quat_w[:, self._gripper_body_idx[0], :]
        jaw_pos = self.robot.data.body_pos_w[:, self._jaw_body_idx[0], :]
        jaw_quat = self.robot.data.body_quat_w[:, self._jaw_body_idx[0], :]
        cube_pos = self.cube.data.root_pos_w
        
        # Compute fingertip world positions from mutable offsets
        gripper_tip_pos = gripper_pos + quat_apply(gripper_quat, self.tip_offset_gripper)
        jaw_tip_pos = jaw_pos + quat_apply(jaw_quat, self.tip_offset_jaw)
        
        # Select which cube face axis to use + left/right assignment (once per episode at frame 1)
        first_frame_mask = self.episode_length_buf == 1
        if first_frame_mask.any():
            cube_quat_w = self.cube.data.root_quat_w  # [num_envs, 4]
            local_x = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            local_y = torch.tensor([0.0, 1.0, 0.0], device=self.device)
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
            
            # Update face marker sphere local positions to match chosen axis
            import isaaclab.sim.utils.stage as stage_utils
            from pxr import UsdGeom, Gf
            stage = stage_utils.get_current_stage()
            half_val = float(self.cfg.cube_scale[0] / 2.0)
            for env_id in range(self.num_envs):
                if not first_frame_mask[env_id]:
                    continue
                for i, sign in enumerate([1.0, -1.0]):
                    prim_path = f"/World/envs/env_{env_id}/Cube/FaceMarker_{i}"
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim:
                        if self._use_x[env_id]:
                            UsdGeom.XformCommonAPI(prim).SetTranslate(
                                Gf.Vec3d(sign * half_val, 0.0, 0.0)
                            )
                        else:
                            UsdGeom.XformCommonAPI(prim).SetTranslate(
                                Gf.Vec3d(0.0, sign * half_val, 0.0)
                            )
        
        # Recompute best_axis for zone checks (every frame, from live pose)
        cube_quat_w = self.cube.data.root_quat_w
        local_x = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        local_y = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        cube_x_world = quat_apply(cube_quat_w, local_x)
        cube_y_world = quat_apply(cube_quat_w, local_y)
        best_axis = torch.where(self._use_x.unsqueeze(-1), cube_x_world, cube_y_world)
        best_axis = best_axis / best_axis.norm(dim=-1, keepdim=True)
        
        half = self.cfg.cube_scale[0] / 2.0
        
        # --- Left/Right zone-entry check for each fingertip ---
        margin = self._zone_margin
        cube_quat_inv = cube_quat_w.clone()
        cube_quat_inv[:, :3] *= -1.0  # conjugate = inverse for unit quat
        half_size = half
        is_local_x = self._use_x
        fixed_local = quat_apply(cube_quat_inv, gripper_tip_pos - cube_pos)
        moving_local = quat_apply(cube_quat_inv, jaw_tip_pos - cube_pos)
        
        def _in_face_zone(tip_local, is_x, positive_side):
            """Check if tip is inside a specific face zone (positive or negative side)."""
            normal_coord = torch.where(is_x, tip_local[:, 0], tip_local[:, 1])
            tangent_coord = torch.where(is_x, tip_local[:, 1], tip_local[:, 0])
            z_coord = tip_local[:, 2]
            # Check the correct side only
            in_zone_pos = (normal_coord >= half_size) & (normal_coord <= half_size + margin)
            in_zone_neg = (normal_coord <= -half_size) & (normal_coord >= -half_size - margin)
            in_normal = torch.where(positive_side, in_zone_pos, in_zone_neg)
            in_face = (tangent_coord.abs() <= half_size) & (z_coord.abs() <= half_size)
            return in_normal & in_face
        
        # Moving jaw (left gripper) -> left face zone
        self._moving_tip_in_left_zone = _in_face_zone(moving_local, is_local_x, self._left_is_positive)
        # Fixed jaw (right gripper) -> right face zone (opposite side)
        self._fixed_tip_in_right_zone = _in_face_zone(fixed_local, is_local_x, ~self._left_is_positive)
        
        # Calculate Local Tip-to-Cube vectors (stationary when cube is held)
        # Transform world-space delta into gripper's local frame
        # Inverse rotation = rotate by conjugate q* = (-x, -y, -z, w)
        gripper_quat_inv = gripper_quat.clone()
        gripper_quat_inv[:, :3] *= -1.0
        
        gripper_tip_local_dist = quat_apply(gripper_quat_inv, cube_pos - gripper_tip_pos)
        jaw_tip_local_dist = quat_apply(gripper_quat_inv, cube_pos - jaw_tip_pos)
        

        
        # Get gripper joint value and target
        gripper_value = self.joint_pos[:, self._gripper_joint_idx]
        target_gripper = self._joint_targets[:, self._gripper_joint_idx]
        
        # Update grasp detector with both jaws
        self.grasp_detector.update(
            gripper_value=gripper_value,
            target_gripper=target_gripper,
            gripper_pos=gripper_pos,
            jaw_pos=jaw_pos,
            cube_pos=cube_pos,
            cup_pos=self._cup_pos,
            cup_height=self.cfg.cup_height,
            cup_inner_radius=self.cfg.cup_inner_radius_top,
            cube_half_size=self.cfg.cube_scale[2] / 2.0,
            droppable_min_height=self.cfg.droppable_min_height,
            in_cup_height_margin=self.cfg.in_cup_height_margin,
        )
        
        # Compute rewards
        total_reward, new_dist, new_transport_dist, new_cube_z, new_stage_grasped, new_stage_lifted, new_stage_droppable, new_stage_success, new_stage_dropped, action_cost, drop_penalty = compute_pick_place_rewards(
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
            cup_height=self.cfg.cup_height,
            cube_half_size=self.cfg.cube_scale[2] / 2.0,
            approach_weight=self.cfg.rew_approach_delta_weight,
            grasp_bonus=self.cfg.rew_grasp_bonus,
            transport_weight=self.cfg.rew_transport_weight,
            lift_bonus=self.cfg.rew_lift_bonus,
            droppable_bonus=self.cfg.rew_droppable_bonus,
            success_bonus=self.cfg.rew_success_bonus,
            lift_shaping_weight=self.cfg.rew_lift_shaping_weight,
            action_cost_weight=self.cfg.rew_action_cost_weight,
            drop_penalty=self.cfg.rew_drop_penalty,
        )
        
        # Update state for next step
        self._prev_gripper_cube_dist = new_dist
        self._prev_transport_dist = new_transport_dist
        self._prev_cube_z = new_cube_z
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
        
        self.extras["task_state"] = {
            "gripper_cube_distance": gripper_cube_dist,  # [num_envs] tensor
            "gripper_width": gripper_value,  # [num_envs] tensor
            "is_grasped": self.grasp_detector.is_grasped,  # [num_envs] bool tensor
            "is_droppable": self.grasp_detector.is_droppable,  # [num_envs] bool tensor
            "is_in_cup": self.grasp_detector.is_in_cup,  # [num_envs] bool tensor
            "gripper_pos": gripper_pos,  # Fixed jaw frame origin
            "jaw_pos": jaw_pos,          # Moving jaw frame origin
            "gripper_tip_pos": gripper_tip_pos, # Fixed jaw physical tip
            "jaw_tip_pos": jaw_tip_pos,         # Moving jaw physical tip
            "left_zone_ok": self._moving_tip_in_left_zone,    # moving jaw in left face zone
            "right_zone_ok": self._fixed_tip_in_right_zone,   # fixed jaw in right face zone
            "cube_pos": cube_pos,
            "penalties": {
                "action_cost": action_cost,
                "drop_penalty": drop_penalty,
                "cup_collision": torch.zeros_like(action_cost),
                "self_collision": torch.zeros_like(action_cost),
            }
        }
        
        # Expose latched flags for PPO metrics
        self.extras["milestone_flags"] = {
            "lifted": self._stage_lifted,
            "droppable": self._stage_droppable,
            "success": self._stage_success,
            "grasped": self._stage_grasped,
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
        
        # Reset robot to default pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Randomize cube positions
        cube_xy = self._sample_workspace_xy(num_reset)
        cube_z = torch.full((num_reset,), self.cfg.cube_scale[2] / 2.0, device=self.device)
        cube_pos = torch.stack([cube_xy[:, 0], cube_xy[:, 1], cube_z], dim=1)
        cube_pos += self.scene.env_origins[env_ids]
        
        # Random yaw rotation for cube
        random_yaw = sample_uniform(-math.pi, math.pi, (num_reset,), self.device)
        cube_quat = torch.zeros(num_reset, 4, device=self.device)
        cube_quat[:, 0] = torch.cos(random_yaw / 2.0)  # w
        cube_quat[:, 3] = torch.sin(random_yaw / 2.0)  # z
        self.cube.write_root_pose_to_sim(torch.cat([cube_pos, cube_quat], dim=1), env_ids)
        self.cube.write_root_velocity_to_sim(torch.zeros(num_reset, 6, device=self.device), env_ids)
        
        # Randomize cup positions (ensuring distance from cube)
        cup_xy = self._sample_workspace_xy(num_reset, existing_xy=cube_xy)
        cup_z = torch.full((num_reset,), self.cfg.cup_height / 2.0, device=self.device)
        cup_pos = torch.stack([cup_xy[:, 0], cup_xy[:, 1], cup_z], dim=1)
        cup_pos_world = cup_pos + self.scene.env_origins[env_ids]
        
        # Update internal cup position tracking
        self._cup_pos[env_ids] = torch.stack([cup_xy[:, 0], cup_xy[:, 1], torch.zeros(num_reset, device=self.device)], dim=1)
        
        # Move cup using RigidObject API
        cup_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(num_reset, 4)
        self.cup.write_root_pose_to_sim(torch.cat([cup_pos_world, cup_quat], dim=1), env_ids)
        self.cup.write_root_velocity_to_sim(torch.zeros(num_reset, 6, device=self.device), env_ids)
        
        # DEBUG: Print positions
        if num_reset <= 4:  # Only print for small resets
            print(f"[DEBUG RESET] cube_xy={cube_xy[0].cpu().numpy()}, cup_xy={cup_xy[0].cpu().numpy()}")
        
        # Reset grasp detector state
        self.grasp_detector.reset(env_ids)
        
        # Reset reward state
        self._prev_gripper_cube_dist[env_ids] = torch.norm(
            self.robot.data.body_pos_w[env_ids, self._gripper_body_idx[0], :] - cube_pos,
            dim=1
        )
        self._prev_cube_z[env_ids] = cube_pos[:, 2]
        self._was_grasped[env_ids] = False
        self._was_droppable[env_ids] = False
        self._was_in_cup[env_ids] = False
        self._stage_grasped[env_ids] = False
        self._stage_lifted[env_ids] = False
        self._stage_droppable[env_ids] = False
        self._stage_success[env_ids] = False
        self._stage_dropped[env_ids] = False
        
        # Reset joint targets
        self._joint_targets[env_ids] = self.robot.data.default_joint_pos[env_ids]

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

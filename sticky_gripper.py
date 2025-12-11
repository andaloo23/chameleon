"""
Surface Gripper for Isaac Sim - Based on Isaac Lab's approach.

Key insight: Objects should be positioned RELATIVE to the gripper frame,
not frozen in world space. This maintains proper spatial relationship
as the gripper moves and rotates.

The approach:
1. When grasping, compute the local offset from gripper to object center
2. Each frame, transform this local offset by the gripper's world pose
3. Set object world position = gripper_world_transform * local_offset

IMPORTANT: Only trigger grasp when cube is actually BETWEEN the gripper fingers,
not just within distance. We check the cube's position in the gripper's local frame.
"""

import numpy as np
from typing import Optional, Tuple


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # Normalize
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-10:
        return np.eye(3)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / np.linalg.norm(q)


class StickyGripper:
    """
    Surface-gripper-style object attachment.
    
    Instead of freezing the object in world space (which causes teleportation issues),
    this gripper maintains the object's position RELATIVE to the gripper frame.
    
    Key features:
    - Only triggers grasp when cube is in valid position (between fingers)
    - Computes and stores local offset from gripper to object at grasp time
    - Each frame, transforms local offset to world space using gripper pose
    - Handles gripper rotation properly
    """
    
    def __init__(
        self,
        env,
        grasp_threshold: float = 0.20,           # Max distance to trigger grasp
        gripper_close_threshold: float = 0.45,   # Gripper joint value to consider "closed"
        gripper_open_threshold: float = 0.7,     # Gripper joint value to consider "open"
        min_close_frames: int = 3,               # Frames gripper must be closed to grasp
        release_delay_frames: int = 5,           # Frames gripper must be open to release
        min_standoff: float = 0.015,             # Keep object slightly offset from finger plane to avoid clipping
        detach_distance: float = 0.35,           # Drop the grasp if pose computation explodes
        debug: bool = True,
    ):
        self.env = env
        self.grasp_threshold = grasp_threshold
        self.gripper_close_threshold = gripper_close_threshold
        self.gripper_open_threshold = gripper_open_threshold
        self.min_close_frames = min_close_frames
        self.release_delay_frames = release_delay_frames
        self.min_standoff = min_standoff
        self.detach_distance = detach_distance
        self.debug = debug
        
        # State
        self._is_grasping = False
        self._close_frame_count = 0
        self._open_frame_count = 0
        self._grasped_object_path: Optional[str] = None
        
        # Store the LOCAL offset from gripper frame to object center
        self._local_grasp_offset: Optional[np.ndarray] = None
        self._local_grasp_orientation: Optional[np.ndarray] = None
        self._local_grasp_rotation: Optional[np.ndarray] = None  # Cube orientation relative to gripper
        self._primary_axis: Optional[int] = None
        
        self._update_count = 0
        self._gripper_prim_path: Optional[str] = None
        self._jaw_prim_path: Optional[str] = None
        
    def reset(self):
        """Reset for a new episode."""
        self._is_grasping = False
        self._close_frame_count = 0
        self._open_frame_count = 0
        self._grasped_object_path = None
        self._local_grasp_offset = None
        self._local_grasp_orientation = None
        self._local_grasp_rotation = None
        self._primary_axis = None
        self._update_count = 0
        self._jaw_prim_path = None
        
    def _find_gripper_prim_path(self, stage, robot_prim_path: str) -> Optional[str]:
        """Find the actual gripper prim path in the USD stage."""
        from pxr import UsdGeom
        
        candidate_paths = [
            f"{robot_prim_path}/gripper",
            f"{robot_prim_path}/base/gripper",
        ]
        
        for path in candidate_paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                if self.debug:
                    print(f"[StickyGripper] Found gripper at: {path}")
                return path
        
        if self.debug:
            print(f"[StickyGripper] Searching for gripper prim under {robot_prim_path}...")
        
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim or not robot_prim.IsValid():
            return None
        
        for prim in robot_prim.GetAllChildren():
            prim_name = prim.GetName().lower()
            prim_path = str(prim.GetPath())
            
            if "gripper" in prim_name and "jaw" not in prim_name:
                if self.debug:
                    print(f"[StickyGripper] Found gripper via search: {prim_path}")
                return prim_path
            
            for child in prim.GetAllChildren():
                child_name = child.GetName().lower()
                child_path = str(child.GetPath())
                if "gripper" in child_name and "jaw" not in child_name:
                    if self.debug:
                        print(f"[StickyGripper] Found gripper via deep search: {child_path}")
                    return child_path
        
        return None

    def _find_jaw_prim_path(self, stage, robot_prim_path: str) -> Optional[str]:
        """Find the moving jaw prim (the other finger) to center the grasp pose."""
        candidate_paths = [
            f"{robot_prim_path}/jaw",
            f"{robot_prim_path}/gripper/jaw",
            f"{robot_prim_path}/base/jaw",
        ]
        for path in candidate_paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                if self.debug:
                    print(f"[StickyGripper] Found jaw at: {path}")
                return path
        # Fallback: search for names containing "jaw"
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        if not robot_prim or not robot_prim.IsValid():
            return None
        for prim in robot_prim.GetAllChildren():
            prim_name = prim.GetName().lower()
            prim_path = str(prim.GetPath())
            if "jaw" in prim_name:
                if self.debug:
                    print(f"[StickyGripper] Found jaw via search: {prim_path}")
                return prim_path
        return None
        
    def _get_gripper_world_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get the gripper's world position and orientation."""
        try:
            stage = self.env.world.stage
            robot_prim_path = getattr(self.env.robot, "prim_path", None)
            
            if robot_prim_path is None or stage is None:
                return None, None
            
            from pxr import UsdGeom, Usd
            
            if self._gripper_prim_path is None:
                self._gripper_prim_path = self._find_gripper_prim_path(stage, robot_prim_path)
            if self._jaw_prim_path is None:
                self._jaw_prim_path = self._find_jaw_prim_path(stage, robot_prim_path)
            
            if self._gripper_prim_path is None:
                return None, None
            
            gripper_prim = stage.GetPrimAtPath(self._gripper_prim_path)
            if not gripper_prim.IsValid():
                self._gripper_prim_path = None
                return None, None
            
            xformable = UsdGeom.Xformable(gripper_prim)
            matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            
            translation = matrix.ExtractTranslation()
            position = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
            
            rot_matrix = np.array([
                [matrix[0][0], matrix[0][1], matrix[0][2]],
                [matrix[1][0], matrix[1][1], matrix[1][2]],
                [matrix[2][0], matrix[2][1], matrix[2][2]],
            ], dtype=np.float32)
            
            orientation = rotation_matrix_to_quaternion(rot_matrix)

            # If we can find the jaw link, center the pose between fingers to
            # keep the cube between both grippers instead of sticking to one side.
            if self._jaw_prim_path is not None:
                jaw_prim = stage.GetPrimAtPath(self._jaw_prim_path)
                if jaw_prim and jaw_prim.IsValid():
                    jaw_xformable = UsdGeom.Xformable(jaw_prim)
                    jaw_matrix = jaw_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    jaw_translation = jaw_matrix.ExtractTranslation()
                    jaw_pos = np.array([jaw_translation[0], jaw_translation[1], jaw_translation[2]], dtype=np.float32)
                    # Center between the two fingers; keep orientation of main gripper
                    position = 0.5 * (position + jaw_pos)
                    if self.debug and self._update_count % 60 == 0:
                        print(f"[StickyGripper] Centering pose between gripper and jaw: {position}")
            
            return position, orientation
            
        except Exception as e:
            if self.debug and self._update_count % 100 == 0:
                print(f"[StickyGripper] Error getting gripper pose: {e}")
            return None, None
    
    def _get_cube_world_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get cube world position and orientation."""
        try:
            cube = self.env.cube
            pos, orient = cube.get_world_pose()
            return np.array(pos, dtype=np.float32), np.array(orient, dtype=np.float32)
        except Exception:
            return None, None
    
    def _compute_local_offset(
        self,
        gripper_pos: np.ndarray,
        gripper_orient: np.ndarray,
        object_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the object's position in the gripper's local frame.
        
        local_offset = inverse(gripper_rotation) * (object_pos - gripper_pos)
        """
        gripper_rot = quaternion_to_rotation_matrix(gripper_orient)
        gripper_rot_inv = gripper_rot.T
        world_offset = object_pos - gripper_pos
        local_offset = gripper_rot_inv @ world_offset
        return local_offset

    def _compute_local_orientation(
        self,
        gripper_orient: np.ndarray,
        object_orient: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the object's orientation in the gripper frame.
        
        local_rot = inverse(gripper_rotation) * object_rotation
        """
        gripper_rot = quaternion_to_rotation_matrix(gripper_orient)
        object_rot = quaternion_to_rotation_matrix(object_orient)
        local_rot = gripper_rot.T @ object_rot
        return rotation_matrix_to_quaternion(local_rot)
    
    def _is_valid_grasp_position(
        self,
        gripper_pos: np.ndarray,
        gripper_orient: np.ndarray,
        cube_pos: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Check if the cube is in a valid position for grasping.
        
        The cube should be positioned such that it's roughly "in front of"
        the gripper (between the fingers), not to the side. We check this by
        ensuring one axis dominates the offset - if the cube is between the
        fingers, the offset will be primarily along one axis.
        
        Returns:
            (is_valid, reason_string)
        """
        # Compute cube position in gripper's local frame
        local_pos = self._compute_local_offset(gripper_pos, gripper_orient, cube_pos)
        
        local_x = local_pos[0]
        local_y = local_pos[1]
        local_z = local_pos[2]
        
        abs_x = abs(local_x)
        abs_y = abs(local_y)
        abs_z = abs(local_z)
        
        # The cube should be roughly along ONE primary axis from the gripper
        # (i.e., directly in front, not off to the side)
        # Find which axis has the largest offset
        max_offset = max(abs_x, abs_y, abs_z)
        
        # If max_offset is too small, cube is very close - that's fine
        if max_offset < 0.03:
            return True, f"very close (local=[{local_x:.3f},{local_y:.3f},{local_z:.3f}])"
        
        # Check that the offset is primarily along one axis
        # (cube should be "in front" not "to the side")
        # The two smaller offsets should be significantly less than the largest
        offsets_sorted = sorted([abs_x, abs_y, abs_z], reverse=True)
        primary = offsets_sorted[0]
        secondary = offsets_sorted[1]
        
        # If secondary offset is too large, cube is off to the side.
        # Relaxed to 0.9 to tolerate pose noise / wrist-camera fallback.
        side_ratio = secondary / (primary + 1e-6)
        
        # Log the actual local position for debugging
        local_str = f"local=[{local_x:.3f},{local_y:.3f},{local_z:.3f}]"
        
        if side_ratio > 0.9:
            # Cube is too far to the side relative to forward distance
            return False, f"cube off to side (ratio={side_ratio:.2f}) {local_str}"
        
        # Also check absolute bounds - cube shouldn't be too far in any direction
        if max_offset > 0.30:
            return False, f"cube too far (max={max_offset:.3f}) {local_str}"
        
        return True, f"valid position (ratio={side_ratio:.2f}) {local_str}"
    
    def _transform_local_to_world(
        self,
        gripper_pos: np.ndarray,
        gripper_orient: np.ndarray,
        local_offset: np.ndarray
    ) -> np.ndarray:
        """Transform a local offset to world coordinates using gripper pose."""
        gripper_rot = quaternion_to_rotation_matrix(gripper_orient)
        world_offset = gripper_rot @ local_offset
        return gripper_pos + world_offset
        
    def update(
        self,
        gripper_position: float,
        gripper_world_pos: Optional[np.ndarray] = None,
        object_world_pos: Optional[np.ndarray] = None,
    ) -> bool:
        """Update sticky gripper state each simulation step."""
        self._update_count += 1
        
        # Check gripper state
        is_gripper_closed = gripper_position < self.gripper_close_threshold
        is_gripper_open = gripper_position > self.gripper_open_threshold
        
        # Update frame counters
        if is_gripper_closed:
            self._close_frame_count += 1
            self._open_frame_count = 0
        elif is_gripper_open:
            self._open_frame_count += 1
            self._close_frame_count = 0
        
        # Get gripper and cube poses
        gripper_pos, gripper_orient = self._get_gripper_world_pose()
        used_fallback_pose = False
        cube_pos, cube_orient = self._get_cube_world_pose()
        
        # Use fallbacks if USD lookup failed
        if gripper_pos is None and gripper_world_pos is not None:
            gripper_pos = gripper_world_pos
            gripper_orient = np.array([0, 0, 0, 1], dtype=np.float32)
            used_fallback_pose = True  # orientation is a guess; skip strict validity checks
        if cube_pos is None and object_world_pos is not None:
            cube_pos = object_world_pos
            cube_orient = np.array([0, 0, 0, 1], dtype=np.float32)
        
        # Calculate distance
        distance = None
        if gripper_pos is not None and cube_pos is not None:
            distance = float(np.linalg.norm(gripper_pos - cube_pos))
        
        # Use a tighter threshold when we only have a fallback pose (wrist camera)
        effective_grasp_threshold = self.grasp_threshold
        if used_fallback_pose:
            effective_grasp_threshold = min(self.grasp_threshold, 0.25)
        
        # Check if cube is in valid grasp position
        grasp_position_valid = False
        grasp_check_reason = "no position data"
        if gripper_pos is not None and gripper_orient is not None and cube_pos is not None:
            if used_fallback_pose:
                # Without a reliable orientation, don't enforce the between-fingers check
                grasp_position_valid = True
                grasp_check_reason = "fallback pose (skip position check)"
            else:
                grasp_position_valid, grasp_check_reason = self._is_valid_grasp_position(
                    gripper_pos, gripper_orient, cube_pos
                )
        
        # Debug logging
        in_range = distance is not None and distance < self.grasp_threshold
        should_log = self.debug and (
            self._update_count % 30 == 0 or 
            in_range or 
            (is_gripper_closed and self._close_frame_count <= 10)
        )
        
        if should_log:
            closed_str = "CLOSED" if is_gripper_closed else ("OPEN" if is_gripper_open else "mid")
            dist_str = f"{distance:.3f}m" if distance is not None else "N/A"
            status = "GRASPING" if self._is_grasping else "idle"
            valid_str = "✓" if grasp_position_valid else "✗"
            print(f"[StickyGripper] step={self._update_count} gripper={gripper_position:.3f}({closed_str}) "
                  f"dist={dist_str} pos_valid={valid_str} status={status} "
                  f"frames={self._close_frame_count}/{self.min_close_frames}")
            if not grasp_position_valid and in_range and is_gripper_closed:
                print(f"[StickyGripper]   Position check: {grasp_check_reason}")
        
        # STATE MACHINE
        if not self._is_grasping:
            # Check if we should START grasping
            # IMPORTANT: Cube must be in valid position (between fingers), not just close!
            should_grasp = (
                self._close_frame_count >= self.min_close_frames and
                distance is not None and
                distance <= effective_grasp_threshold and
                grasp_position_valid  # NEW: Must be in valid position!
            )
            
            if should_grasp and cube_pos is not None and cube_orient is not None:
                self._start_grasp(gripper_pos, gripper_orient, cube_pos, cube_orient)
                
        else:
            # Currently grasping - update object position OR release
            if self._open_frame_count >= self.release_delay_frames:
                self._release_grasp()
            elif gripper_pos is not None and gripper_orient is not None:
                self._update_grasped_object(gripper_pos, gripper_orient)
        
        return self._is_grasping
    
    def _start_grasp(
        self,
        gripper_pos: np.ndarray,
        gripper_orient: np.ndarray,
        cube_pos: np.ndarray,
        cube_orient: np.ndarray
    ):
        """Start a new grasp - compute and store local offset."""
        self._is_grasping = True
        self._grasped_object_path = "/World/Cube"
        
        # Compute local offset from gripper to cube
        local_offset = self._compute_local_offset(
            gripper_pos, gripper_orient, cube_pos
        )
        # Maintain a small standoff so the cube does not clip through the finger plane
        primary_axis = int(np.argmax(np.abs(local_offset)))
        self._primary_axis = primary_axis
        axis_sign = 1.0 if local_offset[primary_axis] >= 0 else -1.0
        local_offset[primary_axis] = axis_sign * max(abs(local_offset[primary_axis]), self.min_standoff)

        # Snap lateral axes to the finger midline so the cube sits between both fingers
        for axis in range(3):
            if axis != primary_axis:
                local_offset[axis] = 0.0

        self._local_grasp_offset = local_offset
        self._local_grasp_orientation = cube_orient.copy()
        # Store cube orientation relative to gripper so rotations follow along
        self._local_grasp_rotation = self._compute_local_orientation(gripper_orient, cube_orient)
        
        if self.debug:
            print(f"[StickyGripper] ★★★ GRASP STARTED!")
            print(f"[StickyGripper]   Gripper pos: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
            print(f"[StickyGripper]   Cube pos: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
            print(f"[StickyGripper]   Local offset: [{self._local_grasp_offset[0]:.3f}, {self._local_grasp_offset[1]:.3f}, {self._local_grasp_offset[2]:.3f}]")
    
    def _update_grasped_object(self, gripper_pos: np.ndarray, gripper_orient: np.ndarray):
        """Update the grasped object's position based on gripper pose."""
        if self._local_grasp_offset is None:
            return
        
        try:
            cube = self.env.cube
            
            # Transform local offset to world position
            new_cube_pos = self._transform_local_to_world(
                gripper_pos, gripper_orient, self._local_grasp_offset
            )
            # If the pose computation blew up, drop the grasp to avoid flying cubes
            if np.linalg.norm(new_cube_pos - gripper_pos) > self.detach_distance:
                if self.debug:
                    print(f"[StickyGripper] Detaching – offset exploded (|Δ|={np.linalg.norm(new_cube_pos - gripper_pos):.3f})")
                self._release_grasp()
                return

            # Rotate cube with the gripper using the stored local rotation
            if self._local_grasp_rotation is not None:
                gripper_rot = quaternion_to_rotation_matrix(gripper_orient)
                local_rot = quaternion_to_rotation_matrix(self._local_grasp_rotation)
                world_rot = gripper_rot @ local_rot
                new_cube_orient = rotation_matrix_to_quaternion(world_rot)
            else:
                new_cube_orient = self._local_grasp_orientation
            
            cube.set_world_pose(
                position=new_cube_pos,
                orientation=new_cube_orient
            )
            
            # Zero out velocities to prevent physics fighting
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            
            if self.debug and self._update_count % 60 == 0:
                print(f"[StickyGripper] Holding: cube at [{new_cube_pos[0]:.3f},{new_cube_pos[1]:.3f},{new_cube_pos[2]:.3f}]")
                
        except Exception as e:
            if self.debug:
                print(f"[StickyGripper] Error updating object: {e}")
    
    def _release_grasp(self):
        """Release the grasped object."""
        self._is_grasping = False
        self._grasped_object_path = None
        self._local_grasp_offset = None
        self._local_grasp_orientation = None
        self._local_grasp_rotation = None
        self._primary_axis = None
        
        try:
            cube = self.env.cube
            if cube is not None:
                cube.set_linear_velocity(np.array([0, 0, -0.5]))
        except Exception:
            pass
        
        if self.debug:
            print(f"[StickyGripper] ★★★ GRASP RELEASED!")
    
    @property
    def is_grasping(self) -> bool:
        return self._is_grasping
    
    @property
    def grasped_object(self) -> Optional[str]:
        return self._grasped_object_path if self._is_grasping else None

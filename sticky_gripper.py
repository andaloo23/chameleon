"""
Pre-emptive Sticky Gripper for Isaac Sim.

Freezes the cube BEFORE the gripper can push it away.
When the gripper gets close, we immediately start controlling the cube's position.
"""

import numpy as np
from typing import Optional


class StickyGripper:
    """
    Pre-emptive grasping - freezes cube when gripper approaches.
    """
    
    def __init__(
        self,
        env,
        grasp_threshold: float = 0.40,
        gripper_close_threshold: float = 0.45,
        gripper_open_threshold: float = 0.7,
        min_close_frames: int = 3,
        release_delay_frames: int = 5,
        attachment_stiffness: float = 1000.0,
        # Pre-emptive freeze settings
        preemptive_freeze_distance: float = 0.35,  # Start freezing before grasp
        debug: bool = True,
    ):
        self.env = env
        self.grasp_threshold = grasp_threshold
        self.gripper_close_threshold = gripper_close_threshold
        self.gripper_open_threshold = gripper_open_threshold
        self.min_close_frames = min_close_frames
        self.release_delay_frames = release_delay_frames
        self.attachment_stiffness = attachment_stiffness
        self.preemptive_freeze_distance = preemptive_freeze_distance
        self.debug = debug
        
        # State
        self._is_grasping = False
        self._is_preemptive_freeze = False  # Freezing before grasp
        self._close_frame_count = 0
        self._open_frame_count = 0
        self._grasped_object_path: Optional[str] = None
        
        # Position tracking
        self._frozen_cube_pos: Optional[np.ndarray] = None
        self._frozen_cube_orient: Optional[np.ndarray] = None
        self._prev_gripper_pos: Optional[np.ndarray] = None
        
        self._update_count = 0
        
    def reset(self):
        """Reset for a new episode."""
        self._is_grasping = False
        self._is_preemptive_freeze = False
        self._close_frame_count = 0
        self._open_frame_count = 0
        self._grasped_object_path = None
        self._frozen_cube_pos = None
        self._frozen_cube_orient = None
        self._prev_gripper_pos = None
        self._update_count = 0
        
    def update(
        self,
        gripper_position: float,
        gripper_world_pos: Optional[np.ndarray] = None,
        object_world_pos: Optional[np.ndarray] = None,
    ) -> bool:
        """Update sticky gripper state."""
        self._update_count += 1
        
        # Get gripper link position
        gripper_link_pos = self._get_gripper_link_position()
        if gripper_link_pos is None:
            gripper_link_pos = gripper_world_pos
        
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
        
        # Calculate distance
        distance = None
        if gripper_world_pos is not None and object_world_pos is not None:
            distance = float(np.linalg.norm(gripper_world_pos - object_world_pos))
        
        # Debug
        in_range = distance is not None and distance < self.grasp_threshold
        preemptive_range = distance is not None and distance < self.preemptive_freeze_distance
        if self.debug and (self._update_count % 30 == 0 or in_range or is_gripper_closed):
            closed_str = "CLOSED" if is_gripper_closed else ("OPEN" if is_gripper_open else "mid")
            dist_str = f"{distance:.3f}m" if distance is not None else "N/A"
            status = "GRASPING" if self._is_grasping else ("PRE-FREEZE" if self._is_preemptive_freeze else "idle")
            print(f"[StickyGripper] step={self._update_count} gripper={gripper_position:.3f}({closed_str}) "
                  f"dist={dist_str} status={status} frames={self._close_frame_count}/{self.min_close_frames}")
        
        # PRE-EMPTIVE FREEZE: When gripper gets close, freeze cube in place
        # This prevents the gripper collision from pushing it away
        if not self._is_grasping and not self._is_preemptive_freeze:
            if preemptive_range and is_gripper_closed:
                self._start_preemptive_freeze()
                print(f"[StickyGripper] ⚡ PRE-EMPTIVE FREEZE at dist={distance:.3f}m")
        
        # If pre-emptively frozen, keep the cube in place
        if self._is_preemptive_freeze and not self._is_grasping:
            self._hold_cube_frozen()
            
            # Check if we should transition to full grasp
            should_grasp = (
                self._close_frame_count >= self.min_close_frames and
                distance is not None and
                distance <= self.grasp_threshold
            )
            
            if should_grasp:
                self._transition_to_grasp(gripper_link_pos)
                print(f"[StickyGripper] ★★★ GRASP STARTED! (from pre-freeze)")
            
            # If gripper opens, cancel pre-freeze
            if is_gripper_open:
                self._cancel_preemptive_freeze()
                print(f"[StickyGripper] Pre-freeze cancelled (gripper opened)")
        
        # Full grasping - move cube with gripper
        if self._is_grasping:
            self._move_cube_with_gripper(gripper_link_pos)
            
            if self.debug and self._update_count % 30 == 0:
                print(f"[StickyGripper] HOLDING: cube at {self._frozen_cube_pos}")
            
            if self._open_frame_count >= self.release_delay_frames:
                self._release_grasp()
                print(f"[StickyGripper] ★★★ GRASP RELEASED!")
        
        return self._is_grasping
    
    def _start_preemptive_freeze(self):
        """Start pre-emptive freeze - capture cube position before gripper pushes it."""
        self._is_preemptive_freeze = True
        
        try:
            cube = self.env.cube
            pos, orient = cube.get_world_pose()
            self._frozen_cube_pos = np.array(pos, dtype=np.float32)
            self._frozen_cube_orient = np.array(orient, dtype=np.float32)
            
            if self.debug:
                print(f"[StickyGripper] Frozen cube at: {self._frozen_cube_pos}")
        except Exception as e:
            if self.debug:
                print(f"[StickyGripper] Error freezing: {e}")
    
    def _hold_cube_frozen(self):
        """Keep the cube frozen in place during pre-freeze."""
        if self._frozen_cube_pos is None:
            return
        
        try:
            cube = self.env.cube
            # Force cube to stay at frozen position
            cube.set_world_pose(
                position=self._frozen_cube_pos,
                orientation=self._frozen_cube_orient
            )
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
    
    def _cancel_preemptive_freeze(self):
        """Cancel pre-emptive freeze - let cube fall naturally."""
        self._is_preemptive_freeze = False
        self._frozen_cube_pos = None
        self._frozen_cube_orient = None
    
    def _transition_to_grasp(self, gripper_pos: np.ndarray):
        """Transition from pre-freeze to full grasp."""
        self._is_grasping = True
        self._is_preemptive_freeze = False
        self._grasped_object_path = "/World/Cube"
        self._prev_gripper_pos = gripper_pos.copy() if gripper_pos is not None else None
        # Keep the frozen position - cube is already where we want it
    
    def _move_cube_with_gripper(self, current_gripper_pos: Optional[np.ndarray]):
        """Move cube with gripper using delta tracking."""
        if current_gripper_pos is None or self._prev_gripper_pos is None:
            self._hold_cube_frozen()  # Fallback to just holding
            return
        if self._frozen_cube_pos is None:
            return
        
        try:
            cube = self.env.cube
            
            # Calculate gripper movement
            delta = current_gripper_pos - self._prev_gripper_pos
            
            # Apply to cube
            self._frozen_cube_pos = self._frozen_cube_pos + delta
            
            cube.set_world_pose(
                position=self._frozen_cube_pos,
                orientation=self._frozen_cube_orient
            )
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            
            # Update for next frame
            self._prev_gripper_pos = current_gripper_pos.copy()
            
        except Exception as e:
            if self.debug:
                print(f"[StickyGripper] Error moving: {e}")
    
    def _get_gripper_link_position(self) -> Optional[np.ndarray]:
        """Get gripper link world position."""
        try:
            stage = self.env.world.stage
            robot_prim_path = getattr(self.env.robot, "prim_path", None)
            if robot_prim_path is None or stage is None:
                return None
            
            from pxr import UsdGeom, Usd
            gripper_path = f"{robot_prim_path}/gripper"
            gripper_prim = stage.GetPrimAtPath(gripper_path)
            
            if not gripper_prim.IsValid():
                return None
            
            xformable = UsdGeom.Xformable(gripper_prim)
            matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = matrix.ExtractTranslation()
            
            return np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
        except Exception:
            return None
    
    def _release_grasp(self):
        """Release the grasp."""
        self._is_grasping = False
        self._is_preemptive_freeze = False
        self._grasped_object_path = None
        self._prev_gripper_pos = None
        self._frozen_cube_pos = None
        self._frozen_cube_orient = None
        
        try:
            cube = self.env.cube
            if cube is not None:
                cube.set_linear_velocity(np.array([0, 0, -1.0]))
        except Exception:
            pass
    
    @property
    def is_grasping(self) -> bool:
        return self._is_grasping
    
    @property
    def grasped_object(self) -> Optional[str]:
        return self._grasped_object_path if self._is_grasping else None

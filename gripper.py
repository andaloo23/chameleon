"""gripper.py

Refined gripper logic for Isaac Sim. 
This version implements a behavioral grasp detection system based on 
bounding box overlap (contact detection).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import deque
import numpy as np


# -------------------------
# Math Utilities (Required by load_scene.py)
# -------------------------

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-10:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
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
    return np.array([w, x, y, z], dtype=np.float32)


# -------------------------
# Behavioral Gripper
# -------------------------

class Gripper:
    """Gripper class with behavioral grasp detection based on gripper state and relative motion."""

    # Thresholds for grasp detection
    CLOSE_COMMAND_THRESHOLD = 0.1  # Target below this is considered a "close" command
    STALL_THRESHOLD = 0.001        # If gripper position changes less than this, it's stalled
    LIFT_THRESHOLD = 0.025         # Cube center Z above this = lifted (ground is ~0.02)
    FOLLOWING_THRESHOLD = 0.0005   # Max distance variation to be considered "following" (0.5mm)
    
    # Frame counts for temporal filtering
    FRAMES_TO_GRASP = 15        # N: frames of following required to confirm grasp
    FRAMES_TO_DROP = 30         # M: frames of not following required to confirm drop
    STALL_FRAMES = 5            # Frames of no movement to consider gripper stalled
    
    # Droppable range detection thresholds
    # Cube must be XY-aligned within the cup's inner radius and above the cup
    DROPPABLE_XY_MARGIN = 1.0   # Fraction of inner radius - cube center within this to be "droppable"
    DROPPABLE_MIN_HEIGHT = 0.005 # Minimum height above cup top to be considered in droppable range (5mm)
    
    # In-cup detection thresholds  
    IN_CUP_XY_MARGIN = 1.0      # Fraction of inner radius - cube center within this to be "in cup"
    IN_CUP_HEIGHT_MARGIN = 0.02 # Tolerance for cube settling (2cm)
    
    # Cup collision detection thresholds
    CUP_COLLISION_MARGIN = 0.015  # Distance threshold for cup collision (15mm)

    def __init__(
        self,
        env,
        *,
        dt: float = 1.0 / 120.0,
        debug: bool = True,
    ) -> None:
        self.env = env
        self.dt = float(dt)
        self.debug = bool(debug)
        self._is_grasped: bool = False
        
        # History for "following" detection
        self._history_len = 10
        self._dist_history = deque(maxlen=self._history_len)
        
        # History for gripper stall detection
        self._gripper_value_history = deque(maxlen=self.STALL_FRAMES)
        
        # Track previous target to detect ACTIVE close commands
        self._prev_target_gripper: Optional[float] = None
        self._closing_intent: bool = False  # True when user is actively closing
        
        # Frame counters for temporal filtering
        self._following_frames = 0
        self._not_following_frames = 0
        
        # Droppable range and in-cup detection state
        self._is_droppable_range: bool = False  # True if cube would land in cup if dropped
        self._is_in_cup: bool = False           # True if cube is inside the cup
        self._is_cup_collision: bool = False    # True if gripper is colliding with cup

    @property
    def is_grasping(self) -> bool:
        return self._is_grasped
    
    @property
    def is_droppable_range(self) -> bool:
        """True if cube is above cup and would land inside if dropped."""
        return self._is_droppable_range
    
    @property
    def is_in_cup(self) -> bool:
        """True if cube is currently inside the cup."""
        return self._is_in_cup
    
    @property
    def is_cup_collision(self) -> bool:
        """True if gripper is colliding with cup."""
        return self._is_cup_collision

    @property
    def last_state(self) -> dict:
        """Return last computed state for debugging."""
        return getattr(self, "_last_state", {})

    def reset(self) -> None:
        self._is_grasped = False
        self._dist_history.clear()
        self._gripper_value_history.clear()
        self._prev_target_gripper = None
        self._closing_intent = False
        self._following_frames = 0
        self._not_following_frames = 0
        self._is_droppable_range = False
        self._is_in_cup = False
        self._is_cup_collision = False
        self._last_state = {}

    def _is_following(self) -> bool:
        """
        Check if cube is 'following' the gripper (constant relative distance).
        
        Following is true when the relative distance is stable.
        Combined with 'lifted', this prevents false positives.
        """
        if len(self._dist_history) < self._history_len:
            return False
        
        dists = list(self._dist_history)
        dist_variation = max(dists) - min(dists)
        return dist_variation < self.FOLLOWING_THRESHOLD

    def _is_gripper_stalled(self) -> bool:
        """Check if gripper has stopped moving (stalled on object or limit)."""
        if len(self._gripper_value_history) < self.STALL_FRAMES:
            return False
        vals = list(self._gripper_value_history)
        variation = max(vals) - min(vals)
        return variation < self.STALL_THRESHOLD

    def update(
        self,
        *,
        gripper_value: Optional[float] = None,
        target_gripper: Optional[float] = None,
        gripper_world_pos: Optional[np.ndarray] = None,
        jaw_world_pos: Optional[np.ndarray] = None,
        object_world_pos: Optional[np.ndarray] = None,
        arm_moving: bool = False,
        cup_pos: Optional[np.ndarray] = None,
        cup_height: float = 0.0,
        cup_inner_radius: float = 0.0,
        cup_outer_radius: float = 0.0,
        cube_half_size: float = 0.0,
    ) -> bool:
        """
        Update grasp state using behavioral detection:
        
        If not grasped:
            if closed AND lifted AND following_for_N:
                grasped = True
        else (grasped):
            if not following_for_M:
                grasped = False (dropped or released)
        """
        # 1. Detect ACTIVE close/open intent based on target changes
        # Closing intent is set when target decreases toward close position
        # Opening intent clears closing_intent
        if target_gripper is not None:
            if self._prev_target_gripper is not None:
                if target_gripper < self._prev_target_gripper:
                    # Target is decreasing (moving toward close)
                    self._closing_intent = True
                elif target_gripper > self._prev_target_gripper:
                    # Target is increasing (moving toward open)
                    self._closing_intent = False
            self._prev_target_gripper = target_gripper
        
        # 2. Check if gripper is stalled (position not changing)
        stalled = False
        if gripper_value is not None:
            self._gripper_value_history.append(float(gripper_value))
            stalled = self._is_gripper_stalled()
        
        # Closed = actively trying to close AND gripper has stalled
        closed = self._closing_intent and stalled
        
        # 2. Check if cube is lifted off the ground
        lifted = False
        cube_z = 0.0
        if object_world_pos is not None:
            cube_z = float(object_world_pos[2])
            lifted = cube_z > self.LIFT_THRESHOLD
        
        # 3. Update distance history for following detection
        following = False
        if gripper_world_pos is not None and object_world_pos is not None:
            curr_dist = float(np.linalg.norm(gripper_world_pos - object_world_pos))
            self._dist_history.append(curr_dist)
            following = self._is_following()
        
        # 4. Droppable range detection: cube is above cup and would land inside if dropped
        droppable_range = False
        in_cup = False
        cube_cup_xy_dist = None
        cup_top_z = 0.0
        
        if cup_pos is not None and object_world_pos is not None and cup_inner_radius > 0:
            # Calculate XY distance from cube center to cup center
            cube_xy = object_world_pos[:2]
            cup_xy = cup_pos[:2]
            cube_cup_xy_dist = float(np.linalg.norm(cube_xy - cup_xy))
            
            # Cup top Z position
            cup_top_z = float(cup_pos[2]) + cup_height
            cup_bottom_z = float(cup_pos[2])
            
            # Use inner radius directly for XY checks (cube center must be within this)
            effective_radius = cup_inner_radius
            
            # Droppable range: cube center XY within inner radius AND cube bottom is above cup top
            cube_bottom_z = cube_z - cube_half_size
            xy_in_range = cube_cup_xy_dist <= effective_radius * self.DROPPABLE_XY_MARGIN
            above_cup = cube_bottom_z > cup_top_z + self.DROPPABLE_MIN_HEIGHT
            droppable_range = xy_in_range and above_cup
            
            # In-cup detection: cube center XY within cup AND cube bottom is inside cup (between bottom and top)
            xy_in_cup = cube_cup_xy_dist <= effective_radius * self.IN_CUP_XY_MARGIN
            # Cube is "in cup" if its bottom is above cup bottom and below cup top
            cube_inside_height = (cube_bottom_z >= cup_bottom_z - self.IN_CUP_HEIGHT_MARGIN and
                                  cube_bottom_z <= cup_top_z)
            in_cup = xy_in_cup and cube_inside_height
        
        # 5. Cup collision detection: gripper or jaw collides with cup
        cup_collision = False
        if cup_pos is not None and cup_outer_radius > 0 and cup_height > 0:
            cup_xy = cup_pos[:2]
            cup_bottom_z = float(cup_pos[2])
            cup_top_z = cup_bottom_z + cup_height
            
            # Check gripper collision
            if gripper_world_pos is not None:
                gripper_xy = gripper_world_pos[:2]
                gripper_z = float(gripper_world_pos[2])
                gripper_xy_dist = float(np.linalg.norm(gripper_xy - cup_xy))
                
                # Collision if: within outer radius + margin, at cup height level, and within wall zone
                at_cup_height = cup_bottom_z <= gripper_z <= cup_top_z + self.CUP_COLLISION_MARGIN
                near_wall = (cup_inner_radius - self.CUP_COLLISION_MARGIN <= gripper_xy_dist <= 
                            cup_outer_radius + self.CUP_COLLISION_MARGIN)
                inside_cup = gripper_xy_dist < cup_inner_radius and at_cup_height
                
                if at_cup_height and (near_wall or inside_cup):
                    cup_collision = True
            
            # Check jaw collision
            if jaw_world_pos is not None and not cup_collision:
                jaw_xy = jaw_world_pos[:2]
                jaw_z = float(jaw_world_pos[2])
                jaw_xy_dist = float(np.linalg.norm(jaw_xy - cup_xy))
                
                at_cup_height = cup_bottom_z <= jaw_z <= cup_top_z + self.CUP_COLLISION_MARGIN
                near_wall = (cup_inner_radius - self.CUP_COLLISION_MARGIN <= jaw_xy_dist <= 
                            cup_outer_radius + self.CUP_COLLISION_MARGIN)
                inside_cup = jaw_xy_dist < cup_inner_radius and at_cup_height
                
                if at_cup_height and (near_wall or inside_cup):
                    cup_collision = True
        
        # Event-based debug output: only print when state changes
        if droppable_range and not self._is_droppable_range:
            print("droppable detected")
        if in_cup and not self._is_in_cup:
            print("cube is in cup")
        if cup_collision and not self._is_cup_collision:
            print("cup collision detected")
        
        self._is_droppable_range = droppable_range
        self._is_in_cup = in_cup
        self._is_cup_collision = cup_collision
        
        # Store state for external debugging
        self._last_state = {
            "closed": closed,
            "closing_intent": self._closing_intent,
            "stalled": stalled,
            "lifted": lifted,
            "following": following,
            "cube_z": cube_z,
            "following_frames": self._following_frames,
            "not_following_frames": self._not_following_frames,
            "droppable_range": droppable_range,
            "in_cup": in_cup,
            "cup_collision": cup_collision,
            "cube_cup_xy_dist": cube_cup_xy_dist,
            "cup_top_z": cup_top_z,
        }
        
        # 5. Apply grasp detection logic
        if not self._is_grasped:
            # To become grasped: need closed + lifted + following for N frames
            if closed and lifted and following:
                self._following_frames += 1
                if self._following_frames >= self.FRAMES_TO_GRASP:
                    self._is_grasped = True
                    self._following_frames = 0
                    print("grasp detected")
            else:
                self._following_frames = 0
        else:
            # To lose grasp: not following for M frames
            if not following:
                self._not_following_frames += 1
                if self._not_following_frames >= self.FRAMES_TO_DROP:
                    self._is_grasped = False
                    self._not_following_frames = 0
                    print("drop detected")
            else:
                self._not_following_frames = 0
        
        return self._is_grasped


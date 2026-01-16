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
# Legacy Compatibility (Required by reward_engine.py)
# -------------------------

@dataclass
class GraspState:
    grasped: bool = False
    confidence: float = 0.0
    stable_frames: int = 0
    was_closing: bool = False
    position_stable: bool = False
    position_blocked: bool = False
    grasp_position: Optional[float] = None


class GraspDetector:
    """Minimal compatibility class for reward_engine.py."""
    def __init__(self, **kwargs):
        self._grasped = False
        self._history = deque(maxlen=15)
        self.min_stable_frames = kwargs.get("min_stable_frames", 5)

    def reset(self):
        self._grasped = False
        self._history.clear()

    def update(self, gripper_position, target_position=None):
        self._history.append(gripper_position)
        # Dummy update that always returns a state
        return GraspState(grasped=self._grasped)

    @property
    def is_grasped(self): return self._grasped

    @property
    def grasp_position(self): return None


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

    @property
    def is_grasping(self) -> bool:
        return self._is_grasped

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
        }
        
        # 4. Apply grasp detection logic
        if not self._is_grasped:
            # To become grasped: need closed + lifted + following for N frames
            if closed and lifted and following:
                self._following_frames += 1
                if self._following_frames >= self.FRAMES_TO_GRASP:
                    self._is_grasped = True
                    self._following_frames = 0
                    if self.debug:
                        print(f"[GRASP] Detected grasp! closed={closed}, lifted={lifted}, cube_z={cube_z:.4f}")
            else:
                self._following_frames = 0
        else:
            # To lose grasp: not following for M frames
            if not following:
                self._not_following_frames += 1
                if self._not_following_frames >= self.FRAMES_TO_DROP:
                    self._is_grasped = False
                    self._not_following_frames = 0
                    if self.debug:
                        print(f"[GRASP] Detected drop! following={following}, cube_z={cube_z:.4f}")
            else:
                self._not_following_frames = 0
        
        return self._is_grasped


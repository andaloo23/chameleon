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
    CLOSED_THRESHOLD = 0.3      # Gripper position below this = closed (0=fully closed, 1=open)
    LIFT_THRESHOLD = 0.025      # Cube center Z above this = lifted (ground is ~0.02)
    FOLLOWING_THRESHOLD = 0.001 # Max distance variation to be considered "following" (1mm)
    
    # Frame counts for temporal filtering
    FRAMES_TO_GRASP = 15        # N: frames of following required to confirm grasp
    FRAMES_TO_DROP = 30         # M: frames of not following required to confirm drop

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
        self._following_frames = 0
        self._not_following_frames = 0
        self._last_state = {}

    def _is_following(self) -> bool:
        """Check if cube is 'following' the gripper (constant relative distance)."""
        if len(self._dist_history) < self._history_len:
            return False
        dists = list(self._dist_history)
        variation = max(dists) - min(dists)
        return variation < self.FOLLOWING_THRESHOLD

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
        # 1. Check if gripper is closed
        closed = False
        if gripper_value is not None:
            closed = gripper_value < self.CLOSED_THRESHOLD
        
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


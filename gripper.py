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
    """Gripper class handling physics-based grasp detection using contact reporting."""

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
        self._contact_reporter = None
        self._gripper_path = "/World/so_arm100/gripper"
        self._jaw_path = "/World/so_arm100/jaw"
        self._cube_path = "/World/Cube"
        self._history_n = 10  # Use a longer window for stability
        self._condition_met_frames = 0
        self._condition_failed_frames = 0
        self._dist_history = deque(maxlen=self._history_n)
        self._lift_threshold = 0.025  # Cube center is at 0.02 when on ground (0.04 height)
        self._stability_threshold = 0.0005  # 0.5mm strict stability

    def _ensure_contact_reporter(self):
        """Sync paths with environment."""
        if hasattr(self.env, "_gripper_prim_path") and self.env._gripper_prim_path:
            self._gripper_path = self.env._gripper_prim_path
        if hasattr(self.env, "_jaw_prim_path") and self.env._jaw_prim_path:
            self._jaw_path = self.env._jaw_prim_path
        return True

    def _check_gripper_contact(self) -> bool:
        """Check if gripper (fixed jaw) is in contact using ContactSensor."""
        sensor = getattr(self.env, "gripper_contact_sensor", None)
        if sensor is None:
            return False
        try:
            reading = sensor.get_current_frame()
            # is_valid indicates the sensor detected something
            return reading.get("in_contact", False) or reading.get("value", 0.0) > 0.0
        except Exception:
            return False

    def _check_jaw_contact(self) -> bool:
        """Check if jaw (moving jaw) is in contact using ContactSensor."""
        sensor = getattr(self.env, "jaw_contact_sensor", None)
        if sensor is None:
            return False
        try:
            reading = sensor.get_current_frame()
            return reading.get("in_contact", False) or reading.get("value", 0.0) > 0.0
        except Exception:
            return False


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
        self._condition_met_frames = 0
        self._condition_failed_frames = 0
        self._last_state = {}

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
        """Update grasp state based on physical contact detection and stability."""
        self._ensure_contact_reporter()
        
        # 1. Check if both gripper and jaw are contacting using ContactSensor
        gripper_contact = self._check_gripper_contact()
        jaw_contact = self._check_jaw_contact()
        contact_met = gripper_contact and jaw_contact


        
        # 2. Check relative distance stability using a window
        stability_met = False
        if gripper_world_pos is not None and object_world_pos is not None:
            curr_dist = float(np.linalg.norm(gripper_world_pos - object_world_pos))
            self._dist_history.append(curr_dist)
            
            if len(self._dist_history) == self._history_n:
                # Relative distance is stable if max variation in window is small
                dists = list(self._dist_history)
                variation = max(dists) - min(dists)
                stability_met = variation < self._stability_threshold
        
        # 3. Check if cube is off the ground
        lift_met = False
        cube_z = 0.0
        if object_world_pos is not None:
            cube_z = float(object_world_pos[2])
            lift_met = cube_z > self._lift_threshold
            
        # Store state for external access
        self._last_state = {
            "contact": contact_met,
            "stability": stability_met,
            "lift": lift_met,
            "cube_z": cube_z,
        }
            
        # Combine conditions
        all_conditions_met = contact_met and stability_met and lift_met
        
        # Update temporal window
        if not self._is_grasped:
            if all_conditions_met:
                self._condition_met_frames += 1
                if self._condition_met_frames >= 30: # Requirement: sustained for 30 frames
                    self._is_grasped = True
                    self._condition_met_frames = 0
            else:
                self._condition_met_frames = 0
        else:
            # If already grasped, check if any condition fails (allow some jitter)
            if not (contact_met or stability_met): # Drop if we lose both contact and stability
                self._condition_failed_frames += 1
                if self._condition_failed_frames >= 30:
                    self._is_grasped = False
                    self._condition_failed_frames = 0
            else:
                self._condition_failed_frames = 0
        
        return self._is_grasped


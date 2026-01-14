"""gripper.py

Refined gripper logic for Isaac Sim. 
This version removes physics welds (sticky gripper) and implements a behavioral 
grasp detection system based on gripper stall and relative distance stability.
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
    """Gripper class handling behavioral grasp detection."""

    def __init__(
        self,
        env,
        *,
        dt: float = 1.0 / 120.0,
        stall_threshold_m: float = 0.001,
        contact_limit_m: float = 0.005,
        distance_stability_threshold: float = 0.002,
        cube_movement_threshold: float = 0.003,  # 3mm movement = cube is moving
        ground_z_threshold: float = 0.025, # Center Z of 4cm cube is 0.02 on ground
        history_len: int = 15,
        debug: bool = True,
    ) -> None:
        self.env = env
        self.dt = float(dt)
        self.stall_threshold_m = stall_threshold_m
        self.contact_limit_m = contact_limit_m
        self.distance_stability_threshold = distance_stability_threshold
        self.cube_movement_threshold = cube_movement_threshold
        self.ground_z_threshold = ground_z_threshold
        self.debug = bool(debug)

        self._gap_history: deque = deque(maxlen=history_len)
        self._distance_history: deque = deque(maxlen=5) 
        self._cube_pos_history: deque = deque(maxlen=5)  # Track cube positions
        self._is_grasped: bool = False

    @property
    def is_grasping(self) -> bool:
        return self._is_grasped

    def reset(self) -> None:
        self._gap_history.clear()
        self._distance_history.clear()
        self._cube_pos_history.clear()
        self._is_grasped = False

    def update(
        self,
        *,
        gripper_value: Optional[float],
        target_gripper: Optional[float],
        gripper_world_pos: Optional[np.ndarray],
        jaw_world_pos: Optional[np.ndarray],
        object_world_pos: Optional[np.ndarray],
        arm_moving: bool = False,
    ) -> bool:
        if gripper_world_pos is None or jaw_world_pos is None or object_world_pos is None:
            self._is_grasped = False
            return False

        gap = float(np.linalg.norm(np.asarray(gripper_world_pos) - np.asarray(jaw_world_pos)))
        self._gap_history.append(gap)

        stalled = False
        if len(self._gap_history) >= 5:
            gap_range = max(self._gap_history) - min(self._gap_history)
            stalled = gap_range < self.stall_threshold_m
        
        is_closing_cmd = False
        if gripper_value is not None and target_gripper is not None:
            is_closing_cmd = target_gripper < gripper_value - 1e-3

        is_contacting = stalled and is_closing_cmd and gap > self.contact_limit_m

        gripper_center = 0.5 * (np.asarray(gripper_world_pos) + np.asarray(jaw_world_pos))
        dist_to_obj = float(np.linalg.norm(gripper_center - np.asarray(object_world_pos)))
        self._distance_history.append(dist_to_obj)
        
        # Track cube position for movement detection
        self._cube_pos_history.append(np.asarray(object_world_pos).copy())
        
        # Detect if cube is moving
        cube_moving = False
        if len(self._cube_pos_history) >= 3:
            positions = list(self._cube_pos_history)
            movement = np.linalg.norm(positions[-1] - positions[0])
            cube_moving = movement > self.cube_movement_threshold

        if len(self._distance_history) >= 5:
            dist_std = np.std(self._distance_history)
            is_dist_constant = dist_std < self.distance_stability_threshold
            
            # Grasp requires:
            # 1. Constant distance between jaws and cube center
            # 2. Arm is moving (lifting)
            # 3. Cube Z has lifted off the ground
            # 4. Currently contacting or already grasped
            cube_is_lifted = object_world_pos[2] > self.ground_z_threshold
            
            if is_dist_constant and arm_moving and cube_is_lifted and (is_contacting or self._is_grasped):
                self._is_grasped = True
            else:
                # Release if: (arm_moving OR cube_moving) AND NOT is_dist_constant
                if (arm_moving or cube_moving) and not is_dist_constant:
                    self._is_grasped = False

        return self._is_grasped

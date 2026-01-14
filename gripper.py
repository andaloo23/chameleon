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
        self._gripper_path = None
        self._jaw_path = None
        self._cube_path = "/World/Cube"

    def _ensure_contact_reporter(self):
        """Lazily initialize the contact reporter."""
        if self._contact_reporter is not None:
            return True
            
        try:
            from omni.physx import get_physx_scene_query_interface
            self._contact_reporter = get_physx_scene_query_interface()
            
            # Find gripper and jaw paths
            robot_path = getattr(self.env.robot, 'prim_path', '/World/so_arm100')
            self._gripper_path = f"{robot_path}/gripper"
            self._jaw_path = f"{robot_path}/jaw"
            return True
        except Exception as e:
            if self.debug:
                print(f"[WARN] Could not initialize contact reporter: {e}")
            return False

    def _check_contact(self, link_path: str, target_path: str) -> bool:
        """Check if link_path is in contact with target_path using overlap detection."""
        try:
            # Use bounding box overlap as a simpler contact check
            stage = self.env.world.stage
            link_prim = stage.GetPrimAtPath(link_path)
            target_prim = stage.GetPrimAtPath(target_path)
            
            if not link_prim.IsValid() or not target_prim.IsValid():
                return False
            
            from pxr import UsdGeom, Gf
            
            # Get world bounds for both prims
            link_bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(link_prim)
            target_bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(target_prim)
            
            link_range = link_bbox.GetBox()
            target_range = target_bbox.GetBox()
            
            if link_range.IsEmpty() or target_range.IsEmpty():
                return False
            
            # Expand boxes slightly for contact margin
            margin = 0.001  # 1mm contact margin
            link_min = link_range.GetMin() - Gf.Vec3d(margin, margin, margin)
            link_max = link_range.GetMax() + Gf.Vec3d(margin, margin, margin)
            target_min = target_range.GetMin()
            target_max = target_range.GetMax()
            
            # Check AABB overlap
            overlap = (link_min[0] <= target_max[0] and link_max[0] >= target_min[0] and
                      link_min[1] <= target_max[1] and link_max[1] >= target_min[1] and
                      link_min[2] <= target_max[2] and link_max[2] >= target_min[2])
            
            return overlap
            
        except Exception as e:
            if self.debug:
                print(f"[WARN] Contact check error: {e}")
            return False

    @property
    def is_grasping(self) -> bool:
        return self._is_grasped

    def reset(self) -> None:
        self._is_grasped = False

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
        """Update grasp state based on physical contact detection."""
        self._ensure_contact_reporter()
        
        # Check if both gripper and jaw are contacting the cube
        gripper_contact = self._check_contact(self._gripper_path, self._cube_path)
        jaw_contact = self._check_contact(self._jaw_path, self._cube_path)
        
        if self.debug:
            print(f"[GRASP DEBUG] gripper_contact={gripper_contact}, jaw_contact={jaw_contact}")
        
        # Grasp = both parts touching the cube
        self._is_grasped = gripper_contact and jaw_contact
        
        return self._is_grasped

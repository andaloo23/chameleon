"""gripper.py

Consolidated gripper logic for Isaac Sim, incorporating:
- Position-based stall/grasp detection (formerly grasp_detector.py)
- Physics-based weld joint creation (formerly gripper_weld.py)
- Improved spatial gating (inspired by sticky_gripper.py)

This module provides an intelligent, object-agnostic grasping system that creates 
a physics FixedJoint (a "weld") between the gripper and an object only after 
evidence of a stable grasp is detected.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import numpy as np


# -------------------------
# Math Utilities
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
# Grasp Detection
# -------------------------

@dataclass
class GraspState:
    """Current state of grasp detection."""
    grasped: bool = False
    confidence: float = 0.0
    stable_frames: int = 0
    closing_intent: bool = False
    position_blocked: bool = False
    position_stable: bool = False
    gripper_position: float = 0.0
    target_position: float = 0.0
    was_closing: bool = False


class GraspDetector:
    """
    Detects grasping using position stall detection.
    
    Identifies when the gripper:
    1. Was actively closing (position decreasing)
    2. Has now stalled (position stable)
    3. Is not at the fully closed position (blocked by object)
    """
    
    def __init__(
        self,
        history_length: int = 15,
        stability_threshold: float = 0.015,
        min_stable_frames: int = 5,
        min_gripper_position: float = 0.005,
        max_gripper_for_grasp: float = 1.2,
        closing_velocity_threshold: float = 0.002,
    ):
        self.history_length = history_length
        self.stability_threshold = stability_threshold
        self.min_stable_frames = min_stable_frames
        self.min_gripper_position = min_gripper_position
        self.max_gripper_for_grasp = max_gripper_for_grasp
        self.closing_velocity_threshold = closing_velocity_threshold
        
        self._position_history: deque = deque(maxlen=history_length)
        self._target_history: deque = deque(maxlen=history_length)
        self._stable_frame_count: int = 0
        self._grasped: bool = False
        self._grasp_position: Optional[float] = None
        self._was_closing: bool = False
        self._closing_detected_at_position: Optional[float] = None
        
    def reset(self) -> None:
        self._position_history.clear()
        self._target_history.clear()
        self._stable_frame_count = 0
        self._grasped = False
        self._grasp_position = None
        self._was_closing = False
        self._closing_detected_at_position = None
        
    def update(
        self,
        gripper_position: float,
        target_position: Optional[float] = None,
    ) -> GraspState:
        self._position_history.append(gripper_position)
        if target_position is not None:
            self._target_history.append(target_position)
        
        if len(self._position_history) < 5:
            return GraspState(
                grasped=self._grasped,
                gripper_position=gripper_position,
                target_position=target_position or gripper_position,
            )
        
        is_currently_closing = self._check_is_closing()
        if is_currently_closing and not self._was_closing:
            self._was_closing = True
            self._closing_detected_at_position = gripper_position
        
        position_stable = self._check_position_stability()
        position_blocked = self._check_position_blocked(gripper_position)
        closing_intent = is_currently_closing or (self._was_closing and position_stable)
        
        grasp_conditions_met = (
            self._was_closing and 
            position_stable and 
            position_blocked
        )
        
        if grasp_conditions_met:
            self._stable_frame_count += 1
        else:
            if not self._grasped:
                self._stable_frame_count = max(0, self._stable_frame_count - 1)
                if self._closing_detected_at_position is not None:
                    if gripper_position > self._closing_detected_at_position + 0.1:
                        self._was_closing = False
                        self._closing_detected_at_position = None
        
        confidence = min(1.0, self._stable_frame_count / self.min_stable_frames)
        
        if not self._grasped:
            if self._stable_frame_count >= self.min_stable_frames:
                self._grasped = True
                self._grasp_position = gripper_position
                print(f"[GRASP DETECTOR] ✓ Grasp detected at position {gripper_position:.4f}")
        else:
            if self._check_grasp_released(gripper_position):
                self._grasped = False
                self._grasp_position = None
                self._stable_frame_count = 0
                self._was_closing = False
                self._closing_detected_at_position = None
        
        return GraspState(
            grasped=self._grasped,
            confidence=confidence,
            stable_frames=self._stable_frame_count,
            closing_intent=closing_intent,
            position_blocked=position_blocked,
            position_stable=position_stable,
            gripper_position=gripper_position,
            target_position=target_position or gripper_position,
            was_closing=self._was_closing,
        )
    
    def _check_is_closing(self) -> bool:
        if len(self._position_history) < 5:
            return False
        positions = list(self._position_history)
        recent = positions[-5:]
        velocities = [recent[i] - recent[i+1] for i in range(len(recent)-1)]
        avg_velocity = np.mean(velocities)
        return avg_velocity > self.closing_velocity_threshold
    
    def _check_position_stability(self) -> bool:
        if len(self._position_history) < 5:
            return False
        recent_positions = list(self._position_history)[-7:]
        if len(recent_positions) < 5:
            return False
        position_range = max(recent_positions) - min(recent_positions)
        return position_range < self.stability_threshold
    
    def _check_position_blocked(self, gripper_position: float) -> bool:
        return (
            gripper_position > self.min_gripper_position
            and gripper_position < self.max_gripper_for_grasp
        )
    
    def _check_grasp_released(self, current_position: float) -> bool:
        if self._grasp_position is None:
            return True
        release_threshold = 0.2
        return current_position > self._grasp_position + release_threshold
    
    @property
    def is_grasped(self) -> bool:
        return self._grasped
    
    @property
    def grasp_position(self) -> Optional[float]:
        return self._grasp_position


# -------------------------
# Gripper Interface
# -------------------------

@dataclass
class GripperDebug:
    closing: bool
    stall: bool
    finger_gap: Optional[float]
    gap_range: Optional[float]
    near_object: bool
    moving_contact: bool
    stationary_contact: bool
    distance: Optional[float]


class Gripper:
    """Consolidated Gripper class handling physics welds and stall detection."""

    def __init__(
        self,
        env,
        *,
        dt: float = 1.0 / 120.0,
        stall_time_s: float = 0.25,
        stall_gap_range_m: float = 0.0015,
        near_distance_m: float = 0.09,
        close_command_margin: float = 1e-3,
        open_release_threshold: float = 0.55,
        debug: bool = True,
        joint_path: str = "/World/GripperWeldJoint",
    ) -> None:
        self.env = env
        self.dt = float(dt)
        self.stall_time_s = float(stall_time_s)
        self.stall_gap_range_m = float(stall_gap_range_m)
        self.near_distance_m = float(near_distance_m)
        self.close_command_margin = float(close_command_margin)
        self.open_release_threshold = float(open_release_threshold)
        self.debug = bool(debug)
        self.joint_path = str(joint_path)

        self._gap_history: list[float] = []
        self._stall_accum_s: float = 0.0
        self._is_welded: bool = False
        self._step: int = 0
        self._grasped_object_path: Optional[str] = None

    @property
    def is_grasping(self) -> bool:
        return self._is_welded

    @property
    def grasped_object(self) -> Optional[str]:
        return self._grasped_object_path

    def reset(self) -> None:
        self._gap_history.clear()
        self._stall_accum_s = 0.0
        self._is_welded = False
        self._step = 0
        self._grasped_object_path = None
        self.release()

    def update(
        self,
        *,
        gripper_value: Optional[float],
        target_gripper: Optional[float],
        gripper_world_pos: Optional[np.ndarray],
        gripper_world_orient: Optional[np.ndarray],
        jaw_world_pos: Optional[np.ndarray],
        object_world_pos: Optional[np.ndarray],
        object_world_orient: Optional[np.ndarray],
        object_prim_path: str,
        gripper_body_path: str,
        jaw_body_path: Optional[str] = None,
    ) -> GripperDebug:
        self._step += 1

        # Release if user opens the gripper
        if self._is_welded and gripper_value is not None and gripper_value > self.open_release_threshold:
            if self.debug:
                print(f"[GRIPPER] Release: gripper opened (val={gripper_value:.3f} > {self.open_release_threshold})")
            self.release()
            return GripperDebug(False, False, None, None, False, False, False, None)

        if gripper_world_pos is None or jaw_world_pos is None or object_world_pos is None:
            self._stall_accum_s = 0.0
            self._gap_history.clear()
            return GripperDebug(False, False, None, None, False, False, False, None)

        gap = float(np.linalg.norm(np.asarray(gripper_world_pos) - np.asarray(jaw_world_pos)))
        
        closing = False
        if gripper_value is not None and target_gripper is not None:
            closing = bool(target_gripper < gripper_value - self.close_command_margin)

        dist_stationary = float(np.linalg.norm(np.asarray(gripper_world_pos) - np.asarray(object_world_pos)))
        dist_moving = float(np.linalg.norm(np.asarray(jaw_world_pos) - np.asarray(object_world_pos)))
        
        effective_near_m = self.near_distance_m if gap > 0.001 else 0.08
        
        # We consider contact if the part is within the near threshold of the cube
        # The cube half-size is ~0.02, and the gripper center to tip is some distance.
        # near_distance_m is default 0.30 in load_scene, which is quite large.
        # Let's use a tighter threshold for "contact" if near_distance_m is meant to be spatial gating.
        # But for "contact", we want it to be very close.
        contact_threshold = 0.045 # Adjusted for 4cm cube + some margin
        
        moving_contact = bool(dist_moving <= contact_threshold)
        stationary_contact = bool(dist_stationary <= contact_threshold)
        near_object = moving_contact and stationary_contact

        window_n = max(2, int(round(self.stall_time_s / max(self.dt, 1e-6))))
        self._gap_history.append(gap)
        if len(self._gap_history) > window_n:
            self._gap_history.pop(0)

        gap_range = None
        stall = False
        if len(self._gap_history) >= window_n:
            gmin, gmax = float(min(self._gap_history)), float(max(self._gap_history))
            gap_range = gmax - gmin
            stall = bool(gap_range <= self.stall_gap_range_m)

        if closing and near_object and stall and not self._is_welded:
            increment = self.dt if gap > 0.001 else 0.5 * self.dt
            self._stall_accum_s += increment
        else:
            self._stall_accum_s = max(0.0, self._stall_accum_s - 2.0 * self.dt)

        if (not self._is_welded) and (self._stall_accum_s >= self.stall_time_s):
            g_orient = gripper_world_orient if gripper_world_orient is not None else np.array([1, 0, 0, 0], dtype=np.float32)
            o_orient = object_world_orient if object_world_orient is not None else np.array([1, 0, 0, 0], dtype=np.float32)
            anchor_world = np.asarray(object_world_pos)
            
            self._create_fixed_joint(
                anchor_world=anchor_world,
                gripper_world_pos=np.asarray(gripper_world_pos),
                gripper_world_orient=np.asarray(g_orient),
                object_world_pos=np.asarray(object_world_pos),
                object_world_orient=np.asarray(o_orient),
                object_prim_path=object_prim_path,
                gripper_body_path=gripper_body_path,
            )
            self._is_welded = True
            self._grasped_object_path = object_prim_path
            if self.debug:
                print(f"[GRIPPER] ★★★ Weld created (stall confirmed, dual-contact, gap={gap:.4f})")

        return GripperDebug(closing, stall, gap, gap_range, near_object, moving_contact, stationary_contact, dist_moving)

    def release(self) -> None:
        try:
            stage = self.env.world.stage
            prim = stage.GetPrimAtPath(self.joint_path)
            if prim and prim.IsValid():
                stage.RemovePrim(prim.GetPath())
        except Exception:
            pass
        self._is_welded = False
        self._stall_accum_s = 0.0
        self._gap_history.clear()
        self._grasped_object_path = None

    def _create_fixed_joint(
        self,
        *,
        anchor_world: np.ndarray,
        gripper_world_pos: np.ndarray,
        gripper_world_orient: np.ndarray,
        object_world_pos: np.ndarray,
        object_world_orient: np.ndarray,
        object_prim_path: str,
        gripper_body_path: str,
    ) -> None:
        stage = self.env.world.stage
        from pxr import UsdPhysics, Gf

        Rg, Ro = quaternion_to_rotation_matrix(gripper_world_orient), quaternion_to_rotation_matrix(object_world_orient)
        anchor_local_g = Rg.T @ (anchor_world - gripper_world_pos)
        anchor_local_o = Ro.T @ (anchor_world - object_world_pos)

        joint = UsdPhysics.FixedJoint.Define(stage, self.joint_path)
        joint.CreateBody0Rel().SetTargets([gripper_body_path])
        joint.CreateBody1Rel().SetTargets([object_prim_path])

        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(anchor_local_g[0]), float(anchor_local_g[1]), float(anchor_local_g[2])))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(anchor_local_o[0]), float(anchor_local_o[1]), float(anchor_local_o[2])))
        
        rel_rot_mat = Ro.T @ Rg
        rel_quat = rotation_matrix_to_quaternion(rel_rot_mat)
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(float(rel_quat[0]), float(rel_quat[1]), float(rel_quat[2]), float(rel_quat[3])))

        try:
            joint.CreateBreakForceAttr().Set(2000.0)
            joint.CreateBreakTorqueAttr().Set(2000.0)
        except Exception:
            pass

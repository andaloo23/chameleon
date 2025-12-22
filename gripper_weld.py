"""gripper_weld.py

Intelligent, object-agnostic grasp logic for Isaac Sim.

Goal
----
Create a *physics* FixedJoint (a "weld") between the gripper and an object
only after we have evidence that the gripper is applying closing force and the
finger separation has stopped changing (stall) for ~1 second.

Key constraints from the user:
 - No hardcoding for a specific object size/shape.
 - Stall is detected via *finger separation* (distance between the two jaw link
   frames), not via cube size.
 - Creating the joint must *not* teleport/shift the object.

How we avoid teleportation:
 - We compute the joint local frames from the current world poses.
 - The joint is defined so that the current relative pose is preserved.

This class does NOT move the cube kinematically (no set_world_pose in the
holding loop). Once welded, physics will carry the object during lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-10:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
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
    q = np.array([x, y, z, w], dtype=np.float32)
    n = float(np.linalg.norm(q))
    return q if n < 1e-10 else (q / n)


@dataclass
class WeldDebug:
    closing: bool
    stall: bool
    finger_gap: Optional[float]
    gap_range: Optional[float]
    near_object: bool
    distance: Optional[float]


class IntelligentGripperWeld:
    """Detect a stalled close and weld an object to the gripper with a FixedJoint."""

    def __init__(
        self,
        env,
        *,
        dt: float = 1.0 / 120.0,
        stall_time_s: float = 1.0,
        stall_gap_range_m: float = 0.0015,
        near_distance_m: float = 0.09,
        close_command_margin: float = 1e-3,
        open_release_threshold: float = 0.55,
        debug: bool = True,
        joint_path: str = "/World/IntelligentGraspWeld",
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

    @property
    def is_grasping(self) -> bool:
        return self._is_welded

    def reset(self) -> None:
        self._gap_history.clear()
        self._stall_accum_s = 0.0
        self._is_welded = False
        self._step = 0
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
    ) -> WeldDebug:
        """Advance the stall detector and create/remove the weld joint."""
        self._step += 1

        # Release if user opens the gripper.
        if self._is_welded and gripper_value is not None and gripper_value > self.open_release_threshold:
            if self.debug:
                print("[WELD] Release: gripper opened")
            self.release()
            self._is_welded = False
            self._stall_accum_s = 0.0
            self._gap_history.clear()
            return WeldDebug(False, False, None, None, False, None)

        # Need jaw/gripper/world poses.
        if gripper_world_pos is None or jaw_world_pos is None or object_world_pos is None:
            if self.debug and self._step % 30 == 0:
                print("[WELD] Missing pose(s): "
                      f"gripper={gripper_world_pos is not None} jaw={jaw_world_pos is not None} obj={object_world_pos is not None}")
            self._stall_accum_s = 0.0
            self._gap_history.clear()
            return WeldDebug(False, False, None, None, False, None)

        # Finger gap proxy (distance between the two jaw link frames).
        gap = float(np.linalg.norm(np.asarray(gripper_world_pos) - np.asarray(jaw_world_pos)))

        # Are we commanding a close? (Target lower than current)
        closing = False
        if gripper_value is not None and target_gripper is not None:
            closing = bool(target_gripper < gripper_value - self.close_command_margin)

        # Near an object (generic): gripper midpoint is near object COM.
        distance = float(np.linalg.norm(0.5 * (np.asarray(gripper_world_pos) + np.asarray(jaw_world_pos)) - np.asarray(object_world_pos)))
        near_object = bool(distance <= self.near_distance_m)

        # Maintain a ~stall_time window of gap values.
        window_n = max(2, int(round(self.stall_time_s / max(self.dt, 1e-6))))
        self._gap_history.append(gap)
        if len(self._gap_history) > window_n:
            self._gap_history.pop(0)

        gap_range = None
        stall = False
        if len(self._gap_history) >= window_n:
            gmin = float(min(self._gap_history))
            gmax = float(max(self._gap_history))
            gap_range = gmax - gmin
            stall = bool(gap_range <= self.stall_gap_range_m)

        # Accumulate stall time only while closing and near object.
        if closing and near_object and stall and not self._is_welded:
            self._stall_accum_s += self.dt
        else:
            # decay quickly so we don't weld after intermittent stalls
            self._stall_accum_s = max(0.0, self._stall_accum_s - 2.0 * self.dt)

        if self.debug and self._step % 30 == 0:
            gr = f"{gap_range:.4f}" if gap_range is not None else "N/A"
            print(
                f"[WELD] step={self._step} close={closing} near={near_object} dist={distance:.3f} "
                f"gap={gap:.4f} range={gr} stall_t={self._stall_accum_s:.2f}s welded={self._is_welded}"
            )

        # Create weld when we have a full second of consistent stall.
        if (not self._is_welded) and (self._stall_accum_s >= self.stall_time_s):
            if gripper_world_orient is None or object_world_orient is None:
                # Need orientations to preserve pose without teleport.
                self._stall_accum_s = 0.0
            else:
                anchor_world = 0.5 * (np.asarray(gripper_world_pos) + np.asarray(jaw_world_pos))
                self._create_fixed_joint(
                    anchor_world=anchor_world,
                    gripper_world_pos=np.asarray(gripper_world_pos),
                    gripper_world_orient=np.asarray(gripper_world_orient),
                    object_world_pos=np.asarray(object_world_pos),
                    object_world_orient=np.asarray(object_world_orient),
                    object_prim_path=object_prim_path,
                    gripper_body_path=gripper_body_path,
                )
                self._is_welded = True
                if self.debug:
                    print("[WELD] ★★★ Weld created (stall confirmed)")

        return WeldDebug(closing, stall, gap, gap_range, near_object, distance)

    def release(self) -> None:
        """Delete the weld joint prim if it exists."""
        try:
            stage = self.env.world.stage
            prim = stage.GetPrimAtPath(self.joint_path)
            if prim and prim.IsValid():
                stage.RemovePrim(prim.GetPath())
        except Exception:
            pass

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
        """Create a FixedJoint preserving current pose (no teleport)."""
        stage = self.env.world.stage
        from pxr import UsdPhysics, Gf

        # Remove stale joint.
        try:
            prim = stage.GetPrimAtPath(self.joint_path)
            if prim and prim.IsValid():
                stage.RemovePrim(prim.GetPath())
        except Exception:
            pass

        Rg = quaternion_to_rotation_matrix(gripper_world_orient)
        Ro = quaternion_to_rotation_matrix(object_world_orient)

        anchor_local_g = Rg.T @ (anchor_world - gripper_world_pos)
        anchor_local_o = Ro.T @ (anchor_world - object_world_pos)

        # relative rotation from object->gripper
        rel_rot = Ro.T @ Rg
        rel_quat = rotation_matrix_to_quaternion(rel_rot)

        joint = UsdPhysics.FixedJoint.Define(stage, self.joint_path)
        joint.CreateBody0Rel().SetTargets([gripper_body_path])
        joint.CreateBody1Rel().SetTargets([object_prim_path])

        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(anchor_local_g[0]), float(anchor_local_g[1]), float(anchor_local_g[2])))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(anchor_local_o[0]), float(anchor_local_o[1]), float(anchor_local_o[2])))
        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(float(rel_quat[3]), float(rel_quat[0]), float(rel_quat[1]), float(rel_quat[2]))
        )

        # Optional safety: break forces prevent solver explosions.
        try:
            joint.CreateBreakForceAttr().Set(750.0)
            joint.CreateBreakTorqueAttr().Set(750.0)
        except Exception:
            pass

"""
Sticky (surface) gripper implemented as a PHYSICS JOINT (no kinematic teleporting).

Goal behavior:
- Detect a "real grasp" using a stall/pressure-like signal: gripper is commanded to close,
  but its actual joint position stops changing (can't close further).
- Only attach if the cube is near AND plausibly between the fingers (local-frame check).
- When grasp is confirmed, create a UsdPhysics.FixedJoint between the gripper link and the cube.
- When the gripper opens for a few frames, delete the joint to release.

IMPORTANT:
- This file does NOT set cube world pose each frame (no cube sliding/teleport).
- The joint anchor is chosen at the cube's CURRENT world position to avoid any initial jump.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# -------------------------
# Quaternion / rotation utils
# -------------------------

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 4:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = q[0], q[1], q[2], q[3]

    # Normalize
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = x / n, y / n, z / n, w / n

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return np.array([0, 0, 0, 1], dtype=np.float64)

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
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

    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0, 0, 0, 1], dtype=np.float64)
    return q / n


# -------------------------
# StickyGripper (joint-based)
# -------------------------

class StickyGripper:
    def __init__(
        self,
        env,
        # Spatial gate for attachment:
        grasp_threshold: float = 0.035,  # meters: must be close to cube to attach
        # Local-frame plausibility gate ("between fingers"):
        lateral_limit: float = 0.03,     # meters: cube must be within this laterally in gripper frame
        forward_min: float = -0.02,      # meters: allow slight behind origin due to frame placement
        forward_max: float = 0.08,       # meters: cube shouldn't be too far in front
        vertical_limit: float = 0.05,    # meters: cube shouldn't be too far above/below in gripper frame

        # Release behavior:
        gripper_open_threshold: float = 0.70,
        release_delay_frames: int = 5,

        # Stall detector behavior (how many consecutive frames of "grasped" to confirm):
        confirm_grasp_frames: int = 2,

        # Joint:
        joint_path: str = "/World/StickyGraspJoint",

        debug: bool = True,
    ):
        self.env = env

        self.grasp_threshold = float(grasp_threshold)
        self.lateral_limit = float(lateral_limit)
        self.forward_min = float(forward_min)
        self.forward_max = float(forward_max)
        self.vertical_limit = float(vertical_limit)

        self.gripper_open_threshold = float(gripper_open_threshold)
        self.release_delay_frames = int(release_delay_frames)
        self.confirm_grasp_frames = int(confirm_grasp_frames)

        self._fixed_joint_path = str(joint_path)
        self.debug = bool(debug)

        # State
        self._is_grasping = False
        self._open_frame_count = 0
        self._stall_grasp_count = 0
        self._update_count = 0

        # Cached prim paths
        self._gripper_prim_path: Optional[str] = None

        # Cache last good pose (in case pose queries fail sporadically)
        self._last_good_gripper_pos: Optional[np.ndarray] = None
        self._last_good_gripper_orient: Optional[np.ndarray] = None

        # Stall/pressure detector (your project already has grasp_detector.py)
        self._stall_detector = None

    # -------------------------
    # Public API
    # -------------------------

    def reset(self):
        self._is_grasping = False
        self._open_frame_count = 0
        self._stall_grasp_count = 0
        self._update_count = 0
        self._last_good_gripper_pos = None
        self._last_good_gripper_orient = None
        # Do NOT delete prim paths; those are stable across episodes
        # Always remove joint on reset
        self._remove_fixed_joint()

    @property
    def is_grasping(self) -> bool:
        return self._is_grasping

    def update(
        self,
        gripper_position: float,
        target_gripper_position: Optional[float] = None,
        gripper_world_pos: Optional[np.ndarray] = None,
        object_world_pos: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Call once per sim step.

        Expected inputs:
        - gripper_position: actual gripper joint value (float)
        - target_gripper_position: commanded gripper target (float). If None, stall detection still works,
          but it is much more reliable if you pass it in.
        - gripper_world_pos/object_world_pos: optional fallbacks (from your observation), used if USD pose fails.
        """
        self._update_count += 1

        # -------- release gate (open for a few frames) --------
        if gripper_position > self.gripper_open_threshold:
            self._open_frame_count += 1
        else:
            self._open_frame_count = 0

        if self._is_grasping:
            if self._open_frame_count >= self.release_delay_frames:
                self._release_grasp()
            return self._is_grasping

        # -------- get poses --------
        gripper_pos, gripper_orient, pose_quality = self._get_gripper_pose_with_quality(gripper_world_pos)
        cube_pos, cube_orient = self._get_cube_pose_with_fallback(object_world_pos)

        if gripper_pos is None or gripper_orient is None or cube_pos is None or cube_orient is None:
            return self._is_grasping

        # -------- spatial gate --------
        distance = float(np.linalg.norm(gripper_pos - cube_pos))
        near_enough = distance <= self.grasp_threshold

        # -------- local plausibility gate (between fingers) --------
        plausible = self._cube_between_fingers(gripper_pos, gripper_orient, cube_pos)

        # -------- stall/pressure detection --------
        if self._stall_detector is None:
            from grasp_detector import GraspDetector
            self._stall_detector = GraspDetector()

        grasp_state = self._stall_detector.update(
            gripper_position=float(gripper_position),
            target_position=float(target_gripper_position) if target_gripper_position is not None else None,
        )
        stall_grasp = bool(getattr(grasp_state, "grasped", False))

        # confirm over multiple frames to avoid transient detection
        if stall_grasp:
            self._stall_grasp_count += 1
        else:
            self._stall_grasp_count = 0

        confirmed_stall = self._stall_grasp_count >= self.confirm_grasp_frames

        # Require decent pose quality for joint creation (avoid "identity quaternion fallback" joints)
        pose_ok_for_joint = (pose_quality == "usd" or pose_quality == "cached")

        if self.debug and (self._update_count % 30 == 0 or confirmed_stall):
            tgt_s = f"{target_gripper_position:.3f}" if target_gripper_position is not None else "None"
            print(
                f"[StickyGripper] step={self._update_count} "
                f"dist={distance:.4f} near={near_enough} plausible={plausible} "
                f"stall={stall_grasp} stall_cnt={self._stall_grasp_count} pose={pose_quality} "
                f"g={gripper_position:.3f} tgt={tgt_s}"
            )

        # -------- attach --------
        if confirmed_stall and near_enough and plausible and pose_ok_for_joint:
            self._start_grasp(gripper_pos, gripper_orient, cube_pos, cube_orient)

        return self._is_grasping

    # -------------------------
    # Pose helpers
    # -------------------------

    def _get_gripper_pose_with_quality(
        self, gripper_world_pos_fallback: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Returns (pos, quat[x,y,z,w], quality) where quality in {"usd","cached","fallback","none"}.
        """
        pos, quat = self._get_gripper_world_pose_usd()
        if pos is not None and quat is not None:
            self._last_good_gripper_pos = pos.copy()
            self._last_good_gripper_orient = quat.copy()
            return pos, quat, "usd"

        if self._last_good_gripper_pos is not None and self._last_good_gripper_orient is not None:
            return self._last_good_gripper_pos.copy(), self._last_good_gripper_orient.copy(), "cached"

        if gripper_world_pos_fallback is not None:
            # NOTE: orientation fallback is unknown; this is NOT good enough to create a stable joint
            return np.asarray(gripper_world_pos_fallback, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32), "fallback"

        return None, None, "none"

    def _get_gripper_world_pose_usd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            stage = self.env.world.stage
            robot_prim_path = getattr(self.env.robot, "prim_path", None)
            if stage is None or robot_prim_path is None:
                return None, None

            from pxr import UsdGeom, Usd

            if self._gripper_prim_path is None:
                self._gripper_prim_path = self._find_gripper_prim_path(stage, robot_prim_path)

            if self._gripper_prim_path is None:
                return None, None

            prim = stage.GetPrimAtPath(self._gripper_prim_path)
            if prim is None or not prim.IsValid():
                self._gripper_prim_path = None
                return None, None

            xform = UsdGeom.Xformable(prim)
            M = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

            t = M.ExtractTranslation()
            pos = np.array([t[0], t[1], t[2]], dtype=np.float32)

            R = np.array(
                [
                    [M[0][0], M[0][1], M[0][2]],
                    [M[1][0], M[1][1], M[1][2]],
                    [M[2][0], M[2][1], M[2][2]],
                ],
                dtype=np.float64,
            )
            quat = rotation_matrix_to_quaternion(R).astype(np.float32)
            return pos, quat
        except Exception:
            return None, None

    def _get_cube_pose_with_fallback(
        self, object_world_pos_fallback: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            cube = self.env.cube
            pos, quat = cube.get_world_pose()
            return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)
        except Exception:
            if object_world_pos_fallback is not None:
                return np.asarray(object_world_pos_fallback, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)
            return None, None

    def _find_gripper_prim_path(self, stage, robot_prim_path: str) -> Optional[str]:
        """Try common paths first, then search descendants."""
        candidate_paths = [
            f"{robot_prim_path}/gripper",
            f"{robot_prim_path}/base/gripper",
            f"{robot_prim_path}/tool/gripper",
        ]
        for p in candidate_paths:
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                if self.debug:
                    print(f"[StickyGripper] Found gripper prim: {p}")
                return p

        # Descendant search
        try:
            root = stage.GetPrimAtPath(robot_prim_path)
            if not root or not root.IsValid():
                return None

            def dfs(prim):
                name = prim.GetName().lower()
                path = str(prim.GetPath())
                if "gripper" in name and "jaw" not in name and "finger" not in name:
                    return path
                for c in prim.GetChildren():
                    got = dfs(c)
                    if got is not None:
                        return got
                return None

            found = dfs(root)
            if found and self.debug:
                print(f"[StickyGripper] Found gripper prim by search: {found}")
            return found
        except Exception:
            return None

    # -------------------------
    # Grasp plausibility
    # -------------------------

    def _cube_between_fingers(self, gripper_pos: np.ndarray, gripper_orient: np.ndarray, cube_pos: np.ndarray) -> bool:
        """
        Lightweight "between fingers" check in the gripper's local frame.

        We don't hardcode which axis is "forward" (robot-dependent), so we:
        - Transform cube into gripper frame
        - Require cube be within bounds along ALL axes, with a slightly larger window in "forward".
        """
        Rg = quaternion_to_rotation_matrix(gripper_orient)
        local = Rg.T @ (cube_pos - gripper_pos)

        lx, ly, lz = float(local[0]), float(local[1]), float(local[2])

        # Choose the axis that looks most "forward" this moment (largest magnitude)
        # and apply forward bounds to that axis; apply lateral/vertical bounds to others.
        abs_local = np.abs(local)
        forward_axis = int(np.argmax(abs_local))

        axes = [0, 1, 2]
        other_axes = [a for a in axes if a != forward_axis]

        # Forward bounds
        f = float(local[forward_axis])
        if not (self.forward_min <= f <= self.forward_max):
            return False

        # Lateral/vertical-ish bounds on the other axes
        for a in other_axes:
            v = float(local[a])
            lim = self.lateral_limit if a != 2 else self.vertical_limit
            if abs(v) > lim:
                return False

        return True

    # -------------------------
    # Joint creation/removal
    # -------------------------

    def _start_grasp(
        self,
        gripper_pos: np.ndarray,
        gripper_orient: np.ndarray,
        cube_pos: np.ndarray,
        cube_orient: np.ndarray,
    ):
        """Create fixed joint. Anchor at cube world position to avoid any initial slide."""
        try:
            self._create_fixed_joint(
                gripper_world_pos=gripper_pos,
                gripper_world_orient=gripper_orient,
                cube_world_pos=cube_pos,
                cube_world_orient=cube_orient,
            )
            self._is_grasping = True
            self._open_frame_count = 0
            if self.debug:
                print("[StickyGripper] ★★★ GRASP CONFIRMED → FixedJoint created")
        except Exception as e:
            self._is_grasping = False
            if self.debug:
                print(f"[StickyGripper] Failed to create FixedJoint: {e}")

    def _release_grasp(self):
        self._remove_fixed_joint()
        self._is_grasping = False
        self._stall_grasp_count = 0
        if self.debug:
            print("[StickyGripper] ★★★ RELEASE → FixedJoint removed")

    def _remove_fixed_joint(self):
        try:
            stage = self.env.world.stage
            prim = stage.GetPrimAtPath(self._fixed_joint_path)
            if prim and prim.IsValid():
                stage.RemovePrim(prim.GetPath())
        except Exception:
            pass

    def _create_fixed_joint(
        self,
        gripper_world_pos: np.ndarray,
        gripper_world_orient: np.ndarray,
        cube_world_pos: np.ndarray,
        cube_world_orient: np.ndarray,
    ) -> None:
        """Create a UsdPhysics.FixedJoint that welds cube to gripper link, preserving current relative pose."""
        stage = self.env.world.stage
        if stage is None:
            raise RuntimeError("USD stage is None")

        from pxr import UsdPhysics, Gf

        robot_prim_path = getattr(self.env.robot, "prim_path", "/World/Robot")

        if self._gripper_prim_path is None:
            self._gripper_prim_path = self._find_gripper_prim_path(stage, robot_prim_path)
        if self._gripper_prim_path is None:
            raise RuntimeError("Could not find gripper prim path for FixedJoint body0.")

        gripper_path = self._gripper_prim_path
        cube_path = getattr(self.env.cube, "prim_path", "/World/Cube")

        # Anchor at the cube's current position so the joint introduces no positional correction.
        anchor_world = np.asarray(cube_world_pos, dtype=np.float64)

        # Remove any existing joint
        try:
            existing = stage.GetPrimAtPath(self._fixed_joint_path)
            if existing and existing.IsValid():
                stage.RemovePrim(existing.GetPath())
        except Exception:
            pass

        # Compute local frames
        Rg = quaternion_to_rotation_matrix(np.asarray(gripper_world_orient, dtype=np.float64))
        Rc = quaternion_to_rotation_matrix(np.asarray(cube_world_orient, dtype=np.float64))

        anchor_local_gripper = Rg.T @ (anchor_world - np.asarray(gripper_world_pos, dtype=np.float64))
        anchor_local_cube = Rc.T @ (anchor_world - np.asarray(cube_world_pos, dtype=np.float64))

        # Relative rotation from cube to gripper: q_rel makes cube frame rotate into gripper frame
        rel_rot = Rc.T @ Rg
        rel_quat = rotation_matrix_to_quaternion(rel_rot)  # [x,y,z,w]

        joint = UsdPhysics.FixedJoint.Define(stage, self._fixed_joint_path)
        joint.CreateBody0Rel().SetTargets([gripper_path])
        joint.CreateBody1Rel().SetTargets([cube_path])

        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(anchor_local_gripper[0]), float(anchor_local_gripper[1]), float(anchor_local_gripper[2])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(anchor_local_cube[0]), float(anchor_local_cube[1]), float(anchor_local_cube[2])))

        # Rotations are quats in (w, x, y, z) for Gf.Quatf
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(float(rel_quat[3]), float(rel_quat[0]), float(rel_quat[1]), float(rel_quat[2])))

        # Optional: break forces for stability (not hardcoding pose; just prevents explosive constraints)
        try:
            joint.CreateBreakForceAttr().Set(750.0)
            joint.CreateBreakTorqueAttr().Set(750.0)
        except Exception:
            pass

        # Optional: damp cube velocities right away (prevents initial jitter)
        try:
            self.env.cube.set_linear_velocity(np.zeros(3))
            self.env.cube.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

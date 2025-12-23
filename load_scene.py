import os
import shutil
import time
from typing import Optional, Tuple

import numpy as np
from isaacsim import SimulationApp

from cup_utils import create_cup_prim, initialize_usd_modules
from domain_randomizer import DomainRandomizer
from image_utils import write_png
from reward_engine import RewardEngine
from gripper_weld import IntelligentGripperWeld, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from workspace import CUP_CUBE_MIN_DISTANCE, WORKSPACE_RADIUS_RANGE, sample_workspace_xy

_SIMULATION_APP = None
_SIM_HEADLESS_FLAG = None

World = None
DynamicCuboid = None
get_context = None
Camera = None
Gf = None
UsdGeom = None
UsdPhysics = None
Usd = None
SO100Robot = None


def _ensure_isaac_sim(headless=False):
    """Create the SimulationApp and late-import Isaac APIs on first use."""
    global _SIMULATION_APP, _SIM_HEADLESS_FLAG
    global World, DynamicCuboid, get_context, Camera, Gf, UsdGeom, UsdPhysics, SO100Robot

    if _SIMULATION_APP is None:
        _SIMULATION_APP = SimulationApp({
            "headless": headless,
            "load_stage_on_start": False,
        })

        from omni.isaac.core import World as _World
        from omni.isaac.core.objects import DynamicCuboid as _DynamicCuboid
        from omni.usd import get_context as _get_context
        from omni.isaac.sensor import Camera as _Camera
        from pxr import Gf as _Gf, UsdGeom as _UsdGeom, UsdPhysics as _UsdPhysics, Usd as _Usd
        from robot import SO100Robot as _SO100Robot

        World = _World
        DynamicCuboid = _DynamicCuboid
        get_context = _get_context
        Camera = _Camera
        Gf = _Gf
        UsdGeom = _UsdGeom
        UsdPhysics = _UsdPhysics
        Usd = _Usd
        SO100Robot = _SO100Robot
        initialize_usd_modules(Gf, UsdGeom, UsdPhysics)
        _SIM_HEADLESS_FLAG = headless
    elif headless != _SIM_HEADLESS_FLAG:
        print(f"[warn] SimulationApp already initialized with headless={_SIM_HEADLESS_FLAG}")

    return _SIMULATION_APP


class IsaacPickPlaceEnv:
    """Isaac Sim environment wrapper for the pick-and-place task."""

    def __init__(self, headless=False, capture_images=False, image_interval=3, random_seed=None, grasp_mode: str = "weld"):
        self.headless = headless
        self.capture_images = capture_images
        self.image_interval = max(1, int(image_interval))
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Unified Intelligent Weld Gripper (Standardized)
        self.grasp_mode = "weld"
        self.use_sticky_gripper = False
        self.use_physics_gripper = False
        self.use_weld_gripper = True

        # Physics parameters for reliable grasping
        self.cube_mass = 0.08
        self._cube_friction = 2.2
        self._gripper_friction = 2.2
        self._gripper_drive_stiffness = 6000.0
        self._gripper_drive_damping = 400.0
        self._gripper_drive_max_force = 100.0
        self._contact_offset = 0.001
        self._rest_offset = 0.0001

        _ensure_isaac_sim(headless=headless)

        self.simulation_app = _SIMULATION_APP
        self.stage_context = get_context()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = os.path.join(self.current_dir, "temp")

        self.world = None
        self.robot = None
        self.robot_articulation = None
        self.top_camera = None
        self.side_camera = None
        self.cup_xform = None
        self.cube = None

        self.cube_scale = np.array([0.075, 0.075, 0.075], dtype=float)
        self.cup_height = 0.18
        self.cup_outer_radius_top = 0.11
        self.cup_outer_radius_bottom = 0.085
        self.cup_wall_thickness = 0.012
        self.cup_inner_radius_top = max(self.cup_outer_radius_top - self.cup_wall_thickness,
                                        self.cup_outer_radius_top * 0.55)
        self.cup_inner_radius_bottom = max(self.cup_outer_radius_bottom - self.cup_wall_thickness,
                                           self.cup_outer_radius_bottom * 0.55)
        self.cup_bottom_thickness = max(0.008, self.cup_wall_thickness * 0.75)
        self.cup_color = (0.8, 0.3, 0.2)
        self.cup_mass = 0.25

        self._step_counter = 0

        self._camera_keys = ("top", "side", "wrist")
        self._last_camera_frames = {key: None for key in self._camera_keys}
        self._camera_failure_logged = {key: False for key in self._camera_keys}
        self._camera_frame_shapes = {}
        self._fixed_camera_poses = {}
        self._last_gripper_pose = (None, None)
        self._last_jaw_pos = None
        self._gripper_prim_path = None
        self._jaw_prim_path = None

        self.reward_engine = RewardEngine(self)
        self.domain_randomizer = DomainRandomizer(self)
        self.last_validation_result = {"ok": True, "issues": [], "flags": {}}
        self._force_terminate = False
        self._termination_reason = None
        self._cup_upright_threshold_rad = np.deg2rad(25.0)
        
        # Intelligent physics weld grasping
        self.gripper_weld = IntelligentGripperWeld(
            env=self,
            dt=1.0 / 120.0,
            stall_time_s=0.25,
            stall_gap_range_m=1.5e-3,
            near_distance_m=0.30,
            close_command_margin=1e-3,
            open_release_threshold=0.65,
            debug=True,
            joint_path="/World/GripperWeldJoint",
        )
        self._gripper_pose_fallback_warned = False
        self._prev_gripper_value = None

        if self.capture_images:
            self._reset_temp_dir()

        self._build_scene()
        self.reward_engine.initialize()
        self.reward_engine.reset()

    def _reset_temp_dir(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

    def _build_scene(self):
        self.stage_context.new_stage()

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        try:
            physics_context = self.world.get_physics_context()
            physics_context.set_solver_type("TGS")
            
            stage = self.world.stage
            scene_prim = None
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    scene_prim = prim
                    break
            
            if scene_prim:
                from pxr import PhysxSchema
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
                physx_scene.CreateTimeStepsPerSecondAttr().Set(120.0)

            try:
                physics_context.enable_gpu_dynamics(True)
            except Exception:
                pass
            
            try:
                physics_context.set_physics_dt(1.0 / 120.0)
            except Exception:
                pass
            
            try:
                physics_context.enable_ccd(True)
            except Exception:
                pass
                
        except Exception as e:
            print(f"[warn] Could not configure physics solver: {e}")

        urdf_path = os.path.join(self.current_dir, "so100.urdf")
        self.robot = SO100Robot(self.world, urdf_path)
        self.robot_articulation = self.robot.get_robot()
        self._base_fixture_pose = None
        self._workspace_origin_xy = None
        self._default_joint_positions = self._compute_default_joint_positions()

        cube_xy, cup_xy = self._sample_object_positions()
        cube_position = np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0])

        self.cube = DynamicCuboid(
            prim_path="/World/Cube",
            name="collision_cube",
            position=cube_position,
            scale=self.cube_scale,
            color=np.array([0.0, 0.5, 1.0]),
            mass=float(self.cube_mass),
        )
        self.world.scene.add(self.cube)
        
        try:
            cube_prim = self.world.stage.GetPrimAtPath("/World/Cube")
            if cube_prim and cube_prim.IsValid():
                if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(cube_prim)
                
                material_api = UsdPhysics.MaterialAPI.Apply(cube_prim)
                material_api.CreateStaticFrictionAttr().Set(float(self._cube_friction))
                material_api.CreateDynamicFrictionAttr().Set(float(self._cube_friction))
                material_api.CreateRestitutionAttr().Set(0.0)
        except Exception as e:
            print(f"[warn] Could not apply collision/friction to cube: {e}")

        cup_translation = np.array([cup_xy[0], cup_xy[1], 0.0])
        self.cup_xform = create_cup_prim(
            self.stage_context.get_stage(),
            prim_path="/World/Cup",
            position=cup_translation,
            outer_radius_top=self.cup_outer_radius_top,
            outer_radius_bottom=self.cup_outer_radius_bottom,
            inner_radius_top=self.cup_inner_radius_top,
            inner_radius_bottom=self.cup_inner_radius_bottom,
            height=self.cup_height,
            bottom_thickness=self.cup_bottom_thickness,
            color=self.cup_color,
            mass=self.cup_mass,
        )

        self.top_camera = Camera(
            prim_path="/World/top_camera",
            name="top_camera",
            position=np.array([0, 0, 1.5]),
            orientation=np.array([0, 0, 0, 1]),
            frequency=30,
            resolution=(128, 128),
        )
        self.side_camera = Camera(
            prim_path="/World/side_camera",
            name="side_camera",
            position=np.array([1.0, 0, 0.5]),
            orientation=np.array([0, 0, 0.707, 0.707]),
            frequency=30,
            resolution=(128, 128),
        )

        self.world.scene.add(self.top_camera)
        self.world.scene.add(self.side_camera)

        self.world.reset()
        self._apply_default_joint_positions()
        self._capture_base_fixture_pose()
        self._resolve_link_paths()
        top_cam_pos = np.array([0, -0.75, 8.0])
        top_cam_orient = np.array([-np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25), np.sqrt(0.25)])
        side_cam_pos = np.array([6.0, -0.5, 0.5])
        side_cam_orient = np.array([0, 0, 0, 1])

        self.top_camera.set_world_pose(position=top_cam_pos, orientation=top_cam_orient)
        self.side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)
        
        self._fixed_camera_poses = {
            "top": (top_cam_pos.copy(), top_cam_orient.copy()),
            "side": (side_cam_pos.copy(), side_cam_orient.copy()),
        }

        self.robot.create_wrist_camera()
        self.robot.update_wrist_camera_position(verbose=False)

        self.top_camera.initialize()
        self.side_camera.initialize()
        if getattr(self.robot, "wrist_camera", None) is not None:
            self.robot.wrist_camera.initialize()

        self._apply_gripper_friction()
        self._configure_gripper_drive()

        for _ in range(5):
            self.world.step(render=not self.headless)

        self._cube_xy = cube_xy
        self._cup_xy = cup_xy
        self._apply_domain_randomization()

    def reset(self, render=None):
        self._step_counter = 0
        render = not self.headless if render is None else bool(render)
        if self.gripper_weld is not None:
            self.gripper_weld.release()

        cube_xy, cup_xy = self._sample_object_positions()
        self._cube_xy = cube_xy
        self._cup_xy = cup_xy

        cube_position = np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0])
        self.cube.set_world_pose(position=cube_position, orientation=np.array([0, 0, 0, 1]))

        cup_translation = Gf.Vec3d(float(cup_xy[0]), float(cup_xy[1]), 0.0)
        UsdGeom.XformCommonAPI(self.cup_xform).SetTranslate(cup_translation)

        self.world.reset()
        self._apply_default_joint_positions()
        self._restore_base_fixture_pose()
        self._restore_fixed_camera_poses()
        self.robot.update_wrist_camera_position(verbose=False)
        self._resolve_link_paths()

        for i in range(5):
            do_render = (render if i == 4 else False)
            self.world.step(render=do_render)

        self._apply_domain_randomization()
        self.reward_engine.reset()
        if self.gripper_weld is not None:
            self.gripper_weld.reset()
        self.last_validation_result = {"ok": True, "issues": [], "flags": {}}
        self._force_terminate = False
        self._termination_reason = None
        self._latest_target_gripper = None
        self._prev_gripper_value = None
        self._last_gripper_pose = (None, None)
        self._last_jaw_pos = None
        
        return self._get_observation()

    def step(self, action, render=None):
        render = not self.headless if render is None else bool(render)
        joint_targets = self._clip_action(action)
        self.robot.set_joint_positions(joint_targets)
        self._latest_target_gripper = float(joint_targets[-1]) if len(joint_targets) > 0 else None

        self._restore_base_fixture_pose()
        self.world.step(render=render)
        self._step_counter += 1
        
        try:
            actual_joint_positions = self.robot_articulation.get_joint_positions()
            actual_gripper_pos = float(actual_joint_positions[-1]) if len(actual_joint_positions) > 0 else 0.0
        except Exception:
            actual_gripper_pos = float(joint_targets[-1]) if len(joint_targets) > 0 else 0.0
        
        cube_world_pos = None
        cube_world_orient = None
        try:
            cube_world_pos, cube_world_orient = self.cube.get_world_pose()
            cube_world_pos = np.array(cube_world_pos, dtype=np.float32)
            cube_world_orient = np.array(cube_world_orient, dtype=np.float32)
        except Exception:
            pass
        
        gripper_world_pos = None
        gripper_world_orient = None
        jaw_world_pos = None
        try:
            if Usd is not None:
                if self._gripper_prim_path is None or self._jaw_prim_path is None:
                    self._resolve_link_paths()

                stage = self.world.stage
                gripper_path = self._gripper_prim_path
                jaw_path = self._jaw_prim_path

                if gripper_path:
                    gripper_prim = stage.GetPrimAtPath(gripper_path)
                    if not (gripper_prim and gripper_prim.IsValid()):
                        self._resolve_link_paths()
                        gripper_path = self._gripper_prim_path
                        gripper_prim = stage.GetPrimAtPath(gripper_path) if gripper_path else None

                    if gripper_prim and gripper_prim.IsValid():
                        xformable = UsdGeom.Xformable(gripper_prim)
                        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        translation = matrix.ExtractTranslation()
                        gripper_world_pos = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
                        rot_matrix = np.array([
                            [matrix[0][0], matrix[0][1], matrix[0][2]],
                            [matrix[1][0], matrix[1][1], matrix[1][2]],
                            [matrix[2][0], matrix[2][1], matrix[2][2]]
                        ], dtype=np.float32)
                        # USD matrices are row-major; transpose to get R such that v_world = R * v_local
                        gripper_world_orient = rotation_matrix_to_quaternion(rot_matrix.T)

                if jaw_path:
                    jaw_prim = stage.GetPrimAtPath(jaw_path)
                    if not (jaw_prim and jaw_prim.IsValid()):
                        jaw_path = self._jaw_prim_path
                        jaw_prim = stage.GetPrimAtPath(jaw_path) if jaw_path else None

                    if jaw_prim and jaw_prim.IsValid():
                        jaw_xformable = UsdGeom.Xformable(jaw_prim)
                        jaw_matrix = jaw_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        jaw_translation = jaw_matrix.ExtractTranslation()
                        jaw_world_pos = np.array([jaw_translation[0], jaw_translation[1], jaw_translation[2]], dtype=np.float32)

            self._last_gripper_pose = (gripper_world_pos, gripper_world_orient)
            self._last_jaw_pos = jaw_world_pos
        except Exception:
            pass
        
        if gripper_world_pos is None:
            try:
                wrist_cam = getattr(self.robot, "wrist_camera", None)
                if wrist_cam is not None:
                    pose = self._compute_gripper_pose_from_camera()
                    if pose is not None:
                        gripper_world_pos, gripper_world_orient = pose
                        self._last_gripper_pose = (gripper_world_pos, gripper_world_orient)
            except Exception:
                pass
        
        if jaw_world_pos is None and gripper_world_pos is not None:
            if self._step_counter % 100 == 0:
                print(f"[ENV] Approximating jaw with gripper (jaw_path={self._jaw_prim_path})")
            jaw_world_pos = gripper_world_pos.copy()
        
        if gripper_world_pos is not None and jaw_world_pos is not None:
            gap = np.linalg.norm(gripper_world_pos - jaw_world_pos)
            if gap < 1e-5 and self._step_counter % 100 == 0:
                print(f"[ENV] WARNING: Zero gap between links. Gripper={gripper_world_pos} Jaw={jaw_world_pos}")
                print(f"      Check if {self._gripper_prim_path} and {self._jaw_prim_path} are the same body.")
        
        if self.gripper_weld is not None:
            was_grasping = self.gripper_weld.is_grasping
            gripper_path = self._gripper_prim_path or f"{getattr(self.robot, 'prim_path', '/World/Robot')}/gripper"
            jaw_path = self._jaw_prim_path or f"{getattr(self.robot, 'prim_path', '/World/Robot')}/jaw"
            cube_prim_path = getattr(self.cube, "prim_path", "/World/Cube")
            
            self.gripper_weld.update(
                gripper_value=actual_gripper_pos,
                target_gripper=self._latest_target_gripper,
                gripper_world_pos=gripper_world_pos,
                gripper_world_orient=gripper_world_orient,
                jaw_world_pos=jaw_world_pos,
                object_world_pos=cube_world_pos,
                object_world_orient=cube_world_orient,
                object_prim_path=cube_prim_path,
                gripper_body_path=gripper_path,
                jaw_body_path=jaw_path,
            )
            now_grasping = self.gripper_weld.is_grasping
            if was_grasping != now_grasping:
                print(f"[ENV] GRASP WELD STATE: {was_grasping} -> {now_grasping}")

        obs = self._get_observation()
        self.reward_engine.compute_reward_components()
        self._validate_state(obs)
        self._prev_gripper_value = actual_gripper_pos
        reward, done, info = self.reward_engine.summarize_reward()

        if self._force_terminate:
            done = True
            info.setdefault("termination_reason", self._termination_reason or "validation_failure")
            info["terminated_by_validation"] = True

        if self.capture_images and (self._step_counter % self.image_interval == 0):
            self._capture_images()

        return obs, reward, done, info

    def close(self):
        if self.gripper_weld is not None:
            self.gripper_weld.release()
        if self.world is not None:
            self.world.clear()
        if self.capture_images:
            print(f"[info] saved images to {self.temp_dir}")

    def shutdown(self):
        self.close()
        if self.simulation_app is not None:
            self.simulation_app.close()

    def _sample_object_positions(self):
        cube_xy = np.array([0.0, -0.50])
        cup_xy = np.array([0.0, -0.75])
        return cube_xy, cup_xy

    def _clip_action(self, action):
        action = np.array(action, dtype=float)
        lower = np.array([self.robot.joint_limits[name][0] for name in self.robot.joint_names], dtype=float)
        upper = np.array([self.robot.joint_limits[name][1] for name in self.robot.joint_names], dtype=float)
        if action.shape != lower.shape:
            raise ValueError(f"Expected action of shape {lower.shape}, received {action.shape}")
        span = np.maximum(upper - lower, 1e-6)
        margin = np.clip(0.05 * span, 1e-3, 0.1)
        clipped = np.clip(action, lower + margin, upper - margin)
        return clipped

    def _get_observation(self):
        camera_frames = self._gather_sensor_observations()
        joint_positions = self.robot_articulation.get_joint_positions()
        joint_velocities = self.robot_articulation.get_joint_velocities()
        target_gripper = getattr(self, '_latest_target_gripper', None)
        self.reward_engine.record_joint_state(joint_positions.astype(np.float32), joint_velocities.astype(np.float32), target_gripper)

        cube_pos = None
        gripper_pos = None
        gripper_orient = None
        try:
            cube_pos, _ = self.cube.get_world_pose()
            cube_pos = np.array(cube_pos, dtype=np.float32)
        except Exception:
            pass
        
        if getattr(self, "_last_gripper_pose", None):
            gp, go = self._last_gripper_pose
            if gp is not None:
                gripper_pos = np.array(gp, dtype=np.float32)
                gripper_orient = go
        if gripper_pos is None:
            try:
                wrist_cam = getattr(self.robot, "wrist_camera", None)
                if wrist_cam is not None:
                    gripper_pos, _ = wrist_cam.get_world_pose()
                    gripper_pos = np.array(gripper_pos, dtype=np.float32)
            except Exception:
                pass

        obs = {
            "joint_positions": joint_positions.astype(np.float32),
            "joint_velocities": joint_velocities.astype(np.float32),
            "cube_pos": cube_pos,
            "gripper_pos": gripper_pos,
            "gripper_orient": gripper_orient,
        }
        for name, frame in camera_frames.items():
            obs[name] = frame.copy() if frame is not None else None
        return obs

    def _gather_sensor_observations(self):
        camera_map = {"top": self.top_camera, "side": self.side_camera, "wrist": getattr(self.robot, "wrist_camera", None)}
        frames = {}
        for key, camera in camera_map.items():
            processed = None
            if camera is not None:
                try:
                    raw_frame = camera.get_rgba()
                    processed = self._prepare_camera_frame(raw_frame, key)
                    if processed is not None:
                        self._last_camera_frames[key] = processed
                        self._camera_failure_logged[key] = False
                except Exception:
                    if not self._camera_failure_logged.get(key, False):
                        self._camera_failure_logged[key] = True
            
            if processed is None:
                cached = self._last_camera_frames.get(key)
                processed = cached if cached is not None else self._allocate_empty_camera_frame(camera, key)
                self._last_camera_frames[key] = processed
            frames[f"{key}_camera_rgb"] = processed
        return frames

    def _prepare_camera_frame(self, frame, key):
        if frame is None:
            return None
        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3:
            channels = array.shape[-1]
            if channels >= 3:
                array = array[..., :3]
            elif channels == 1:
                array = np.repeat(array, 3, axis=-1)
            else:
                pad = 3 - channels
                array = np.concatenate([array, np.zeros(array.shape[:2] + (pad,), dtype=array.dtype)], axis=-1)
        else:
            return None
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        else:
            array = np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)
        self._camera_frame_shapes[key] = array.shape
        return np.ascontiguousarray(array)

    def _allocate_empty_camera_frame(self, camera, key):
        if key in self._camera_frame_shapes:
            h, w, _ = self._camera_frame_shapes[key]
        else:
            w, h = 128, 128
            if camera:
                res = getattr(camera, "get_resolution", lambda: getattr(camera, "resolution", (128, 128)))()
                w, h = res[0], res[1]
            self._camera_frame_shapes[key] = (int(h), int(w), 3)
        h, w, _ = self._camera_frame_shapes[key]
        return np.zeros((h, w, 3), dtype=np.float32)

    def _compute_gripper_pose_from_camera(self):
        wrist_cam = getattr(self.robot, "wrist_camera", None)
        if wrist_cam is None:
            return None
        try:
            p_w, o_w = wrist_cam.get_world_pose()
            p_l, o_l = wrist_cam.get_local_pose()
            R_w = quaternion_to_rotation_matrix(np.asarray(o_w, dtype=float))
            R_l = quaternion_to_rotation_matrix(np.asarray(o_l, dtype=float))
            R_g = R_w @ R_l.T
            return np.asarray(p_w, dtype=float) - R_g @ np.asarray(p_l, dtype=float), rotation_matrix_to_quaternion(R_g)
        except Exception:
            return None

    def _compute_default_joint_positions(self):
        defaults = {"shoulder_pan": 0.0, "shoulder_lift": -1.5, "elbow_flex": 1.1, "wrist_flex": 0.3, "wrist_roll": 0.0, "gripper": 0.03}
        limits = self.robot.joint_limits
        return np.asarray([float(np.clip(defaults.get(n, 0.0), *limits[n])) for n in self.robot.joint_names], dtype=np.float32)

    def _apply_default_joint_positions(self):
        if self.robot and self._default_joint_positions is not None:
            try:
                self.robot.set_joint_positions(self._default_joint_positions)
            except Exception:
                pass

    def _capture_base_fixture_pose(self):
        if self.robot_articulation:
            try:
                p, o = self.robot_articulation.get_world_pose()
                self._base_fixture_pose = (np.asarray(p, dtype=float), np.asarray(o, dtype=float))
                self._workspace_origin_xy = self._base_fixture_pose[0][:2].copy()
            except Exception:
                pass

    def _restore_base_fixture_pose(self):
        if self.robot_articulation and self._base_fixture_pose:
            try:
                self.robot_articulation.set_world_pose(position=self._base_fixture_pose[0], orientation=self._base_fixture_pose[1])
            except Exception:
                pass
    
    def _restore_fixed_camera_poses(self):
        if not self._fixed_camera_poses:
            return
        try:
            for k, cam in [("top", self.top_camera), ("side", self.side_camera)]:
                if k in self._fixed_camera_poses and cam:
                    p, o = self._fixed_camera_poses[k]
                    cam.set_world_pose(position=p, orientation=o)
        except Exception:
            pass

    def _resolve_link_paths(self):
        stage = getattr(self.world, "stage", None)
        root = getattr(self.robot, "prim_path", None)
        if not stage or not root:
            return

        def _find(name: str):
            direct = f"{root}/{name}"
            if stage.GetPrimAtPath(direct).IsValid():
                return direct
            alts = [f"{root}/base_link/{name}", f"{root}/{name}_link", f"{root}/base/{name}",
                    f"{root}/gripper/{name}" if name == "jaw" else None,
                    f"{root}/fixed_jaw" if name == "gripper" else None,
                    f"{root}/moving_jaw" if name == "jaw" else None]
            for alt in alts:
                if alt and stage.GetPrimAtPath(alt).IsValid():
                    return alt
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(root):
                    continue
                leaf = path.split("/")[-1].lower()
                if leaf == name.lower() or leaf.startswith(f"{name.lower()}_") or leaf.endswith(f"_{name.lower()}") or \
                   (name == "jaw" and "moving" in leaf) or (name == "gripper" and "fixed" in leaf):
                    if prim.IsA(UsdGeom.Xformable):
                        return path
            return None

        self._gripper_prim_path = _find("gripper") or self._gripper_prim_path
        self._jaw_prim_path = _find("jaw") or self._jaw_prim_path
        
        if not self._gripper_prim_path or not self._jaw_prim_path:
            try:
                for prim in stage.Traverse():
                    if not str(prim.GetPath()).startswith(root):
                        continue
                    if prim.IsA(UsdPhysics.Joint):
                        joint = UsdPhysics.Joint(prim)
                        rel0 = getattr(joint, "GetChild0Rel", lambda: None)()
                        targets = rel0.GetTargets() if rel0 else []
                        for t in targets:
                            tl = str(t).split("/")[-1].lower()
                            if "gripper" in tl and not self._gripper_prim_path:
                                self._gripper_prim_path = str(t)
                            if ("jaw" in tl or "finger" in tl) and not self._jaw_prim_path:
                                self._jaw_prim_path = str(t)
            except Exception:
                pass

        if self._gripper_prim_path and not self._jaw_prim_path:
            g_prim = stage.GetPrimAtPath(self._gripper_prim_path)
            for c in g_prim.GetChildren():
                if c.IsA(UsdGeom.Xformable):
                    self._jaw_prim_path = str(c.GetPath())
                    break

        if self._step_counter % 200 == 0:
            if self._jaw_prim_path:
                print(f"[ENV] Links: gripper={self._gripper_prim_path} jaw={self._jaw_prim_path}")
            else:
                print(f"[ENV] WARNING: Link resolution failed under {root}")

    def _apply_gripper_friction(self):
        try:
            from pxr import PhysxSchema
            stage = self.world.stage
            root = getattr(self.robot, "prim_path", "/World/Robot")
            for p in [f"{root}/gripper", f"{root}/jaw"]:
                prim = stage.GetPrimAtPath(p)
                if prim.IsValid():
                    mat = UsdPhysics.MaterialAPI.Apply(prim)
                    mat.CreateStaticFrictionAttr().Set(float(self._gripper_friction))
                    mat.CreateDynamicFrictionAttr().Set(float(self._gripper_friction))
                    mat.CreateRestitutionAttr().Set(0.0)
                    ps = PhysxSchema.PhysxMaterialAPI.Apply(prim)
                    ps.CreateFrictionCombineModeAttr().Set("max")
                    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    rb.CreateContactOffsetAttr().Set(float(self._contact_offset))
                    rb.CreateRestOffsetAttr().Set(float(self._rest_offset))
        except Exception:
            pass

    def _configure_gripper_drive(self):
        try:
            from pxr import PhysxSchema
            stage = self.world.stage
            root = getattr(self.robot, "prim_path", "/World/Robot")
            gj = None
            for p in [f"{root}/gripper/gripper", f"{root}/gripper_joint", f"{root}/joints/gripper"]:
                if stage.GetPrimAtPath(p).IsValid():
                    gj = stage.GetPrimAtPath(p)
                    break
            if not gj:
                for prim in stage.Traverse():
                    if "gripper" in str(prim.GetPath()).lower() and prim.IsA(UsdPhysics.RevoluteJoint):
                        gj = prim
                        break
            if gj:
                if not gj.HasAPI(PhysxSchema.PhysxJointAPI):
                    PhysxSchema.PhysxJointAPI.Apply(gj)
                dr = UsdPhysics.DriveAPI.Apply(gj, "angular")
                dr.CreateStiffnessAttr().Set(float(self._gripper_drive_stiffness))
                dr.CreateDampingAttr().Set(float(self._gripper_drive_damping))
                dr.CreateMaxForceAttr().Set(float(self._gripper_drive_max_force))
        except Exception:
            pass

    def _apply_domain_randomization(self):
        if self.domain_randomizer:
            self.domain_randomizer.randomize()

    def _get_recent_collisions(self):
        try:
            physics = self.world.get_physics_context()
            getter = getattr(physics, "get_contact_report", lambda: None)
            report = getter()
            if not report:
                return []
            entries = report.get("contacts", []) if isinstance(report, dict) else report
            return [{"prim0": e.get("body0"), "prim1": e.get("body1")} for e in entries if isinstance(e, dict)]
        except Exception:
            return []

    def _validate_state(self, _obs):
        issues = []
        pos = self.reward_engine.latest_joint_positions
        if pos is None:
            issues.append("No joint pos")
        elif not np.isfinite(pos).all():
            issues.append("NaN joints")
        cube_pos = getattr(self.reward_engine, "task_state", {}).get("cube_pos")
        if cube_pos is None:
            issues.append("No cube pos")
        elif cube_pos[2] < -0.05:
            issues.append("Cube fell")
        if issues:
            print(f"[warn] Validation: {'; '.join(issues)}")
        if any("fell" in i for i in issues):
            self._force_terminate = True
            self._termination_reason = "cube_fell"

    def _capture_images(self):
        ts = int(time.time() * 1000)
        for k, cam in [("top", self.top_camera), ("side", self.side_camera)]:
            if cam:
                try:
                    write_png(os.path.join(self.temp_dir, f"{k}_{ts}.png"), cam.get_rgba())
                except Exception:
                    pass
        try:
            wrist = getattr(self.robot, "wrist_camera", None)
            if wrist:
                write_png(os.path.join(self.temp_dir, f"wrist_{ts}.png"), wrist.get_rgba())
        except Exception:
            pass

def run_demo():
    env = IsaacPickPlaceEnv(headless=False)
    env.reset()
    try:
        for _ in range(100):
            env.step(env.robot.get_random_joint_positions(), render=True)
    finally:
        env.shutdown()

if __name__ == "__main__":
    run_demo()

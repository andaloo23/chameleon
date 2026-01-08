import os
import shutil
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
from isaacsim import SimulationApp

from cup_utils import create_cup_prim, initialize_usd_modules
from domain_randomizer import DomainRandomizer
from image_utils import write_png
from reward_engine import RewardEngine
from gripper_weld import IntelligentGripperWeld, quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from workspace import CUP_CUBE_MIN_DISTANCE, WORKSPACE_RADIUS_RANGE, sample_workspace_xy
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
    global World, DynamicCuboid, get_context, Camera, Gf, UsdGeom, UsdPhysics, Usd, SO100Robot

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

        # Standardized Intelligent Weld system
        self.grasp_mode = "weld"
        self.use_sticky_gripper = False
        self.use_physics_gripper = False
        self.use_weld_gripper = True

        # Physics parameters
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
        self._gripper_link_idx = None
        self._jaw_link_idx = None

        self.reward_engine = RewardEngine(self)
        self.domain_randomizer = DomainRandomizer(self)
        self.last_validation_result = {"ok": True, "issues": [], "flags": {}}
        self._force_terminate = False
        self._termination_reason = None
        self._cup_upright_threshold_rad = np.deg2rad(25.0)
        
        # Intelligent physics weld
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
        
        # Initialize RMPFlow Controller
        try:
            from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
            
            # Scaled URDF and split config paths
            urdf_path = os.path.join(self.current_dir, "so100.urdf")
            robot_description_path = os.path.join(self.current_dir, "robot_description.yaml")
            rmpflow_config_path = os.path.join(self.current_dir, "rmpflow_config.yaml")
            
            # Initialize RMPFlow
            self.rmpflow = RmpFlow(
                robot_description_path=robot_description_path,
                urdf_path=urdf_path,
                rmpflow_config_path=rmpflow_config_path,
                end_effector_frame_name="gripper",
                maximum_substep_size=0.0033
            )
            
            # Define articulation policy
            self.motion_policy = ArticulationMotionPolicy(self.robot_articulation, self.rmpflow)
            
            # Sync initial base pose
            p, o = self.robot_articulation.get_world_pose()
            self.rmpflow.set_robot_base_pose(p, o)
            
            print("[INFO] RMPFlow Controller initialized with scaled assets and base pose synced.")
        except Exception as e:
            print(f"[CRITICAL] Failed to initialize RMPFlow: {e}")
            raise

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
            try: physics_context.enable_gpu_dynamics(True)
            except Exception: pass
            try: physics_context.set_physics_dt(1.0 / 120.0)
            except Exception: pass
            try: physics_context.enable_ccd(True)
            except Exception: pass
        except Exception as e:
            print(f"[warn] Could not configure physics: {e}")

        self.robot = SO100Robot(self.world, os.path.join(self.current_dir, "so100.urdf"))
        self.robot_articulation = self.robot.get_robot()
        
        self._base_fixture_pose = None
        self._workspace_origin_xy = None
        self._default_joint_positions = self._compute_default_joint_positions()

        cube_xy, cup_xy = self._sample_object_positions()
        cube_position = np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0])

        self.cube = DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=np.array([0, 0, 0]),
            scale=self.cube_scale,
            size=1.0,
            color=np.array([1, 0, 0]),
            mass=2.0, # Increased mass for stability
        )
        self.world.scene.add(self.cube)
        
        try:
            cube_prim = self.world.stage.GetPrimAtPath("/World/Cube")
            if cube_prim and cube_prim.IsValid():
                UsdPhysics.CollisionAPI.Apply(cube_prim)
                material_api = UsdPhysics.MaterialAPI.Apply(cube_prim)
                material_api.CreateStaticFrictionAttr().Set(2.0) # High friction
                material_api.CreateDynamicFrictionAttr().Set(2.0)
                material_api.CreateRestitutionAttr().Set(0.0)
        except Exception: pass

        # cup_translation = np.array([cup_xy[0], cup_xy[1], 0.0])
        # self.cup_xform = create_cup_prim(
        #     self.stage_context.get_stage(), prim_path="/World/Cup", position=cup_translation,
        #     outer_radius_top=self.cup_outer_radius_top, outer_radius_bottom=self.cup_outer_radius_bottom,
        #     inner_radius_top=self.cup_inner_radius_top, inner_radius_bottom=self.cup_inner_radius_bottom,
        #     height=self.cup_height, bottom_thickness=self.cup_bottom_thickness, color=self.cup_color, mass=self.cup_mass
        # )

        self.top_camera = Camera(prim_path="/World/top_camera", name="top_camera", position=np.array([0, 0, 1.5]), resolution=(128, 128))
        self.side_camera = Camera(prim_path="/World/side_camera", name="side_camera", position=np.array([1.0, 0, 0.5]), resolution=(128, 128))
        
        # Ensure square pixels and consistent aspect ratio
        for cam in [self.top_camera, self.side_camera]:
            if cam.prim.IsValid():
                cam.set_vertical_aperture(2.0955)
                cam.set_horizontal_aperture(2.0955)
        
        self.world.scene.add(self.top_camera)
        self.world.scene.add(self.side_camera)

        self.world.reset()
        
        # Set object positions AFTER world.reset() to prevent them being snapped back to defaults
        self.cube.set_world_pose(position=np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0]), orientation=np.array([1, 0, 0, 0]))
        if self.cup_xform:
            from pxr import Gf, UsdGeom
            UsdGeom.XformCommonAPI(self.cup_xform).SetTranslate(Gf.Vec3d(float(cup_xy[0]), float(cup_xy[1]), 0.0))

        self._apply_default_joint_positions()
        self._capture_base_fixture_pose()
        self._resolve_link_paths()
        
        tp, to = np.array([0, -0.75, 8.0]), np.array([-0.5, -0.5, -0.5, 0.5])
        sp, so = np.array([6.0, -0.5, 0.5]), np.array([0, 0, 0, 1])
        self.top_camera.set_world_pose(position=tp, orientation=to)
        self.side_camera.set_world_pose(position=sp, orientation=so)
        self._fixed_camera_poses = {"top": (tp, to), "side": (sp, so)}

        self.robot.create_wrist_camera()
        self.robot.update_wrist_camera_position(verbose=False)
        self.top_camera.initialize()
        self.side_camera.initialize()
        if getattr(self.robot, "wrist_camera", None): self.robot.wrist_camera.initialize()

        self._apply_gripper_friction()
        self._configure_gripper_drive()
        self.robot.configure_drives() # Set up PD controllers for arm
        
        # Increased warm-up for camera/synthetic data stability
        for _ in range(20): self.world.step(render=not self.headless)
        
        self._cube_xy, self._cup_xy = cube_xy, cup_xy
        self._apply_domain_randomization()

    def reset(self, render=None):
        self._step_counter = 0
        render = not self.headless if render is None else bool(render)
        if self.gripper_weld: self.gripper_weld.release()
        cube_xy, cup_xy = self._sample_object_positions()
        self._cube_xy, self._cup_xy = cube_xy, cup_xy
        self.world.reset()
        
        # Set object positions AFTER world.reset()
        pos = np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0])
        print(f"[DEBUG] Setting cube world pose to: {pos}")
        self.cube.set_world_pose(position=pos, orientation=np.array([1, 0, 0, 0]))
        if self.cup_xform:
            from pxr import Gf, UsdGeom
            UsdGeom.XformCommonAPI(self.cup_xform).SetTranslate(Gf.Vec3d(float(cup_xy[0]), float(cup_xy[1]), 0.0))

        self._apply_default_joint_positions()
        self._restore_base_fixture_pose()
        self._restore_fixed_camera_poses()
        self.robot.update_wrist_camera_position(verbose=False)
        self._resolve_link_paths()
        
        # Sync robot base pose to RMPFlow after reset
        if self.rmpflow:
            p, o = self.robot_articulation.get_world_pose()
            self.rmpflow.set_robot_base_pose(p, o)
            
        for i in range(5): self.world.step(render=(render if i == 4 else False))
        self._apply_domain_randomization()
        self.reward_engine.reset()
        if self.gripper_weld: self.gripper_weld.reset()
        self._force_terminate, self._termination_reason = False, None
        self._latest_target_gripper, self._prev_gripper_value = None, None
        self._last_gripper_pose, self._last_jaw_pos = (None, None), None
        
        # Sync initial state to RMPFlow
        if self.rmpflow:
            self.rmpflow.set_robot_base_pose(*self.robot_articulation.get_world_pose())
            
        # Ensure cameras have a chance to render before observation
        for _ in range(10): self.world.step(render=render)
            
        return self._get_observation()

    def _get_link_poses(self):
        """Helper to get link poses reliably across versions."""
        if not self.robot_articulation: return None, None
        
        # 1. Try ArticulationView if available
        view = getattr(self.robot_articulation, "_articulation_view", None)
        if view is not None:
            # Try different method names across Isaac Sim versions
            for method_name in ["get_link_poses", "get_world_poses"]:
                method = getattr(view, method_name, None)
                if method:
                    try:
                        lp, lo = method()
                        return lp[0], lo[0]
                    except Exception: pass
            
        # 2. Try direct Articulation method
        try:
            return self.robot_articulation.get_link_poses()
        except Exception: pass
        
        # 3. Fallback to USD prims (might be 1 frame behind but works)
        try:
            # If we have the prim paths, we can query them directly
            if self._gripper_prim_path and self._jaw_prim_path:
                stage = self.world.stage
                lp = [None] * (max(self._gripper_link_idx or 0, self._jaw_link_idx or 0) + 1)
                lo = [None] * (max(self._gripper_link_idx or 0, self._jaw_link_idx or 0) + 1)
                
                for path, idx in [(self._gripper_prim_path, self._gripper_link_idx), 
                                 (self._jaw_prim_path, self._jaw_link_idx)]:
                    if idx is not None:
                        prim = stage.GetPrimAtPath(path)
                        if prim.IsValid():
                            mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                            lp[idx] = np.array(mat.ExtractTranslation(), dtype=np.float32)
                            # Simple rotation extraction
                            rm = np.array([[mat[0][0], mat[0][1], mat[0][2]], 
                                         [mat[1][0], mat[1][1], mat[1][2]], 
                                         [mat[2][0], mat[2][1], mat[2][2]]], dtype=np.float32)
                            lo[idx] = rotation_matrix_to_quaternion(rm.T)
                return lp, lo
        except Exception: pass
        
        return None, None

    def step(self, action, render=None):
        render = not self.headless if render is None else bool(render)
        jt = self._clip_action(action)
        self.robot.set_joint_positions(jt)
        self._latest_target_gripper = float(jt[-1])
        self._restore_base_fixture_pose()
        self.world.step(render=render)
        self._step_counter += 1
        
        try:
            ap = self.robot_articulation.get_joint_positions()
            agp = float(ap[-1])
        except Exception: agp = float(jt[-1])
        
        cp, co = None, None
        try:
            cp, co = self.cube.get_world_pose()
            cp, co = np.array(cp, dtype=np.float32), np.array(co, dtype=np.float32)
        except Exception: pass
        
        gwp, gwo, jwp = None, None, None
        try:
            if self.robot_articulation and self._gripper_link_idx is not None and self._jaw_link_idx is not None:
                lp, lo = self._get_link_poses()
                if lp is not None:
                    gwp, gwo = np.array(lp[self._gripper_link_idx], dtype=np.float32), np.array(lo[self._gripper_link_idx], dtype=np.float32)
                    jwp = np.array(lp[self._jaw_link_idx], dtype=np.float32)
            
            if gwp is None and Usd is not None:
                stage = self.world.stage
                if not self._gripper_prim_path or not self._jaw_prim_path: self._resolve_link_paths()
                if self._gripper_prim_path:
                    gp = stage.GetPrimAtPath(self._gripper_prim_path)
                    if gp.IsValid():
                        mat = UsdGeom.Xformable(gp).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        gwp = np.array(mat.ExtractTranslation(), dtype=np.float32)
                        rm = np.array([[mat[0][0], mat[0][1], mat[0][2]], [mat[1][0], mat[1][1], mat[1][2]], [mat[2][0], mat[2][1], mat[2][2]]], dtype=np.float32)
                        gwo = rotation_matrix_to_quaternion(rm.T)
                if self._jaw_prim_path:
                    jp = stage.GetPrimAtPath(self._jaw_prim_path)
                    if jp.IsValid():
                        jwp = np.array(UsdGeom.Xformable(jp).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation(), dtype=np.float32)
            self._last_gripper_pose, self._last_jaw_pos = (gwp, gwo), jwp
        except Exception: pass
        
        if gwp is None:
            res = self._compute_gripper_pose_from_camera()
            if res: gwp, gwo = res; self._last_gripper_pose = (gwp, gwo)
        if jwp is None and gwp is not None:
            if self._step_counter % 100 == 0: print(f"[ENV] Approx jaw with gripper")
            jwp = gwp.copy()
        
        if self.gripper_weld:
            self.gripper_weld.update(
                gripper_value=agp, target_gripper=self._latest_target_gripper,
                gripper_world_pos=gwp, gripper_world_orient=gwo, jaw_world_pos=jwp,
                object_world_pos=cp, object_world_orient=co,
                object_prim_path=getattr(self.cube, "prim_path", "/World/Cube"),
                gripper_body_path=self._gripper_prim_path or "/World/Robot/gripper",
                jaw_body_path=self._jaw_prim_path or "/World/Robot/jaw"
            )

        # RMPFlow doesn't strictly need a "publish" step here as it queries state when calculate_joint_command is called
        # but we ensure the base pose is up to date if the robot moves
        if self.rmpflow:
            self.rmpflow.set_robot_base_pose(*self.robot_articulation.get_world_pose())

        obs = self._get_observation()
        self.reward_engine.compute_reward_components()
        self._validate_state(obs)
        self._prev_gripper_value = agp
        reward, done, info = self.reward_engine.summarize_reward()
        if self._force_terminate:
            done, info["termination_reason"] = True, self._termination_reason or "validation_failure"
        if self.capture_images and (self._step_counter % self.image_interval == 0): self._capture_images()
        return obs, reward, done, info

    def close(self):
        if self.gripper_weld: self.gripper_weld.release()
        if self.world: self.world.clear()

    def shutdown(self):
        self.close()
        if self.simulation_app: self.simulation_app.close()


    def _sample_object_positions(self):
        # Use the workspace sampling logic to ensure reachable positions
        from workspace import sample_workspace_xy
        import random
        
        rng = np.random.RandomState(random.randint(0, 10000))
        
        cube_xy = sample_workspace_xy(rng)
        cup_xy = sample_workspace_xy(rng, existing=[cube_xy])
        
        print(f"[INFO] _sample_object_positions: cube={cube_xy}, cup={cup_xy}")
        return cube_xy, cup_xy

    def _clip_action(self, action):
        lower = np.array([self.robot.joint_limits[n][0] for n in self.robot.joint_names])
        upper = np.array([self.robot.joint_limits[n][1] for n in self.robot.joint_names])
        span = np.maximum(upper - lower, 1e-6)
        margin = np.clip(0.05 * span, 1e-3, 0.1)
        return np.clip(action, lower + margin, upper - margin)

    def _get_observation(self):
        cf = self._gather_sensor_observations()
        jp, jv = self.robot_articulation.get_joint_positions(), self.robot_articulation.get_joint_velocities()
        
        # Safety check for cases where the simulation view isn't fully ready
        if jp is None:
            jp = self._default_joint_positions
        if jv is None:
            jv = np.zeros_like(jp)
            
        self.reward_engine.record_joint_state(jp.astype(np.float32), jv.astype(np.float32), self._latest_target_gripper)
        
        cube_pos, _ = self.cube.get_world_pose()
        
        # Get cup position from USD world transform
        cup_pos = np.zeros(3, dtype=np.float32)
        if self.cup_xform:
            try:
                cup_mat = UsdGeom.Xformable(self.cup_xform).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                cup_pos_gf = cup_mat.ExtractTranslation()
                cup_pos = np.array([cup_pos_gf[0], cup_pos_gf[1], cup_pos_gf[2]], dtype=np.float32)
            except Exception:
                pass

        gwp, gwo = self._last_gripper_pose
        obs = {
            "joint_positions": jp.astype(np.float32), 
            "joint_velocities": jv.astype(np.float32), 
            "cube_pos": np.array(cube_pos, dtype=np.float32),
            "cup_pos": cup_pos,
            "gripper_pos": gwp, 
            "gripper_orient": gwo
        }
        obs.update(cf)
        return obs

    def _gather_sensor_observations(self):
        cams = {"top": self.top_camera, "side": self.side_camera, "wrist": getattr(self.robot, "wrist_camera", None)}
        frames = {}
        for k, c in cams.items():
            processed = None
            if c:
                try:
                    processed = self._prepare_camera_frame(c.get_rgba(), k)
                    if processed is not None: self._last_camera_frames[k], self._camera_failure_logged[k] = processed, False
                except Exception: self._camera_failure_logged[k] = True
            if processed is None:
                cached = self._last_camera_frames.get(k)
                processed = cached if cached is not None else self._allocate_empty_camera_frame(c, k)
                self._last_camera_frames[k] = processed
            frames[f"{k}_camera_rgb"] = processed
        return frames

    def _prepare_camera_frame(self, frame, key):
        if frame is None: return None
        array = np.asarray(frame)
        if array.ndim == 2: array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3:
            ch = array.shape[-1]
            if ch >= 3: array = array[..., :3]
            elif ch == 1: array = np.repeat(array, 3, axis=-1)
            else: array = np.concatenate([array, np.zeros(array.shape[:2] + (3-ch,), dtype=array.dtype)], axis=-1)
        if np.issubdtype(array.dtype, np.integer): array = array.astype(np.float32) / 255.0
        else: array = np.clip(array.astype(np.float32, copy=False), 0.0, 1.0)
        self._camera_frame_shapes[key] = array.shape
        return np.ascontiguousarray(array)

    def _allocate_empty_camera_frame(self, camera, key):
        if key in self._camera_frame_shapes: h, w, _ = self._camera_frame_shapes[key]
        else:
            w, h = 128, 128
            if camera:
                res = getattr(camera, "get_resolution", lambda: getattr(camera, "resolution", (128, 128)))()
                w, h = res[0], res[1]
            self._camera_frame_shapes[key] = (int(h), int(w), 3)
        h, w, _ = self._camera_frame_shapes[key]
        return np.zeros((h, w, 3), dtype=np.float32)

    def _compute_gripper_pose_from_camera(self):
        wrist = getattr(self.robot, "wrist_camera", None)
        if not wrist: return None
        try:
            pw, ow = wrist.get_world_pose()
            pl, ol = wrist.get_local_pose()
            Rw, Rl = quaternion_to_rotation_matrix(np.asarray(ow, dtype=float)), quaternion_to_rotation_matrix(np.asarray(ol, dtype=float))
            Rg = Rw @ Rl.T
            return np.asarray(pw, dtype=float) - Rg @ np.asarray(pl, dtype=float), rotation_matrix_to_quaternion(Rg)
        except Exception: return None

    def _compute_default_joint_positions(self):
        # Default "home" position: arm raised and ready
        # Based on empirical testing:
        # - Lower lift + more bent elbow = arm raised/retracted
        # - Higher lift + straighter elbow = arm extended forward/down
        d = {
            "shoulder_pan": 0.0,      # Centered
            "shoulder_lift": 2.0,     # Moderate - arm not fully raised
            "elbow_flex": -1.0,       # Bent elbow (retracted)
            "wrist_flex": -0.5,       # Wrist angled
            "wrist_roll": 0.0,        # No roll
            "gripper": 1.0            # Gripper open
        }
        return np.asarray([float(np.clip(d.get(n, 0.0), *self.robot.joint_limits[n])) for n in self.robot.joint_names], dtype=np.float32)

    def _apply_default_joint_positions(self):
        if self.robot and self._default_joint_positions is not None:
            try: self.robot.set_joint_positions(self._default_joint_positions, use_targets=False)
            except Exception: pass

    def _capture_base_fixture_pose(self):
        if self.robot_articulation:
            try:
                p, o = self.robot_articulation.get_world_pose()
                self._base_fixture_pose = (np.asarray(p, dtype=float), np.asarray(o, dtype=float))
                self._workspace_origin_xy = self._base_fixture_pose[0][:2].copy()
            except Exception: pass

    def _restore_base_fixture_pose(self):
        if self.robot_articulation and self._base_fixture_pose:
            try: self.robot_articulation.set_world_pose(position=self._base_fixture_pose[0], orientation=self._base_fixture_pose[1])
            except Exception: pass
    
    def _restore_fixed_camera_poses(self):
        if not self._fixed_camera_poses: return
        try:
            for k, (p, o) in self._fixed_camera_poses.items():
                cam = getattr(self, f"{k}_camera", None)
                if cam: cam.set_world_pose(position=p, orientation=o)
        except Exception: pass

    def _resolve_link_paths(self):
        stage, root = self.world.stage, getattr(self.robot, "prim_path", None)
        if not stage or not root: return
        def _find(name):
            d = f"{root}/{name}"
            if stage.GetPrimAtPath(d).IsValid(): return d
            alts = [f"{root}/base_link/{name}", f"{root}/{name}_link", f"{root}/base/{name}", f"{root}/gripper/{name}" if name=="jaw" else None, f"{root}/fixed_jaw" if name=="gripper" else None, f"{root}/moving_jaw" if name=="jaw" else None]
            for a in alts:
                if a and stage.GetPrimAtPath(a).IsValid(): return a
            for p in stage.Traverse():
                ps = str(p.GetPath())
                if not ps.startswith(root): continue
                l = ps.split("/")[-1].lower()
                if l == name.lower() or l.startswith(name.lower()+"_") or l.endswith("_"+name.lower()) or (name=="jaw" and "moving" in l) or (name=="gripper" and "fixed" in l):
                    if p.IsA(UsdGeom.Xformable): return ps
            return None
        self._gripper_prim_path, self._jaw_prim_path = _find("gripper") or self._gripper_prim_path, _find("jaw") or self._jaw_prim_path
        if not self._gripper_prim_path or not self._jaw_prim_path:
            try:
                for p in stage.Traverse():
                    if not str(p.GetPath()).startswith(root): continue
                    if p.IsA(UsdPhysics.Joint):
                        jt = UsdPhysics.Joint(p)
                        rel0 = getattr(jt, "GetChild0Rel", lambda: None)()
                        if rel0:
                            for t in rel0.GetTargets():
                                tl = str(t).split("/")[-1].lower()
                                if "gripper" in tl and not self._gripper_prim_path: self._gripper_prim_path = str(t)
                                if ("jaw" in tl or "finger" in tl) and not self._jaw_prim_path: self._jaw_prim_path = str(t)
            except Exception: pass
        if self._gripper_prim_path and not self._jaw_prim_path:
            for c in stage.GetPrimAtPath(self._gripper_prim_path).GetChildren():
                if c.IsA(UsdGeom.Xformable): self._jaw_prim_path = str(c.GetPath()); break
        if self.robot_articulation:
            try:
                if self._gripper_prim_path: self._gripper_link_idx = self.robot_articulation.get_link_index(self._gripper_prim_path.split("/")[-1])
                if self._jaw_prim_path: self._jaw_link_idx = self.robot_articulation.get_link_index(self._jaw_prim_path.split("/")[-1])
                print(f"[ENV] Links resolved: G={self._gripper_link_idx} J={self._jaw_link_idx}")
            except Exception: pass

    def _apply_gripper_friction(self):
        try:
            from pxr import PhysxSchema
            stage, root = self.world.stage, getattr(self.robot, "prim_path", "/World/Robot")
            for p in [f"{root}/gripper", f"{root}/jaw"]:
                pr = stage.GetPrimAtPath(p)
                if pr.IsValid():
                    mat = UsdPhysics.MaterialAPI.Apply(pr)
                    mat.CreateStaticFrictionAttr().Set(float(self._gripper_friction))
                    mat.CreateDynamicFrictionAttr().Set(float(self._gripper_friction))
                    ps = PhysxSchema.PhysxMaterialAPI.Apply(pr)
                    ps.CreateFrictionCombineModeAttr().Set("max")
                    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(pr)
                    rb.CreateContactOffsetAttr().Set(float(self._contact_offset))
                    rb.CreateRestOffsetAttr().Set(float(self._rest_offset))
        except Exception: pass

    def _configure_gripper_drive(self):
        try:
            from pxr import PhysxSchema
            stage, root = self.world.stage, getattr(self.robot, "prim_path", "/World/Robot")
            gj = None
            for p in [f"{root}/gripper/gripper", f"{root}/gripper_joint", f"{root}/joints/gripper"]:
                if stage.GetPrimAtPath(p).IsValid(): gj = stage.GetPrimAtPath(p); break
            if not gj:
                for p in stage.Traverse():
                    if "gripper" in str(p.GetPath()).lower() and p.IsA(UsdPhysics.RevoluteJoint): gj = p; break
            if gj:
                if not gj.HasAPI(PhysxSchema.PhysxJointAPI): PhysxSchema.PhysxJointAPI.Apply(gj)
                dr = UsdPhysics.DriveAPI.Apply(gj, "angular")
                dr.CreateStiffnessAttr().Set(float(self._gripper_drive_stiffness))
                dr.CreateDampingAttr().Set(float(self._gripper_drive_damping))
                dr.CreateMaxForceAttr().Set(float(self._gripper_drive_max_force))
        except Exception: pass
    
    def _apply_domain_randomization(self):
        if self.domain_randomizer: self.domain_randomizer.randomize()

    def _get_recent_collisions(self):
        try:
            physics = self.world.get_physics_context()
            report = getattr(physics, "get_contact_report", lambda: None)()
            if not report: return []
            entries = report.get("contacts", []) if isinstance(report, dict) else report
            return [{"prim0": e.get("body0"), "prim1": e.get("body1")} for e in entries if isinstance(e, dict)]
        except Exception: return []

    def _validate_state(self, _obs):
        issues = []
        pos = self.robot_articulation.get_joint_positions()
        if not np.isfinite(pos).all(): issues.append("NaN joints")
        cube_pos = getattr(self.reward_engine, "task_state", {}).get("cube_pos")
        if cube_pos is not None and cube_pos[2] < -0.05: issues.append("Cube fell")
        if issues: print(f"[warn] Validation: {'; '.join(issues)}")
        if any("fell" in i for i in issues): self._force_terminate, self._termination_reason = True, "cube_fell"

    def _capture_images(self):
        ts = int(time.time() * 1000)
        for k in ["top", "side"]:
            cam = getattr(self, f"{k}_camera", None)
            if cam:
                try: write_png(os.path.join(self.temp_dir, f"{k}_{ts}.png"), cam.get_rgba())
                except Exception: pass
        wrist = getattr(self.robot, "wrist_camera", None)
        if wrist:
            try: write_png(os.path.join(self.temp_dir, f"wrist_{ts}.png"), wrist.get_rgba())
            except Exception: pass

def run_demo():
    env = IsaacPickPlaceEnv(headless=False)
    env.reset()
    try:
        for _ in range(100): env.step(env.robot.get_random_joint_positions(), render=True)
    finally: env.shutdown()

if __name__ == "__main__": run_demo()

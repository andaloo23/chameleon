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
        print(f"[warn] SimulationApp already initialized with headless={_SIM_HEADLESS_FLAG}; "
              f"requested headless={headless} ignored.")

    return _SIMULATION_APP


class IsaacPickPlaceEnv:
    """Isaac Sim environment wrapper for the pick-and-place task."""

    def __init__(self, headless=False, capture_images=False, image_interval=3, random_seed=None):
        self.headless = headless
        self.capture_images = capture_images
        self.image_interval = max(1, int(image_interval))
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

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

        self.cube_scale = np.array([0.075, 0.075, 0.075], dtype=float)  # 2.5x larger (was 0.03)
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
        self._fixed_camera_poses = {}  # Store fixed camera positions to restore after reset

        self.reward_engine = RewardEngine(self)
        self.domain_randomizer = DomainRandomizer(self)
        self.last_validation_result = {"ok": True, "issues": [], "flags": {}}
        self._force_terminate = False
        self._termination_reason = None
        self._cup_upright_threshold_rad = np.deg2rad(25.0)
        

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
        
        # Comprehensive physics configuration for reliable grasping
        try:
            physics_context = self.world.get_physics_context()
            
            # Use TGS solver - better for articulations and contacts
            physics_context.set_solver_type("TGS")
            
            # Significantly increase solver iterations for accurate contact resolution
            # Default is 4/1, we use much higher for reliable grasping
            physics_context.set_position_iteration_count(64)   # Very high for accurate contacts
            physics_context.set_velocity_iteration_count(32)   # High for stable friction
            
            # Enable GPU dynamics if available for better performance with high iterations
            try:
                physics_context.enable_gpu_dynamics(True)
            except Exception:
                pass
            
            # Set physics timestep - smaller = more accurate but slower
            # Default is usually 1/60, we use 1/120 for better contact accuracy
            try:
                physics_context.set_physics_dt(1.0 / 120.0)  # 120 Hz physics
                print("[info] Set physics timestep to 1/120s")
            except Exception:
                pass
            
            # Enable CCD (Continuous Collision Detection) for fast-moving objects
            try:
                physics_context.enable_ccd(True)
            except Exception:
                pass
                
            print("[info] Configured physics: TGS solver, 64/32 iterations, 120Hz")
            
        except Exception as e:
            print(f"[warn] Could not configure physics solver: {e}")

        urdf_path = os.path.join(self.current_dir, "so100.urdf")
        self.robot = SO100Robot(self.world, urdf_path)
        self.robot_articulation = self.robot.get_robot()
        self._base_fixture_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._workspace_origin_xy: Optional[np.ndarray] = None
        self._default_joint_positions = self._compute_default_joint_positions()

        cube_xy, cup_xy = self._sample_object_positions()
        cube_position = np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0])

        self.cube = DynamicCuboid(
            prim_path="/World/Cube",
            name="collision_cube",
            position=cube_position,
            scale=self.cube_scale,
            color=np.array([0.0, 0.5, 1.0]),
            mass=0.05,  # Give cube some mass for proper physics
        )
        self.world.scene.add(self.cube)
        
        # Ensure cube has collision and friction properties
        try:
            cube_prim = self.world.stage.GetPrimAtPath("/World/Cube")
            if cube_prim and cube_prim.IsValid():
                # Make sure collision is enabled
                if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(cube_prim)
                
                # Add friction material for graspability
                material_api = UsdPhysics.MaterialAPI.Apply(cube_prim)
                material_api.CreateStaticFrictionAttr().Set(1.0)
                material_api.CreateDynamicFrictionAttr().Set(1.0)
                material_api.CreateRestitutionAttr().Set(0.0)
                print("[info] Applied friction properties to cube (static=1.0, dynamic=1.0)")
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
        top_cam_pos = np.array([0, -0.75, 8.0])
        top_cam_orient = np.array([-np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25), np.sqrt(0.25)])
        side_cam_pos = np.array([6.0, -0.5, 0.5])
        side_cam_orient = np.array([0, 0, 0, 1])

        self.top_camera.set_world_pose(position=top_cam_pos, orientation=top_cam_orient)
        self.side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)
        
        # Store fixed camera poses for restoration after reset
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

        # Apply friction to gripper for proper grasping
        self._apply_gripper_friction()
        
        # Configure gripper joint drive for high gripping force
        self._configure_gripper_drive()

        for _ in range(5):
            self.world.step(render=not self.headless)

        self._cube_xy = cube_xy
        self._cup_xy = cup_xy

        self._apply_domain_randomization()

    def reset(self, render=None):
        """Reset the scene with new randomized object placements."""
        self._step_counter = 0
        render = not self.headless if render is None else bool(render)

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
        self._restore_fixed_camera_poses()  # Restore camera positions after world reset
        self.robot.update_wrist_camera_position(verbose=False)

        for i in range(5):
            is_last = (i == 4)
            do_render = render if is_last else False
            self.world.step(render=do_render)

        self._apply_domain_randomization()
        self.reward_engine.reset()
        self.last_validation_result = {"ok": True, "issues": [], "flags": {}}
        self._force_terminate = False
        self._termination_reason = None
        self._latest_target_gripper = None  # Reset target gripper for grasp detection
        
        return self._get_observation()

    def step(self, action, render=None):
        """Advance the simulation by one control step."""
        render = not self.headless if render is None else bool(render)

        joint_targets = self._clip_action(action)
        self.robot.set_joint_positions(joint_targets)
        
        # Store target gripper position for pressure-based grasp detection
        # Gripper is the last joint (index 5) in the action array
        self._latest_target_gripper = float(joint_targets[-1]) if len(joint_targets) > 0 else None

        self._restore_base_fixture_pose()
        self.world.step(render=render)
        self._step_counter += 1

        obs = self._get_observation()
        self.reward_engine.compute_reward_components()
        self._validate_state(obs)
        reward, done, info = self.reward_engine.summarize_reward()

        if self._force_terminate:
            done = True
            info.setdefault("termination_reason", self._termination_reason or "validation_failure")
            info["terminated_by_validation"] = True

        if self.capture_images and (self._step_counter % self.image_interval == 0):
            self._capture_images()

        return obs, reward, done, info

    def close(self):
        if self.world is not None:
            self.world.clear()
        if self.capture_images:
            print(f"[info] saved images to {self.temp_dir}")

    def shutdown(self):
        self.close()
        if self.simulation_app is not None:
            self.simulation_app.close()

    def _sample_object_positions(self):
        # Position cube further away from robot base for easier reaching
        # Y axis: negative = forward from robot
        cube_xy = np.array([0.0, -0.50])  # Was -0.336, now further forward
        
        # Position cup even further away to maintain separation
        cup_xy = np.array([0.0, -0.75])  # Was -0.55, now further forward
        
        return cube_xy, cup_xy

    def _clip_action(self, action):
        action = np.array(action, dtype=float)
        lower = np.array([self.robot.joint_limits[name][0] for name in self.robot.joint_names], dtype=float)
        upper = np.array([self.robot.joint_limits[name][1] for name in self.robot.joint_names], dtype=float)
        if action.shape != lower.shape:
            raise ValueError(f"Expected action of shape {lower.shape}, received {action.shape}")
        span = np.maximum(upper - lower, 1e-6)
        margin = np.minimum(0.05 * span, 0.1)
        margin = np.minimum(margin, span / 2.0 - 1e-4)
        margin = np.maximum(margin, 1e-3)
        lower_safe = lower + margin
        upper_safe = upper - margin
        clipped = np.clip(action, lower_safe, upper_safe)
        return clipped

    def _get_observation(self):
        camera_frames = self._gather_sensor_observations()
        joint_positions = self.robot_articulation.get_joint_positions()
        joint_velocities = self.robot_articulation.get_joint_velocities()
        joint_positions = joint_positions.astype(np.float32, copy=True)
        joint_velocities = joint_velocities.astype(np.float32, copy=True)

        # Pass target gripper position for pressure-based grasp detection
        target_gripper = getattr(self, '_latest_target_gripper', None)
        self.reward_engine.record_joint_state(joint_positions.copy(), joint_velocities.copy(), target_gripper)

        # Add cube position to observation for policy use
        cube_pos = None
        gripper_pos = None
        try:
            cube_pos, _ = self.cube.get_world_pose()
            cube_pos = np.array(cube_pos, dtype=np.float32)
        except Exception:
            pass
        
        try:
            wrist_cam = getattr(self.robot, "wrist_camera", None)
            if wrist_cam is not None:
                gripper_pos, _ = wrist_cam.get_world_pose()
                # NOTE: This is wrist_camera position, not actual gripper finger position
                # Camera is offset by [0.0, 0.05, -0.08] from gripper link in local frame
                # Actual offset distance is ~sqrt(0.05^2 + 0.08^2) ≈ 0.094m
                gripper_pos = np.array(gripper_pos, dtype=np.float32)
        except Exception:
            pass

        obs = {
            "joint_positions": joint_positions.copy(),
            "joint_velocities": joint_velocities.copy(),
            "cube_pos": cube_pos,
            "gripper_pos": gripper_pos,
        }
        for name, frame in camera_frames.items():
            if frame is not None:
                obs[name] = frame.copy()
            else:
                obs[name] = None
        return obs

    def _gather_sensor_observations(self):
        camera_map = {
            "top": self.top_camera,
            "side": self.side_camera,
            "wrist": getattr(self.robot, "wrist_camera", None),
        }
        frames = {}

        for key, camera in camera_map.items():
            processed = None
            if camera is not None:
                try:
                    raw_frame = camera.get_rgba()
                except Exception as exc:
                    if not self._camera_failure_logged.get(key, False):
                        print(f"[warn] failed to fetch {key} camera frame: {exc}")
                        self._camera_failure_logged[key] = True
                    raw_frame = None
                else:
                    processed = self._prepare_camera_frame(raw_frame, key)
                    if processed is not None:
                        self._last_camera_frames[key] = processed
                        self._camera_failure_logged[key] = False
            
            if processed is None:
                cached_frame = self._last_camera_frames.get(key)
                if cached_frame is not None:
                    processed = cached_frame
                else:
                    processed = self._allocate_empty_camera_frame(camera, key)
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
                pad_width = 3 - channels
                pad_shape = array.shape[:2] + (pad_width,)
                array = np.concatenate([array, np.zeros(pad_shape, dtype=array.dtype)], axis=-1)
        else:
            return None

        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32, copy=False)
            array = np.clip(array, 0.0, 1.0)

        array = np.ascontiguousarray(array)
        self._camera_frame_shapes[key] = array.shape
        return array

    def _allocate_empty_camera_frame(self, camera, key):
        if key in self._camera_frame_shapes:
            height, width, _ = self._camera_frame_shapes[key]
        else:
            width, height = 128, 128

            if camera is not None:
                resolution = None
                try:
                    resolution = camera.get_resolution()
                except Exception:
                    resolution = getattr(camera, "resolution", None)

                if resolution is not None and len(resolution) >= 2:
                    width, height = resolution[0], resolution[1]

            width = int(width)
            height = int(height)
            self._camera_frame_shapes[key] = (height, width, 3)

        height, width, _ = self._camera_frame_shapes[key]
        return np.zeros((height, width, 3), dtype=np.float32)

    def _compute_default_joint_positions(self):
        defaults = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.5, 
            "elbow_flex": 1.1,
            "wrist_flex": 0.3,
            "wrist_roll": 0.0,
            "gripper": 0.03,
        }
        joint_values = []
        for name in self.robot.joint_names:
            lower, upper = self.robot.joint_limits[name]
            value = defaults.get(name, 0.0)
            joint_values.append(float(np.clip(value, lower, upper)))
        return np.asarray(joint_values, dtype=np.float32)

    def _apply_default_joint_positions(self):
        if self.robot is None or self._default_joint_positions is None:
            return
        try:
            self.robot.set_joint_positions(self._default_joint_positions)
        except Exception:
            pass

    def _capture_base_fixture_pose(self):
        if self.robot_articulation is None:
            return
        try:
            position, orientation = self.robot_articulation.get_world_pose()
        except Exception:
            return
        position_arr = np.asarray(position, dtype=float)
        orientation_arr = np.asarray(orientation, dtype=float)
        self._base_fixture_pose = (position_arr, orientation_arr)
        self._workspace_origin_xy = position_arr[:2].copy()

    def _restore_base_fixture_pose(self):
        if self.robot_articulation is None or self._base_fixture_pose is None:
            return
        try:
            position, orientation = self._base_fixture_pose
            self.robot_articulation.set_world_pose(position=position, orientation=orientation)
        except Exception:
            pass
    
    def _restore_fixed_camera_poses(self):
        """Restore top and side camera positions after world reset."""
        if not self._fixed_camera_poses:
            return
        
        try:
            if "top" in self._fixed_camera_poses and self.top_camera is not None:
                top_pos, top_orient = self._fixed_camera_poses["top"]
                self.top_camera.set_world_pose(position=top_pos, orientation=top_orient)
            
            if "side" in self._fixed_camera_poses and self.side_camera is not None:
                side_pos, side_orient = self._fixed_camera_poses["side"]
                self.side_camera.set_world_pose(position=side_pos, orientation=side_orient)
        except Exception as e:
            print(f"[warn] Could not restore camera poses: {e}")


    def _apply_gripper_friction(self):
        """Apply high friction and contact properties to gripper fingers."""
        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None
        
        FRICTION = 10.0  # Match cube friction
            
        try:
            stage = self.world.stage
            robot_prim_path = getattr(self.robot, "prim_path", "/World/Robot")
            
            # Based on URDF structure: gripper (fixed jaw) and jaw (moving jaw)
            gripper_paths = [
                f"{robot_prim_path}/gripper",
                f"{robot_prim_path}/jaw",
            ]
            
            applied_count = 0
            for gripper_path in gripper_paths:
                gripper_prim = stage.GetPrimAtPath(gripper_path)
                if gripper_prim and gripper_prim.IsValid():
                    # Apply extreme friction to the link
                    material_api = UsdPhysics.MaterialAPI.Apply(gripper_prim)
                    material_api.CreateStaticFrictionAttr().Set(FRICTION)
                    material_api.CreateDynamicFrictionAttr().Set(FRICTION)
                    material_api.CreateRestitutionAttr().Set(0.0)
                    applied_count += 1
                    print(f"[info] Applied friction={FRICTION} to: {gripper_path}")
                    
                    # PhysX material properties
                    if PhysxSchema:
                        try:
                            physx_material = PhysxSchema.PhysxMaterialAPI.Apply(gripper_prim)
                            physx_material.CreateFrictionCombineModeAttr().Set("max")
                            physx_material.CreateRestitutionCombineModeAttr().Set("min")
                            # Compliant contact for better grip
                            physx_material.CreateCompliantContactStiffnessAttr().Set(1e6)
                            physx_material.CreateCompliantContactDampingAttr().Set(1e4)
                        except Exception:
                            pass
                        
                        # Rigid body contact properties
                        try:
                            rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(gripper_prim)
                            rigid_body.CreateContactOffsetAttr().Set(0.02)
                            rigid_body.CreateRestOffsetAttr().Set(0.001)
                        except Exception:
                            pass
                    
                    # Also apply to collision meshes within the link
                    for child_prim in gripper_prim.GetChildren():
                        child_path_str = str(child_prim.GetPath()).lower()
                        if "collision" in child_path_str or "collisions" in child_path_str:
                            try:
                                child_material_api = UsdPhysics.MaterialAPI.Apply(child_prim)
                                child_material_api.CreateStaticFrictionAttr().Set(FRICTION)
                                child_material_api.CreateDynamicFrictionAttr().Set(FRICTION)
                                child_material_api.CreateRestitutionAttr().Set(0.0)
                                
                                if PhysxSchema:
                                    child_physx = PhysxSchema.PhysxMaterialAPI.Apply(child_prim)
                                    child_physx.CreateFrictionCombineModeAttr().Set("max")
                                    child_physx.CreateCompliantContactStiffnessAttr().Set(1e6)
                                    child_physx.CreateCompliantContactDampingAttr().Set(1e4)
                            except Exception:
                                pass
            
            if applied_count == 0:
                print("[warn] Could not find gripper finger prims to apply friction")
            else:
                print(f"[info] Applied friction={FRICTION} + compliant contact to {applied_count} gripper parts")
                
        except Exception as e:
            print(f"[warn] Could not apply gripper friction: {e}")

    def _configure_gripper_drive(self):
        """Configure gripper joint drive for high gripping force.
        
        This increases the stiffness (position gain) and max force of the gripper
        joint so it applies significant pressure when trying to close past an object.
        """
        try:
            from pxr import PhysxSchema
            
            stage = self.world.stage
            robot_prim_path = getattr(self.robot, "prim_path", "/World/Robot")
            
            # Try multiple possible joint paths (depends on URDF import)
            possible_paths = [
                f"{robot_prim_path}/gripper/gripper",      # Joint inside parent link
                f"{robot_prim_path}/gripper_joint",        # Joint at robot root
                f"{robot_prim_path}/joints/gripper",       # Joint in joints folder
            ]
            
            gripper_joint_prim = None
            used_path = None
            for path in possible_paths:
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    gripper_joint_prim = prim
                    used_path = path
                    break
            
            if not gripper_joint_prim:
                # Try to find it by traversing the stage
                print(f"[warn] Gripper joint not found at expected paths, searching...")
                for prim in stage.Traverse():
                    prim_path = str(prim.GetPath())
                    if "gripper" in prim_path.lower() and prim.IsA(UsdPhysics.RevoluteJoint):
                        gripper_joint_prim = prim
                        used_path = prim_path
                        print(f"[info] Found gripper joint at: {used_path}")
                        break
            
            if not gripper_joint_prim:
                print(f"[warn] Could not find gripper joint prim")
                # Still try to increase cube friction
                self._increase_cube_friction()
                return
            
            # Apply PhysxJointAPI if not already applied
            if not gripper_joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(gripper_joint_prim)
            
            # Get or create the drive API for angular (revolute) joint
            drive_api = UsdPhysics.DriveAPI.Get(gripper_joint_prim, "angular")
            if not drive_api:
                drive_api = UsdPhysics.DriveAPI.Apply(gripper_joint_prim, "angular")
            
            # Configure very high stiffness and force for strong gripping
            # These values create significant normal force between gripper and object
            STIFFNESS = 50000.0   # Extremely high position gain
            DAMPING = 1000.0      # High damping to prevent oscillation
            MAX_FORCE = 5000.0    # Very high max force (50x normal)
            
            drive_api.CreateStiffnessAttr().Set(STIFFNESS)
            drive_api.CreateDampingAttr().Set(DAMPING)
            drive_api.CreateMaxForceAttr().Set(MAX_FORCE)
            
            print(f"[info] Configured gripper drive at {used_path}: stiffness={STIFFNESS}, damping={DAMPING}, maxForce={MAX_FORCE}")
            
            # Also increase friction on cube for better grip
            self._increase_cube_friction()
            
        except ImportError as e:
            print(f"[warn] PhysxSchema not available ({e}), trying alternative approach...")
            self._configure_gripper_via_articulation()
        except Exception as e:
            print(f"[warn] Could not configure gripper drive: {e}")
            import traceback
            traceback.print_exc()
            # Still try articulation approach
            self._configure_gripper_via_articulation()

    def _configure_gripper_via_articulation(self):
        """Alternative method to configure gripper using ArticulationController."""
        try:
            if self.robot_articulation is None:
                return
            
            controller = self.robot_articulation.get_articulation_controller()
            if controller is not None:
                joint_count = len(self.robot.joint_names)
                
                # High gains for arm joints
                stiffnesses = [5000.0] * joint_count
                dampings = [500.0] * joint_count
                
                # Extremely high for gripper (last joint)
                stiffnesses[-1] = 50000.0  # Gripper stiffness - very high
                dampings[-1] = 1000.0      # Gripper damping
                
                try:
                    controller.set_gains(kps=stiffnesses, kds=dampings)
                    print(f"[info] Set articulation gains - gripper stiffness=50000, damping=1000")
                except Exception as e:
                    print(f"[warn] Could not set articulation gains: {e}")
            
            # Always increase cube friction
            self._increase_cube_friction()
            
        except Exception as e:
            print(f"[warn] Could not configure gripper via articulation: {e}")

    def _increase_cube_friction(self):
        """Increase cube friction and contact properties for reliable grasping."""
        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None
            
        try:
            stage = self.world.stage
            cube_prim = stage.GetPrimAtPath("/World/Cube")
            if cube_prim and cube_prim.IsValid():
                # Apply extremely high friction
                material_api = UsdPhysics.MaterialAPI.Get(cube_prim)
                if not material_api:
                    material_api = UsdPhysics.MaterialAPI.Apply(cube_prim)
                
                # Extreme friction values for reliable grasping
                FRICTION = 10.0  # Very high friction coefficient
                material_api.CreateStaticFrictionAttr().Set(FRICTION)
                material_api.CreateDynamicFrictionAttr().Set(FRICTION)
                material_api.CreateRestitutionAttr().Set(0.0)
                print(f"[info] Set cube friction to {FRICTION}")
                
                # PhysX-specific contact properties
                if PhysxSchema:
                    try:
                        # Material combine mode
                        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(cube_prim)
                        physx_material.CreateFrictionCombineModeAttr().Set("max")
                        physx_material.CreateRestitutionCombineModeAttr().Set("min")
                        
                        # Compliant contact for better grip (adds softness to contact)
                        physx_material.CreateCompliantContactStiffnessAttr().Set(1e6)  # High stiffness
                        physx_material.CreateCompliantContactDampingAttr().Set(1e4)    # Some damping
                        
                        print("[info] Set cube PhysX material: max friction, compliant contact")
                    except Exception as e:
                        print(f"[warn] Could not set PhysX material: {e}")
                    
                    # Configure rigid body contact properties
                    try:
                        rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(cube_prim)
                        # Contact offset - how close objects need to be to generate contacts
                        rigid_body.CreateContactOffsetAttr().Set(0.02)  # 2cm contact offset
                        # Rest offset - minimum maintained separation 
                        rigid_body.CreateRestOffsetAttr().Set(0.001)    # 1mm rest offset
                        # Enable CCD for this body
                        rigid_body.CreateEnableCCDAttr().Set(True)
                        
                        print("[info] Set cube rigid body: contact offset=0.02, CCD enabled")
                    except Exception as e:
                        print(f"[warn] Could not set rigid body properties: {e}")
                    
        except Exception as e:
            print(f"[warn] Could not increase cube friction: {e}")

    
    def _apply_domain_randomization(self):
        if self.domain_randomizer is not None:
            self.domain_randomizer.randomize()

    def _get_recent_collisions(self):
        try:
            physics = self.world.get_physics_context()
        except Exception:
            return None

        report = None
        for attr in ("get_contact_report", "consume_contact_report", "get_contact_reports"):
            getter = getattr(physics, attr, None)
            if callable(getter):
                try:
                    report = getter()
                except Exception:
                    report = None
                if report:
                    break
        if report is None:
            return None

        collisions = []

        if isinstance(report, dict):
            entries = report.get("contacts") or report.get("contactReports") or report.get("pairs") or []
        else:
            entries = report

        for entry in entries:
            prim0 = entry.get("body0") if isinstance(entry, dict) else None
            prim1 = entry.get("body1") if isinstance(entry, dict) else None
            if prim0 is None and isinstance(entry, dict):
                prim0 = entry.get("prim0") or entry.get("entity0")
            if prim1 is None and isinstance(entry, dict):
                prim1 = entry.get("prim1") or entry.get("entity1")
            if prim0 is None or prim1 is None:
                continue

            impulse = entry.get("impulse") if isinstance(entry, dict) else None
            collisions.append({
                "prim0": prim0,
                "prim1": prim1,
                "impulse": impulse,
            })

        return collisions

    def _validate_state(self, _obs):
        flags = {
            "joints_unavailable": False,
            "joint_limit_violation": False,
            "joint_nan": False,
            "vel_nan": False,
            "cube_pose_unavailable": False,
            "cube_below_ground": False,
            "cube_out_of_workspace": False,
            "cup_knocked_over": False,
            "cup_collision": False,
            "collision_report_unavailable": False,
        }
        issues = []

        joint_positions = self.reward_engine.latest_joint_positions
        joint_velocities = self.reward_engine.latest_joint_velocities

        if joint_positions is None:
            flags["joints_unavailable"] = True
            issues.append("Joint positions unavailable.")
        else:
            if not np.isfinite(joint_positions).all():
                flags["joint_nan"] = True
                issues.append("Joint positions contain non-finite values.")
            for idx, name in enumerate(self.robot.joint_names):
                lower, upper = self.robot.joint_limits[name]
                value = float(joint_positions[idx])
                tolerance = max(1e-2, 0.05 * (upper - lower))
                if value < lower - tolerance or value > upper + tolerance:
                    flags["joint_limit_violation"] = True
                    issues.append(
                        f"Joint '{name}' outside limits: {value:.3f} not in [{lower:.3f}, {upper:.3f}] (tol {tolerance:.3f})."
                    )
                    break

        if joint_velocities is not None and not np.isfinite(joint_velocities).all():
            flags["vel_nan"] = True
            issues.append("Joint velocities contain non-finite values.")

        state = getattr(self.reward_engine, "task_state", {})
        workspace_origin_xy = getattr(self, "_workspace_origin_xy", None)
        if workspace_origin_xy is None:
            workspace_origin_xy = np.zeros(2, dtype=float)
        else:
            workspace_origin_xy = np.asarray(workspace_origin_xy, dtype=float)
        cube_pos = state.get("cube_pos")
        if cube_pos is None:
            flags["cube_pose_unavailable"] = True
            issues.append("Cube pose unavailable.")
        else:
            if cube_pos[2] < -0.01:
                flags["cube_below_ground"] = True
                issues.append(f"Cube below ground plane: z={cube_pos[2]:.3f}.")
            relative_xy = cube_pos[:2] - workspace_origin_xy
            xy_radius = float(np.linalg.norm(relative_xy))
            max_radius = WORKSPACE_RADIUS_RANGE[1] + 0.35
            if xy_radius > max_radius:
                flags["cube_out_of_workspace"] = True
                issues.append(f"Cube XY radius {xy_radius:.3f} exceeds workspace limit {max_radius:.3f}.")

        cup_tilt = None
        if Gf is not None and UsdGeom is not None and Usd is not None:
            try:
                xformable = UsdGeom.Xformable(self.cup_xform.GetPrim())
                matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                up_vec = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
                up = np.array([up_vec[0], up_vec[1], up_vec[2]], dtype=float)
                up_norm = up / (np.linalg.norm(up) + 1e-6)
                cup_tilt = float(np.arccos(np.clip(np.dot(up_norm, np.array([0.0, 0.0, 1.0])), -1.0, 1.0)))
            except Exception:
                cup_tilt = None

        if cup_tilt is not None and cup_tilt > self._cup_upright_threshold_rad:
            flags["cup_knocked_over"] = True
            tilt_deg = np.degrees(cup_tilt)
            issues.append(f"Cup tilt {tilt_deg:.1f} deg exceeds threshold.")

        collisions = self._get_recent_collisions()
        if collisions is None:
            flags["collision_report_unavailable"] = True
            collisions = []
        else:
            cup_path = str(self.cup_xform.GetPath())
            cube_path = getattr(self.cube, "prim_path", "/World/Cube")
            robot_path = getattr(self.robot, "prim_path", "/World/Robot")
            for entry in collisions:
                prim0 = str(entry.get("prim0", ""))
                prim1 = str(entry.get("prim1", ""))
                pair = (prim0, prim1)
                if (cup_path in prim0) or (cup_path in prim1):
                    flags["cup_collision"] = True
                    if flags["cup_knocked_over"]:
                        self._termination_reason = "cup_knocked_over"
                    issues.append(f"Cup involved in collision: {pair}.")
                    break
                if ((cube_path in prim0 and robot_path in prim1) or
                        (cube_path in prim1 and robot_path in prim0)):
                    issues.append(f"Cube-robot collision detected: {pair}.")

        ok = not issues
        result = {
            "ok": ok,
            "issues": issues,
            "flags": flags,
            "collisions": collisions,
            "cup_tilt_rad": cup_tilt,
            "terminate_episode": self._force_terminate,
        }
        self.last_validation_result = result

        if not ok:
            print("[warn] state validation issues detected:", "; ".join(issues))

        if flags.get("cup_knocked_over"):
            self._force_terminate = True
            if not self._termination_reason:
                self._termination_reason = "cup_knocked_over"

    def _capture_images(self):
        ts = int(time.time() * 1000)
        try:
            top_rgba = self.top_camera.get_rgba()
        except Exception:
            top_rgba = None
        try:
            side_rgba = self.side_camera.get_rgba()
        except Exception:
            side_rgba = None
        try:
            wrist_cam = getattr(self.robot, "wrist_camera", None)
            wrist_rgba = wrist_cam.get_rgba() if wrist_cam is not None else None
        except Exception:
            wrist_rgba = None

        try:
            if top_rgba is not None:
                write_png(os.path.join(self.temp_dir, f"top_{ts}.png"), top_rgba)
            if side_rgba is not None:
                write_png(os.path.join(self.temp_dir, f"side_{ts}.png"), side_rgba)
            if wrist_rgba is not None:
                write_png(os.path.join(self.temp_dir, f"wrist_{ts}.png"), wrist_rgba)
        except Exception as exc:
            print(f"[warn] image save failed: {exc}")


def run_demo(max_steps=360):
    env = IsaacPickPlaceEnv(headless=False, capture_images=True, image_interval=3)
    env.reset()

    try:
        while _SIMULATION_APP.is_running() and env._step_counter < max_steps:
            if env._step_counter % 30 == 0:
                action = env.robot.get_random_joint_positions()
            else:
                action = env.robot_articulation.get_joint_positions()

            env.step(action, render=True)
    finally:
        env.shutdown()


if __name__ == "__main__":
    run_demo()

import os
import shutil
import time

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
        from pxr import Gf as _Gf, UsdGeom as _UsdGeom, UsdPhysics as _UsdPhysics
        from robot import SO100Robot as _SO100Robot

        World = _World
        DynamicCuboid = _DynamicCuboid
        get_context = _get_context
        Camera = _Camera
        Gf = _Gf
        UsdGeom = _UsdGeom
        UsdPhysics = _UsdPhysics
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

        self.cube_scale = np.array([0.1, 0.1, 0.1], dtype=float)
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

        self.reward_engine = RewardEngine(self)
        self.domain_randomizer = DomainRandomizer(self)
        self.last_validation_result = {"ok": True, "issues": [], "flags": {}}

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

        urdf_path = os.path.join(self.current_dir, "so100.urdf")
        self.robot = SO100Robot(self.world, urdf_path)
        self.robot_articulation = self.robot.get_robot()

        cube_xy, cup_xy = self._sample_object_positions()
        cube_position = np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0])

        self.cube = DynamicCuboid(
            prim_path="/World/Cube",
            name="collision_cube",
            position=cube_position,
            scale=self.cube_scale,
            color=np.array([0.0, 0.5, 1.0]),
        )
        self.world.scene.add(self.cube)

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
            resolution=(640, 480),
        )
        self.side_camera = Camera(
            prim_path="/World/side_camera",
            name="side_camera",
            position=np.array([1.0, 0, 0.5]),
            orientation=np.array([0, 0, 0.707, 0.707]),
            frequency=30,
            resolution=(640, 480),
        )

        self.world.scene.add(self.top_camera)
        self.world.scene.add(self.side_camera)

        self.world.reset()

        top_cam_pos = np.array([0, -0.75, 8.0])
        top_cam_orient = np.array([-np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25), np.sqrt(0.25)])
        side_cam_pos = np.array([6.0, -0.5, 0.5])
        side_cam_orient = np.array([0, 0, 0, 1])

        self.top_camera.set_world_pose(position=top_cam_pos, orientation=top_cam_orient)
        self.side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)

        self.robot.create_wrist_camera()
        self.robot.update_wrist_camera_position(verbose=False)

        self.top_camera.initialize()
        self.side_camera.initialize()
        if getattr(self.robot, "wrist_camera", None) is not None:
            self.robot.wrist_camera.initialize()

        for _ in range(5):
            self.world.step(render=not self.headless)

        self._cube_xy = cube_xy
        self._cup_xy = cup_xy

        self._apply_domain_randomization()

    def reset(self):
        """Reset the scene with new randomized object placements."""
        self._step_counter = 0

        cube_xy, cup_xy = self._sample_object_positions()
        self._cube_xy = cube_xy
        self._cup_xy = cup_xy

        cube_position = np.array([cube_xy[0], cube_xy[1], self.cube_scale[2] / 2.0])
        self.cube.set_world_pose(position=cube_position, orientation=np.array([0, 0, 0, 1]))

        cup_translation = Gf.Vec3d(float(cup_xy[0]), float(cup_xy[1]), 0.0)
        UsdGeom.XformCommonAPI(self.cup_xform).SetTranslate(cup_translation)

        self.world.reset()
        self.robot.update_wrist_camera_position(verbose=False)

        for _ in range(5):
            self.world.step(render=not self.headless)

        self._apply_domain_randomization()
        self.reward_engine.reset()
        self.last_validation_result = {"ok": True, "issues": [], "flags": {}}

        return self._get_observation()

    def step(self, action, render=None):
        """Advance the simulation by one control step."""
        render = not self.headless if render is None else bool(render)

        joint_targets = self._clip_action(action)
        self.robot.set_joint_positions(joint_targets)

        self.world.step(render=render)
        self._step_counter += 1

        obs = self._get_observation()
        self.reward_engine.compute_reward_components()
        self._validate_state(obs)
        reward, done, info = self.reward_engine.summarize_reward()

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

    # ------------------------------------------------------------------
    # Helpers

    def _sample_object_positions(self):
        cube_xy = sample_workspace_xy(self.rng)
        cup_xy = sample_workspace_xy(self.rng, existing=[cube_xy], min_separation=CUP_CUBE_MIN_DISTANCE)
        return cube_xy, cup_xy

    def _clip_action(self, action):
        action = np.array(action, dtype=float)
        lower = np.array([self.robot.joint_limits[name][0] for name in self.robot.joint_names], dtype=float)
        upper = np.array([self.robot.joint_limits[name][1] for name in self.robot.joint_names], dtype=float)
        if action.shape != lower.shape:
            raise ValueError(f"Expected action of shape {lower.shape}, received {action.shape}")
        return np.clip(action, lower, upper)

    def _get_observation(self):
        camera_frames = self._gather_sensor_observations()
        joint_positions = self.robot_articulation.get_joint_positions()
        joint_velocities = self.robot_articulation.get_joint_velocities()
        joint_positions = joint_positions.astype(np.float32, copy=True)
        joint_velocities = joint_velocities.astype(np.float32, copy=True)

        self.reward_engine.record_joint_state(joint_positions.copy(), joint_velocities.copy())

        obs = {
            "joint_positions": joint_positions.copy(),
            "joint_velocities": joint_velocities.copy(),
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
            width, height = 640, 480

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

    def _apply_domain_randomization(self):
        if self.domain_randomizer is not None:
            self.domain_randomizer.randomize()

    def _validate_state(self, _obs):
        flags = {
            "joints_unavailable": False,
            "joint_limit_violation": False,
            "joint_nan": False,
            "vel_nan": False,
            "cube_pose_unavailable": False,
            "cube_below_ground": False,
            "cube_out_of_workspace": False,
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
                if value < lower - 1e-3 or value > upper + 1e-3:
                    flags["joint_limit_violation"] = True
                    issues.append(f"Joint '{name}' outside limits: {value:.3f} not in [{lower:.3f}, {upper:.3f}].")
                    break

        if joint_velocities is not None and not np.isfinite(joint_velocities).all():
            flags["vel_nan"] = True
            issues.append("Joint velocities contain non-finite values.")

        state = getattr(self.reward_engine, "task_state", {})
        cube_pos = state.get("cube_pos")
        if cube_pos is None:
            flags["cube_pose_unavailable"] = True
            issues.append("Cube pose unavailable.")
        else:
            if cube_pos[2] < -0.01:
                flags["cube_below_ground"] = True
                issues.append(f"Cube below ground plane: z={cube_pos[2]:.3f}.")
            xy_radius = float(np.linalg.norm(cube_pos[:2]))
            max_radius = WORKSPACE_RADIUS_RANGE[1] + 0.15
            if xy_radius > max_radius:
                flags["cube_out_of_workspace"] = True
                issues.append(f"Cube XY radius {xy_radius:.3f} exceeds workspace limit {max_radius:.3f}.")

        ok = not issues
        result = {
            "ok": ok,
            "issues": issues,
            "flags": flags,
        }
        self.last_validation_result = result

        if not ok:
            print("[warn] state validation issues detected:", "; ".join(issues))

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

import os
import shutil
import time

import numpy as np
from isaacsim import SimulationApp

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
        _SIM_HEADLESS_FLAG = headless
    elif headless != _SIM_HEADLESS_FLAG:
        print(f"[warn] SimulationApp already initialized with headless={_SIM_HEADLESS_FLAG}; "
              f"requested headless={headless} ignored.")

    return _SIMULATION_APP


WORKSPACE_RADIUS_RANGE = (0.35, 0.65)
WORKSPACE_ANGLE_RANGE = (-np.pi / 2, np.pi / 2)
MIN_OBJECT_SEPARATION = 0.18
CUP_CUBE_MIN_DISTANCE = 0.28

WORKSPACE_FORWARD_AXIS = np.array([0.0, -1.0], dtype=float)
WORKSPACE_FORWARD_AXIS /= np.linalg.norm(WORKSPACE_FORWARD_AXIS)
WORKSPACE_RIGHT_AXIS = np.array([WORKSPACE_FORWARD_AXIS[1], -WORKSPACE_FORWARD_AXIS[0]], dtype=float)
WORKSPACE_RIGHT_AXIS /= np.linalg.norm(WORKSPACE_RIGHT_AXIS)


def sample_workspace_xy(rng, existing=None, min_separation=None, max_attempts=32):
    """Sample an (x, y) point inside the forward-facing semi-circle workspace."""
    if existing is None:
        existing = []
    if min_separation is None:
        min_separation = MIN_OBJECT_SEPARATION

    candidate = None
    for _ in range(max_attempts):
        radius = rng.uniform(*WORKSPACE_RADIUS_RANGE)
        angle = rng.uniform(*WORKSPACE_ANGLE_RANGE)
        forward_component = radius * np.cos(angle)
        right_component = radius * np.sin(angle)
        candidate = forward_component * WORKSPACE_FORWARD_AXIS + right_component * WORKSPACE_RIGHT_AXIS
        if all(np.linalg.norm(candidate - other) >= min_separation for other in existing):
            return candidate

    if candidate is not None:
        return candidate

    fallback = WORKSPACE_FORWARD_AXIS * WORKSPACE_RADIUS_RANGE[0]
    if existing:
        for shift_sign in (1, -1):
            offset = fallback + shift_sign * min_separation * WORKSPACE_RIGHT_AXIS
            if all(np.linalg.norm(offset - other) >= min_separation for other in existing):
                return offset
    return fallback


def build_cup_mesh(outer_radius_top, outer_radius_bottom, height,
                   inner_radius_top, inner_radius_bottom,
                   bottom_thickness, segments=32):
    """Create mesh data for a hollow tapered cup with a flat bottom."""
    bottom_thickness = min(bottom_thickness, height * 0.4)

    points = []
    face_counts = []
    face_indices = []

    def angle(i):
        return (2.0 * np.pi * i) / segments

    def add_point(x, y, z):
        points.append(Gf.Vec3f(float(x), float(y), float(z)))

    for i in range(segments):
        ang = angle(i)
        add_point(outer_radius_bottom * np.cos(ang), outer_radius_bottom * np.sin(ang), 0.0)
    outer_top_offset = len(points)
    for i in range(segments):
        ang = angle(i)
        add_point(outer_radius_top * np.cos(ang), outer_radius_top * np.sin(ang), height)
    inner_top_offset = len(points)
    for i in range(segments):
        ang = angle(i)
        add_point(inner_radius_top * np.cos(ang), inner_radius_top * np.sin(ang), height)
    inner_bottom_offset = len(points)
    for i in range(segments):
        ang = angle(i)
        add_point(inner_radius_bottom * np.cos(ang), inner_radius_bottom * np.sin(ang), bottom_thickness)

    bottom_center_top_idx = len(points)
    add_point(0.0, 0.0, bottom_thickness)
    bottom_center_bottom_idx = len(points)
    add_point(0.0, 0.0, 0.0)

    def add_triangle(a, b, c):
        face_counts.append(3)
        face_indices.extend([a, b, c])

    segments_range = range(segments)
    for i in segments_range:
        next_i = (i + 1) % segments
        ob_i = i
        ob_next = next_i
        ot_i = outer_top_offset + i
        ot_next = outer_top_offset + next_i
        add_triangle(ob_i, ob_next, ot_i)
        add_triangle(ot_i, ob_next, ot_next)

    for i in segments_range:
        next_i = (i + 1) % segments
        ib_i = inner_bottom_offset + i
        ib_next = inner_bottom_offset + next_i
        it_i = inner_top_offset + i
        it_next = inner_top_offset + next_i
        add_triangle(ib_i, it_i, ib_next)
        add_triangle(it_i, it_next, ib_next)

    for i in segments_range:
        next_i = (i + 1) % segments
        ob_i = i
        ob_next = next_i
        ib_i = inner_bottom_offset + i
        ib_next = inner_bottom_offset + next_i
        add_triangle(ob_i, ib_i, ob_next)
        add_triangle(ob_next, ib_i, ib_next)

    for i in segments_range:
        next_i = (i + 1) % segments
        ib_i = inner_bottom_offset + i
        ib_next = inner_bottom_offset + next_i
        add_triangle(ib_i, ib_next, bottom_center_top_idx)

    for i in segments_range:
        next_i = (i + 1) % segments
        ob_i = i
        ob_next = next_i
        add_triangle(ob_i, bottom_center_bottom_idx, ob_next)

    return points, face_counts, face_indices


def create_cup_prim(stage, prim_path, position,
                    outer_radius_top, outer_radius_bottom,
                    inner_radius_top, inner_radius_bottom,
                    height, bottom_thickness, color, mass):
    """Create a hollow cup mesh prim with collision enabled and return its Xform."""
    xform = UsdGeom.Xform.Define(stage, prim_path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    mesh_path = f"{prim_path}/CupMesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    points, counts, indices = build_cup_mesh(
        outer_radius_top, outer_radius_bottom, height,
        inner_radius_top, inner_radius_bottom, bottom_thickness
    )
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexDecomposition")
    xform_prim = xform.GetPrim()
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(xform_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    rigid_api.CreateKinematicEnabledAttr(False)
    mass_api = UsdPhysics.MassAPI.Apply(xform_prim)
    mass_attr = mass_api.GetMassAttr()
    if not mass_attr:
        mass_attr = mass_api.CreateMassAttr()
    mass_attr.Set(float(mass))
    return xform


def write_png(path, rgba_uint8):
    """Serialize an RGBA frame to disk, dropping alpha if present."""
    import imageio.v2 as imageio

    if rgba_uint8 is None:
        return False

    rgb = rgba_uint8[..., :3] if rgba_uint8.shape[-1] == 4 else rgba_uint8
    imageio.imwrite(path, rgb)
    return True


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

        if self.capture_images:
            self._reset_temp_dir()

        self._build_scene()

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

        return self._get_observation()

    def step(self, action, render=None):
        """Advance the simulation by one control step."""
        render = not self.headless if render is None else bool(render)

        joint_targets = self._clip_action(action)
        self.robot.set_joint_positions(joint_targets)

        self.world.step(render=render)
        self._step_counter += 1

        obs = self._get_observation()
        reward, done, info = self._compute_reward(obs)

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
        joint_positions = self.robot_articulation.get_joint_positions()
        joint_velocities = self.robot_articulation.get_joint_velocities()
        obs = {
            "joint_positions": joint_positions.copy(),
            "joint_velocities": joint_velocities.copy(),
        }
        return obs

    def _compute_reward(self, _obs):
        # Placeholder to be filled with task-specific shaping.
        return 0.0, False, {}

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

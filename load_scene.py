import os, shutil, time
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "load_stage_on_start": False
})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.usd import get_context
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom, UsdPhysics
from robot import SO100Robot

current_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(current_dir, "temp")
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir, exist_ok=True)

get_context().new_stage()
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

urdf_path = os.path.join(current_dir, "so100.urdf")
robot = SO100Robot(world, urdf_path)

_rng = np.random.default_rng()
WORKSPACE_RADIUS_RANGE = (0.35, 0.65)
WORKSPACE_ANGLE_RANGE = (-np.pi / 2, np.pi / 2)
MIN_OBJECT_SEPARATION = 0.18
CUP_CUBE_MIN_DISTANCE = 0.28
WORKSPACE_FORWARD_AXIS = np.array([0.0, -1.0], dtype=float)
WORKSPACE_FORWARD_AXIS /= np.linalg.norm(WORKSPACE_FORWARD_AXIS)
WORKSPACE_RIGHT_AXIS = np.array([WORKSPACE_FORWARD_AXIS[1], -WORKSPACE_FORWARD_AXIS[0]], dtype=float)
WORKSPACE_RIGHT_AXIS /= np.linalg.norm(WORKSPACE_RIGHT_AXIS)


def sample_workspace_xy(existing=None, min_separation=None, max_attempts=32):
    """Sample an (x, y) point inside the forward-facing semi-circle workspace."""
    if existing is None:
        existing = []
    if min_separation is None:
        min_separation = MIN_OBJECT_SEPARATION

    candidate = None
    for _ in range(max_attempts):
        radius = _rng.uniform(*WORKSPACE_RADIUS_RANGE)
        angle = _rng.uniform(*WORKSPACE_ANGLE_RANGE)
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


cube_scale = np.array([0.1, 0.1, 0.1])
cube_xy = sample_workspace_xy()
cube_position = np.concatenate([cube_xy, [cube_scale[2] / 2.0]])

cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="collision_cube",
    position=cube_position,
    scale=cube_scale,
    color=np.array([0.0, 0.5, 1.0])
)
world.scene.add(cube)

cup_xy = sample_workspace_xy(existing=[cube_xy], min_separation=CUP_CUBE_MIN_DISTANCE)
cup_height = 0.18
cup_outer_radius_top = 0.11
cup_outer_radius_bottom = 0.085
cup_wall_thickness = 0.012
cup_inner_radius_top = max(cup_outer_radius_top - cup_wall_thickness, cup_outer_radius_top * 0.55)
cup_inner_radius_bottom = max(cup_outer_radius_bottom - cup_wall_thickness, cup_outer_radius_bottom * 0.55)
cup_bottom_thickness = max(0.008, cup_wall_thickness * 0.75)
cup_color = (0.8, 0.3, 0.2)
cup_mass = 0.25


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
    """Create a hollow cup mesh prim with collision enabled."""
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

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


cup_translation = np.array([cup_xy[0], cup_xy[1], 0.0])
create_cup_prim(
    get_context().get_stage(),
    prim_path="/World/Cup",
    position=cup_translation,
    outer_radius_top=cup_outer_radius_top,
    outer_radius_bottom=cup_outer_radius_bottom,
    inner_radius_top=cup_inner_radius_top,
    inner_radius_bottom=cup_inner_radius_bottom,
    height=cup_height,
    bottom_thickness=cup_bottom_thickness,
    color=cup_color,
    mass=cup_mass,
)

top_camera = Camera(
    prim_path="/World/top_camera",
    name="top_camera",
    position=np.array([0, 0, 1.5]),
    orientation=np.array([0, 0, 0, 1]),
    frequency=30,
    resolution=(640, 480)
)

side_camera = Camera(
    prim_path="/World/side_camera",
    name="side_camera",
    position=np.array([1.0, 0, 0.5]),
    orientation=np.array([0, 0, 0.707, 0.707]),
    frequency=30,
    resolution=(640, 480)
)

world.scene.add(top_camera)
world.scene.add(side_camera)

world.reset()

top_cam_pos = np.array([0, -0.75, 8.0])
top_cam_orient = np.array([-np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25),  np.sqrt(0.25)])
side_cam_pos = np.array([6.0, -0.5, 0.5])
side_cam_orient = np.array([0, 0, 0, 1])

top_camera.set_world_pose(position=top_cam_pos,  orientation=top_cam_orient)
side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)

robot.create_wrist_camera()
robot.update_wrist_camera_position(verbose=False)

top_camera.initialize()
side_camera.initialize()
if hasattr(robot, "wrist_camera") and robot.wrist_camera is not None:
    robot.wrist_camera.initialize()

for _ in range(5):
    world.step(render=True)

def write_png(path, rgba_uint8):
    # Lazy import to keep top clean
    import imageio.v2 as imageio
    if rgba_uint8 is None:
        return False
    # Drop alpha if present
    if rgba_uint8.shape[-1] == 4:
        rgb = rgba_uint8[..., :3]
    else:
        rgb = rgba_uint8
    imageio.imwrite(path, rgb)
    return True

step_count = 0
while simulation_app.is_running():
    world.step(render=True)
    step_count += 1

    if step_count % 30 == 0:
        robot.set_joint_positions(robot.get_random_joint_positions())

    if step_count % 3 == 0:
        ts = int(time.time() * 1000)

        try:
            top_rgba   = top_camera.get_rgba()
        except Exception:
            top_rgba = None
        try:
            side_rgba  = side_camera.get_rgba()
        except Exception:
            side_rgba = None
        try:
            wrist_cam = getattr(robot, "wrist_camera", None)
            wrist_rgba = wrist_cam.get_rgba() if wrist_cam is not None else None
        except Exception:
            wrist_rgba = None

        try:
            if top_rgba is not None:
                write_png(os.path.join(temp_dir, f"top_{ts}.png"), top_rgba)
            if side_rgba is not None:
                write_png(os.path.join(temp_dir, f"side_{ts}.png"), side_rgba)
            if wrist_rgba is not None:
                write_png(os.path.join(temp_dir, f"wrist_{ts}.png"), wrist_rgba)
        except Exception as e:
            print(f"[warn] image save failed: {e}")

simulation_app.close()

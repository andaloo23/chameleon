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

cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="collision_cube",
    position=np.array([0.5, -0.5, 0.0]),
    scale=np.array([0.1, 0.1, 0.1]),
    color=np.array([0.0, 0.5, 1.0])
)
world.scene.add(cube)

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
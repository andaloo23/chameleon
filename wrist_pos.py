import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False, 
    "load_stage_on_start": False 
})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.robots import Robot
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
from omni.usd import get_context
import numpy as np
from omni.isaac.sensor import Camera
from pxr import UsdGeom, Usd
from scipy.spatial.transform import Rotation as R
import threading
import sys

get_context().new_stage()

current_dir = os.path.dirname(os.path.abspath(__file__))
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

urdf_interface = _urdf.acquire_urdf_interface()
import_config = _urdf.ImportConfig()
import_config.convex_decomp = False
import_config.fix_base = True
import_config.make_default_prim = True
import_config.self_collision = False
import_config.distance_scale = 2.5
import_config.density = 0.0

urdf_path = os.path.join(current_dir, "so100.urdf")

result, robot_model = omni.kit.commands.execute(
    "URDFParseFile",
    urdf_path=urdf_path,
    import_config=import_config
)

result, prim_path = omni.kit.commands.execute(
    "URDFImportRobot",
    urdf_robot=robot_model,
    import_config=import_config,
)

print(f"Robot imported at path: {prim_path}")

robot = Robot(prim_path=prim_path, name="so_arm100")
world.scene.add(robot)

cube1 = DynamicCuboid(
    prim_path="/World/Cube1",
    name="red_cube",
    position=np.array([0.7, 0.4, 0.3]),
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([1.0, 0.0, 0.0])
)

cube2 = DynamicCuboid(
    prim_path="/World/Cube2",
    name="blue_cube",
    position=np.array([0.7, -0.4, 0.3]),
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([0.0, 0.5, 1.0])
)

world.scene.add(cube1)
world.scene.add(cube2)

wrist_camera = Camera(
    prim_path="/World/wrist_camera_standalone",
    name="wrist_camera",
    frequency=30,
    resolution=(640, 480)
)
world.scene.add(wrist_camera)

world.reset()

print("\n" + "="*70)
print("TERMINAL COMMAND CAMERA CONTROL")
print("="*70)
print("Type commands in this terminal to adjust camera:")
print("")
print("  Position: w/s (fwd/back), a/d (left/right), q/e (up/down)")
print("  Rotation: i/k (pitch up/down), j/l (yaw left/right), u/o (roll)")
print("  Presets:  1-8 (load preset)")
print("  Utility:  r (reset), p (print config), h (help)")
print("")
print("Example: Type 'k' and press Enter to look down more")
print("="*70)
print("\nSwitch Isaac Sim viewport to 'wrist_camera' to see the view!")
print("="*70 + "\n")

def get_link_world_pose(link_name):
    """Get world position and orientation of a robot link"""
    stage = get_context().get_stage()
    link_path = f"{prim_path}/{link_name}"
    prim = stage.GetPrimAtPath(link_path)
    
    if prim:
        xform = UsdGeom.Xformable(prim)
        transform_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        translation = transform_matrix.ExtractTranslation()
        position = np.array([translation[0], translation[1], translation[2]])
        
        rotation = transform_matrix.ExtractRotationQuat()
        orientation = np.array([rotation.GetImaginary()[0], 
                               rotation.GetImaginary()[1], 
                               rotation.GetImaginary()[2], 
                               rotation.GetReal()])
        
        return position, orientation
    
    return None, None

def update_camera_from_wrist(offset_x, offset_y, offset_z, rot_pitch, rot_yaw, rot_roll):
    """Update camera position relative to wrist link"""
    wrist_pos, wrist_orient = get_link_world_pose("wrist")
    
    if wrist_pos is None:
        return None, None
    
    wrist_rot = R.from_quat([wrist_orient[0], wrist_orient[1], wrist_orient[2], wrist_orient[3]])
    wrist_matrix = wrist_rot.as_matrix()
    
    local_offset = np.array([offset_x, offset_y, offset_z])
    world_offset = wrist_matrix @ local_offset
    cam_pos = wrist_pos + world_offset
    
    cam_rot = R.from_euler('xyz', [rot_pitch, rot_yaw, rot_roll], degrees=True)
    cam_quat = cam_rot.as_quat()
    
    cam_orient = np.array([cam_quat[0], cam_quat[1], cam_quat[2], cam_quat[3]])
    
    wrist_camera.set_world_pose(position=cam_pos, orientation=cam_orient)
    
    return cam_pos, cam_orient

cam_offset_x = 0.0
cam_offset_y = -0.25
cam_offset_z = 0.15
cam_rot_pitch = -60.0
cam_rot_yaw = 0.0
cam_rot_roll = 0.0

move_step = 0.02
rotate_step = 5.0

presets = [
    (0.0, -0.25, 0.15, -70, 0, 0),
    (0.0, -0.25, 0.15, -60, 0, 0),
    (0.0, -0.25, 0.15, -50, 0, 0),
    (0.0, -0.20, 0.25, -80, 0, 0),
    (0.0, -0.15, 0.08, -65, 0, 0),
    (0.0, -0.35, 0.18, -55, 0, 0),
    (0.15, -0.15, 0.10, -60, -30, 0),
    (0.0, -0.20, 0.20, -85, 0, 0),
]

def print_config():
    """Print current camera configuration"""
    print(f"\n{'='*70}")
    print("CURRENT CAMERA CONFIGURATION:")
    print(f"  Position offset: X={cam_offset_x:.3f}, Y={cam_offset_y:.3f}, Z={cam_offset_z:.3f}")
    print(f"  Rotation: Pitch={cam_rot_pitch:.1f}°, Yaw={cam_rot_yaw:.1f}°, Roll={cam_rot_roll:.1f}°")
    print(f"\nURDF format for wrist_camera_joint:")
    print(f'  <origin xyz="{cam_offset_x:.3f} {cam_offset_y:.3f} {cam_offset_z:.3f}"')
    print(f'          rpy="{np.radians(cam_rot_roll):.4f} {np.radians(cam_rot_pitch):.4f} {np.radians(cam_rot_yaw):.4f}"/>')
    print(f"{'='*70}\n")
    print("Enter command: ", end='', flush=True)

def print_help():
    """Print help message"""
    print("\nCOMMANDS:")
    print("  w - Forward    s - Backward")
    print("  a - Left       d - Right")
    print("  q - Up         e - Down")
    print("  i - Pitch up   k - Pitch down")
    print("  j - Yaw left   l - Yaw right")
    print("  u - Roll left  o - Roll right")
    print("  1-8 - Load preset config")
    print("  r - Reset      p - Print config")
    print("  h - Help\n")
    print("Enter command: ", end='', flush=True)

command_queue = []
command_lock = threading.Lock()

def input_thread():
    """Background thread to read user input"""
    print("Enter command: ", end='', flush=True)
    while simulation_app.is_running():
        try:
            cmd = sys.stdin.readline().strip().lower()
            if cmd:
                with command_lock:
                    command_queue.append(cmd)
        except:
            break

input_thread_obj = threading.Thread(target=input_thread, daemon=True)
input_thread_obj.start()

step_count = 0
joint_names = robot.dof_names or ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

rotation_angle = 0.0
rotation_direction = 1

print_config()

while simulation_app.is_running():
    world.step(render=True)
    
    with command_lock:
        commands_to_process = command_queue.copy()
        command_queue.clear()
    
    for cmd in commands_to_process:
        if cmd == 'w':
            cam_offset_y += move_step
            print(f"Forward: Y={cam_offset_y:.3f}")
            print("Enter command: ", end='', flush=True)
        elif cmd == 's':
            cam_offset_y -= move_step
            print(f"Backward: Y={cam_offset_y:.3f}")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'a':
            cam_offset_x -= move_step
            print(f"Left: X={cam_offset_x:.3f}")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'd':
            cam_offset_x += move_step
            print(f"Right: X={cam_offset_x:.3f}")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'q':
            cam_offset_z += move_step
            print(f"Up: Z={cam_offset_z:.3f}")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'e':
            cam_offset_z -= move_step
            print(f"Down: Z={cam_offset_z:.3f}")
            print("Enter command: ", end='', flush=True)
        
        elif cmd == 'i':
            cam_rot_pitch += rotate_step
            print(f"Look up: Pitch={cam_rot_pitch:.1f}°")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'k':
            cam_rot_pitch -= rotate_step
            print(f"Look down: Pitch={cam_rot_pitch:.1f}°")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'j':
            cam_rot_yaw += rotate_step
            print(f"Look left: Yaw={cam_rot_yaw:.1f}°")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'l':
            cam_rot_yaw -= rotate_step
            print(f"Look right: Yaw={cam_rot_yaw:.1f}°")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'u':
            cam_rot_roll += rotate_step
            print(f"Roll left: Roll={cam_rot_roll:.1f}°")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'o':
            cam_rot_roll -= rotate_step
            print(f"Roll right: Roll={cam_rot_roll:.1f}°")
            print("Enter command: ", end='', flush=True)
        
        elif cmd == 'r':
            cam_offset_x, cam_offset_y, cam_offset_z = 0.0, -0.25, 0.15
            cam_rot_pitch, cam_rot_yaw, cam_rot_roll = -60.0, 0.0, 0.0
            print("Reset to default")
            print("Enter command: ", end='', flush=True)
        elif cmd == 'p':
            print_config()
        elif cmd == 'h':
            print_help()
        
        elif cmd in ['1', '2', '3', '4', '5', '6', '7', '8']:
            preset_idx = int(cmd) - 1
            preset = presets[preset_idx]
            cam_offset_x, cam_offset_y, cam_offset_z = preset[0], preset[1], preset[2]
            cam_rot_pitch, cam_rot_yaw, cam_rot_roll = preset[3], preset[4], preset[5]
            print(f"Loaded preset {cmd}")
            print("Enter command: ", end='', flush=True)
    
    update_camera_from_wrist(cam_offset_x, cam_offset_y, cam_offset_z, 
                            cam_rot_pitch, cam_rot_yaw, cam_rot_roll)
    
    if step_count % 60 == 0:
        rotation_angle += rotation_direction * 0.1
        
        if rotation_angle >= 1.57:
            rotation_direction = -1
        elif rotation_angle <= -1.57:
            rotation_direction = 1
        
        try:
            joint_positions = np.zeros(len(joint_names))
            if "shoulder_pan" in joint_names:
                shoulder_pan_index = joint_names.index("shoulder_pan")
                joint_positions[shoulder_pan_index] = rotation_angle
                robot.set_joint_positions(joint_positions)
        except Exception as e:
            pass
    
    step_count += 1

simulation_app.close()
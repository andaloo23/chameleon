import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False, 
    "load_stage_on_start": False 
})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
from omni.usd import get_context
import numpy as np
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
import omni.kit.viewport.utility as viewport_utils

get_context().new_stage()

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

world = World(stage_units_in_meters=1.0)

world.scene.add_default_ground_plane()

urdf_interface = _urdf.acquire_urdf_interface()

import_config = _urdf.ImportConfig()
import_config.convex_decomp = False
import_config.fix_base = True
import_config.make_default_prim = True
import_config.self_collision = False
import_config.distance_scale = 2.5  # scale robot by 2.5x
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

cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="collision_cube",
    position=np.array([0.5, -0.5, 0]),
    scale=np.array([0.1, 0.1, 0.1]), # 10 cm cube
    color=np.array([0.0, 0.5, 1.0]) # blue
)

world.scene.add(cube)

print(f"Cube added to scene at position: {cube.get_world_pose()[0]}")

wrist_camera = Camera(
    prim_path="/World/wrist_camera",
    name="wrist_camera",
    position=np.array([0.3, 0.3, 0.3]),
    orientation=np.array([0.707, 0, 0, 0.707]),
    frequency=30,
    resolution=(640, 480)
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

world.scene.add(wrist_camera)
world.scene.add(top_camera) 
world.scene.add(side_camera)

print("Cameras added to scene:")
print(f"- Wrist camera: {wrist_camera.prim_path}")
print(f"- Top camera: {top_camera.prim_path}")
print(f"- Side camera: {side_camera.prim_path}")

world.reset()

def look_at_quaternion(camera_pos, target_pos, up_vector=np.array([0, 0, 1])):
    """Calculate quaternion to look from camera_pos to target_pos"""

    forward = target_pos - camera_pos
    forward_norm = np.linalg.norm(forward)
    
    if forward_norm < 1e-6:
        return np.array([1, 0, 0, 0])
    
    forward = forward / forward_norm
    
    if abs(np.dot(forward, up_vector)) > 0.99:
        if abs(forward[0]) < 0.9:
            up_vector = np.array([1, 0, 0])
        else:
            up_vector = np.array([0, 1, 0])
    
    right = np.cross(forward, up_vector)
    right_norm = np.linalg.norm(right)
    
    if right_norm < 1e-6:
        return np.array([1, 0, 0, 0])
    
    right = right / right_norm
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    rotation_matrix = np.array([
        [right[0], up[0], -forward[0]],
        [right[1], up[1], -forward[1]],
        [right[2], up[2], -forward[2]]
    ])
    
    trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s
    
    quat = np.array([w, x, y, z])
    quat_norm = np.linalg.norm(quat)
    
    if quat_norm < 1e-6:
        return np.array([1, 0, 0, 0])
    
    return quat / quat_norm

robot_pos = np.array([0, 0, 0.3])

def apply_roll_rotation(quat, roll_degrees):
    """Apply a roll rotation (around Z-axis) to an existing quaternion"""

    roll_radians = np.radians(roll_degrees)
    roll_quat = np.array([np.cos(roll_radians/2), 0, 0, np.sin(roll_radians/2)])
    
    w1, x1, y1, z1 = quat
    w2, x2, y2, z2 = roll_quat
    
    result = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])
    
    return result / np.linalg.norm(result)

wrist_cam_pos = np.array([0.8, 0.8, 0.6])
wrist_cam_orient_base = look_at_quaternion(wrist_cam_pos, robot_pos)
wrist_cam_orient = apply_roll_rotation(wrist_cam_orient_base, 90)

top_cam_pos = np.array([0, 0, 4.0])  
top_cam_orient = look_at_quaternion(top_cam_pos, robot_pos)

side_cam_pos = np.array([2.0, 0, 1.0])
side_cam_orient_base = look_at_quaternion(side_cam_pos, robot_pos)
side_cam_orient = apply_roll_rotation(side_cam_orient_base, 90)

print("Calculated camera orientations:")
print(f"- Wrist camera quaternion: {wrist_cam_orient}")
print(f"- Top camera quaternion: {top_cam_orient}")
print(f"- Side camera quaternion: {side_cam_orient}")

try:
    wrist_camera.set_world_pose(position=wrist_cam_pos, orientation=wrist_cam_orient)
    print("Wrist camera pose set successfully")
except Exception as e:
    print(f"Error setting wrist camera pose: {e}")

try:
    top_camera.set_world_pose(position=top_cam_pos, orientation=top_cam_orient)
    print("Top camera pose set successfully")
except Exception as e:
    print(f"Error setting top camera pose: {e}")

try:
    side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)
    print("Side camera pose set successfully")
except Exception as e:
    print(f"Error setting side camera pose: {e}")

print("Updated camera positions and orientations to look at robot:")
print(f"- Wrist camera: pos={wrist_cam_pos}, looking at robot")
print(f"- Top camera: pos={top_cam_pos}, looking down at robot") 
print(f"- Side camera: pos={side_cam_pos}, looking at robot from side")

joint_names = robot.dof_names
print(f"Robot joints: {joint_names}")
print(f"Number of DOFs: {robot.num_dof}")

if joint_names is None:
    print("Warning: joint_names is None, trying alternative method...")
    try:
        articulation = robot.get_articulation_controller()
        if articulation:
            joint_names = articulation.get_dof_names()
            print(f"Alternative joint names: {joint_names}")
    except Exception as e:
        print(f"Error getting joint names: {e}")
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        print(f"Using fallback joint names: {joint_names}")

step_count = 0
rotation_angle = 0.0
rotation_direction = 1
camera_switch_counter = 0
current_camera = 0

viewport_api = viewport_utils.get_active_viewport()

while simulation_app.is_running():
    world.step(render=True)
    
    if step_count % 60 == 0: 
        rotation_angle += rotation_direction * 0.1 
        
        if rotation_angle >= 1.57:
            rotation_direction = -1
        elif rotation_angle <= -1.57:
            rotation_direction = 1
        
        try:
            if joint_names is not None and len(joint_names) > 0:
                joint_positions = np.zeros(len(joint_names))
                if "shoulder_pan" in joint_names:
                    shoulder_pan_index = joint_names.index("shoulder_pan")
                    joint_positions[shoulder_pan_index] = rotation_angle
                    
                    robot.set_joint_positions(joint_positions)
                    
                    print(f"Step {step_count}: Shoulder rotation: {rotation_angle:.2f} radians ({np.degrees(rotation_angle):.1f} degrees)")
                else:
                    print(f"Warning: 'shoulder_pan' not found in joint names: {joint_names}")
            else:
                print(f"Warning: No valid joint names available. joint_names = {joint_names}")
        except Exception as e:
            print(f"Error controlling robot: {e}")
            print(f"joint_names type: {type(joint_names)}, value: {joint_names}")
    
    if step_count % 300 == 0 and step_count > 0:
        camera_switch_counter += 1
        current_camera = camera_switch_counter % 3
        
        try:
            if current_camera == 0:
                viewport_api.set_active_camera(wrist_camera.prim_path)
                print("Switched to wrist camera view")
            elif current_camera == 1:
                viewport_api.set_active_camera(top_camera.prim_path)
                print("Switched to top camera view")
            elif current_camera == 2:
                viewport_api.set_active_camera(side_camera.prim_path)
                print("Switched to side camera view")
        except Exception as e:
            print(f"Error switching camera: {e}")
    
    step_count += 1

simulation_app.close()
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
from pxr import UsdGeom, Usd
from scipy.spatial.transform import Rotation as R

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

wrist_camera = Camera(
    prim_path="/World/wrist_camera_standalone",
    name="wrist_camera",
    frequency=30,
    resolution=(640, 480)
)

world.scene.add(top_camera) 
world.scene.add(side_camera)
world.scene.add(wrist_camera)

print("Cameras added to scene:")
print(f"- Top camera: {top_camera.prim_path}")
print(f"- Side camera: {side_camera.prim_path}")
print(f"- Wrist camera: {wrist_camera.prim_path} (follows robot wrist with offset)")

world.reset()

def get_link_world_pose(link_name):
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

cam_offset_x = -0.370  
cam_offset_y = 0.7  
cam_offset_z = 2
cam_rot_pitch = -120.0  
cam_rot_yaw = -90
cam_rot_roll = 0.0

top_cam_pos = np.array([0.5, 0, 8.0])  
top_cam_orient = np.array([1, 0, 1, 0])

side_cam_pos = np.array([6.0, -0.5, 0.5])
side_cam_orient = np.array([0, 0, 0, 1])

print("Calculated camera orientations:")
print(f"- Top camera quaternion (manual): {top_cam_orient}")
print(f"- Side camera quaternion: {side_cam_orient}")

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
print(f"- Top camera: pos={top_cam_pos}, looking straight down at robot") 
print(f"- Side camera: pos={side_cam_pos}, looking at robot from side")
print(f"- Wrist camera: follows wrist with offset X={cam_offset_x}, Y={cam_offset_y}, Z={cam_offset_z}")
print(f"  Wrist camera rotation: Pitch={cam_rot_pitch}°, Yaw={cam_rot_yaw}°, Roll={cam_rot_roll}°")

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

while simulation_app.is_running():
    world.step(render=True)
    
    update_camera_from_wrist(cam_offset_x, cam_offset_y, cam_offset_z, 
                            cam_rot_pitch, cam_rot_yaw, cam_rot_roll)
    
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
    
    if step_count % 120 == 0 and step_count > 0:
        try:
            wrist_camera.initialize()
            wrist_cam_data = wrist_camera.get_current_frame()
            if wrist_cam_data is not None:
                print(f"Wrist camera capturing data at step {step_count}")
        except Exception as e:
            print(f"Error accessing wrist camera data: {e}")
    
    step_count += 1

simulation_app.close()
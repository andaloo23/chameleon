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
from scipy.spatial.transform import Rotation as R
import omni.kit.viewport.utility
from pxr import UsdGeom

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

import time
time.sleep(1.0)

def create_wrist_camera():
    global wrist_camera
    wrist_camera_path = f"{prim_path}/wrist/wrist_camera_sensor"
    
    wrist_camera = Camera(
        prim_path=wrist_camera_path,
        name="wrist_camera",
        frequency=30,
        resolution=(640, 480),
    )
    
    world.scene.add(wrist_camera)
    
    wrist_camera.initialize()
    
    stage = get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(wrist_camera_path)
    
    camera_schema = UsdGeom.Camera(camera_prim)
    camera_schema.GetFocalLengthAttr().Set(18.0)

create_wrist_camera()

def update_wrist_camera_position(verbose=False):
    if wrist_camera is None:
        return None, None
        
    try:
        local_translation = np.array([0.0, 0.25, 1])
        
        rotation_attempts = [
            ("Roll +90°", R.from_euler('xyz', [1.57079632679, 0.0, 0.0])),
            ("Pitch -90°", R.from_euler('xyz', [0.0, -1.57079632679, 0.0])),
            ("Roll +90° + Yaw 180°", R.from_euler('xyz', [1.57079632679, 0.0, 3.14159265359])),
            ("Matrix approach with 45° pitch", R.from_matrix(np.array([
                [1, 0, 0],           # X-axis: right
                [0, 0, 1],           # Y-axis: forward  
                [0, -1, 0]           # Z-axis: down
            ])) * R.from_euler('z', -0.785398163))
        ]
        
        local_rotation = rotation_attempts[3][1]
        local_quat = local_rotation.as_quat()
        wrist_camera.set_local_pose(
            translation=local_translation,
            orientation=np.array([local_quat[0], local_quat[1], local_quat[2], local_quat[3]])
        )
        
        if verbose:
            print(f"Camera quaternion: {local_quat}")

            world_pos, world_orient = wrist_camera.get_world_pose()
            print(f"Camera local transform set. World pose: pos={world_pos}, orient={world_orient}")
            
            fov = wrist_camera.get_horizontal_fov()
            print(f"Camera FOV: {fov} degrees")
        
        return wrist_camera.get_world_pose()
                
    except Exception as e:
        if verbose:
            print(f"Error setting camera local pose: {e}")
    
    return None, None


update_wrist_camera_position(verbose=False)

top_cam_pos = np.array([0.5, 0, 8.0])  
top_cam_orient = np.array([1, 0, 1, 0])

side_cam_pos = np.array([6.0, -0.5, 0.5])
side_cam_orient = np.array([0, 0, 0, 1])


top_camera.set_world_pose(position=top_cam_pos, orientation=top_cam_orient)
side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

step_count = 0
rotation_angle = 0.0
rotation_direction = 1

while simulation_app.is_running():
    world.step(render=True)
    
    if step_count % 60 == 0: 
        rotation_angle += rotation_direction * 0.1 
        
        if rotation_angle >= 1.57:
            rotation_direction = -1
        elif rotation_angle <= -1.57:
            rotation_direction = 1
        
        joint_positions = np.zeros(len(joint_names))
        if "shoulder_pan" in joint_names:
            shoulder_pan_index = joint_names.index("shoulder_pan")
            joint_positions[shoulder_pan_index] = rotation_angle
            
            robot.set_joint_positions(joint_positions)
    
    step_count += 1

simulation_app.close()
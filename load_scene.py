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
import omni.kit.viewport.utility

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

# Create camera that will be positioned relative to wrist every frame
wrist_camera = Camera(
    prim_path="/World/wrist_camera_sensor", 
    name="wrist_camera",
    frequency=30,
    resolution=(640, 480)
)

world.scene.add(top_camera) 
world.scene.add(side_camera)
world.scene.add(wrist_camera)


world.reset()

def update_wrist_camera_position():
    """Update camera to match wrist_camera link transform exactly - maintains constant view relative to gripper"""
    stage = get_context().get_stage()
    
    # First try to use the wrist_camera link from URDF
    wrist_camera_pos, wrist_camera_orient = get_link_world_pose("wrist_camera")
    
    if wrist_camera_pos is not None and wrist_camera_orient is not None and not np.allclose(wrist_camera_pos, [0, 0, 0]):
        # The wrist_camera link is available and has valid position
        try:
            wrist_camera.set_world_pose(position=wrist_camera_pos, orientation=wrist_camera_orient)
            return wrist_camera_pos, wrist_camera_orient
        except:
            pass
    
    # Fallback: use wrist link with manual offset matching URDF configuration
    wrist_pos, wrist_orient = get_link_world_pose("wrist")
    if wrist_pos is not None and wrist_orient is not None:
        from scipy.spatial.transform import Rotation as R
        
        # Get wrist rotation
        wrist_rot = R.from_quat([wrist_orient[0], wrist_orient[1], wrist_orient[2], wrist_orient[3]])
        
        # Apply camera offset: position much higher up for wider top-down view
        local_offset = np.array([0.0, 0.0, 0.8])  # Directly above wrist, 80cm up for wide view
        world_offset = wrist_rot.as_matrix() @ local_offset
        cam_pos = wrist_pos + world_offset
        
        # Apply camera rotation: straight down top-down view
        local_cam_rot = R.from_euler('xyz', [0.0, -1.57079, 0.0])  # -90° pitch (straight down)
        combined_rot = wrist_rot * local_cam_rot
        cam_quat = combined_rot.as_quat()
        
        try:
            wrist_camera.set_world_pose(position=cam_pos, orientation=cam_quat)
            return cam_pos, cam_quat
        except:
            pass
    
    return None, None

# Initialize wrist camera
try:
    wrist_camera.initialize()
    
    # Check camera pose after a moment
    import time
    time.sleep(0.1)
    cam_pos, cam_orient = wrist_camera.get_world_pose()
except:
    pass

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


top_cam_pos = np.array([0.5, 0, 8.0])  
top_cam_orient = np.array([1, 0, 1, 0])

side_cam_pos = np.array([6.0, -0.5, 0.5])
side_cam_orient = np.array([0, 0, 0, 1])


try:
    top_camera.set_world_pose(position=top_cam_pos, orientation=top_cam_orient)
except:
    pass

try:
    side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)
except:
    pass


# Create a dedicated viewport for the wrist camera
try:
    viewport_api = omni.kit.viewport.utility.get_viewport_interface()
    if viewport_api:
        # Create a new viewport window for the wrist camera
        viewport_api.create_instance()
except:
    pass

joint_names = robot.dof_names

if joint_names is None:
    try:
        articulation = robot.get_articulation_controller()
        if articulation:
            joint_names = articulation.get_dof_names()
    except:
        pass
    if joint_names is None:
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

step_count = 0
rotation_angle = 0.0
rotation_direction = 1

while simulation_app.is_running():
    world.step(render=True)
    
    # Update camera to match wrist_camera link transform exactly
    update_wrist_camera_position()
    
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
        except:
            pass
    
    if step_count % 120 == 0 and step_count > 0:
        try:
            wrist_camera.initialize()
            # Debug camera pose
            cam_pos, cam_orient = wrist_camera.get_world_pose()
            if cam_pos is not None:
                print(f"  Camera position: {cam_pos}")
                print(f"  Camera orientation: {cam_orient}")
        except:
            pass
    
    step_count += 1

simulation_app.close()
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

world.scene.add(top_camera) 
world.scene.add(side_camera)


world.reset()

import time
time.sleep(1.0)

def create_wrist_camera():
    global wrist_camera
    try:
        wrist_camera_path = f"{prim_path}/wrist/wrist_camera_sensor"
        
        wrist_camera = Camera(
            prim_path=wrist_camera_path,
            name="wrist_camera",
            frequency=30,
            resolution=(640, 480),
        )
        
        world.scene.add(wrist_camera)
        
        wrist_camera.initialize()
        
        try:
            stage = get_context().get_stage()
            camera_prim = stage.GetPrimAtPath(wrist_camera_path)
            
            if camera_prim:
                try:
                    wrist_camera.set_horizontal_fov(60.0)
                    print(f"Set camera FOV to 60 degrees via Isaac Sim method")
                except:
                    try:
                        from pxr import UsdGeom
                        camera_schema = UsdGeom.Camera(camera_prim)
                        camera_schema.GetFocalLengthAttr().Set(18.0)
                        print(f"Set camera focal length to 18mm for wider FOV")
                    except Exception as usd_e:
                        print(f"Could not set FOV via USD: {usd_e}")
        except Exception as fov_e:
            print(f"Could not set FOV: {fov_e}")
        
        print(f"Created wrist camera at {wrist_camera_path}")
        return True
        
    except Exception as e:
        print(f"Error creating wrist camera: {e}")
        return False

create_wrist_camera()

def update_wrist_camera_position(verbose=False):
    if wrist_camera is None:
        return None, None
        
    try:
        stage = get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(wrist_camera.prim_path)
        
        if camera_prim and camera_prim.IsValid():
            from pxr import Gf
            from scipy.spatial.transform import Rotation as R
            
            local_translation = np.array([0.0, -0.15, 0.50])  # 50cm above wrist, 15cm forward
            
            rotation_attempts = [
                ("Roll +90°", R.from_euler('xyz', [1.57079632679, 0.0, 0.0])),
                ("Pitch -90°", R.from_euler('xyz', [0.0, -1.57079632679, 0.0])),
                ("Roll +90° + Yaw 180°", R.from_euler('xyz', [1.57079632679, 0.0, 3.14159265359])),
                ("Matrix approach", R.from_matrix(np.array([
                    [1, 0, 0],    # X-axis: right
                    [0, 0, 1],    # Y-axis: forward
                    [0, -1, 0]    # Z-axis: down
                ])))
            ]
            
            local_rotation = rotation_attempts[3][1]  # Matrix approach
            local_quat = local_rotation.as_quat()
            
            if verbose:
                print(f"Using matrix rotation approach for clean downward view")
                print(f"Camera quaternion: {local_quat}")
            
            usd_quat = Gf.Quatd(local_quat[3], local_quat[0], local_quat[1], local_quat[2])
            
            wrist_camera.set_local_pose(
                translation=local_translation,
                orientation=np.array([local_quat[0], local_quat[1], local_quat[2], local_quat[3]])
            )
            
            if verbose:
                world_pos, world_orient = wrist_camera.get_world_pose()
                print(f"Camera local transform set. World pose: pos={world_pos}, orient={world_orient}")
                
                try:
                    fov = wrist_camera.get_horizontal_fov()
                    print(f"Camera FOV: {fov} degrees")
                except:
                    print("Could not get camera FOV")
            
            return wrist_camera.get_world_pose()
            
        else:
            if verbose:
                print("Camera prim not found or invalid")
                
    except Exception as e:
        if verbose:
            print(f"Error setting camera local pose: {e}")
    
    return None, None


# Set initial camera position
if wrist_camera is not None:
    try:
        update_wrist_camera_position(verbose=True)
        
        render_product = wrist_camera.get_render_product_path()
        if render_product:
            print(f"Wrist camera render product: {render_product}")
        else:
            print("Warning: Wrist camera has no render product")
            
    except Exception as e:
        print(f"Failed to setup wrist camera: {e}")

def get_link_world_pose(link_name, verbose=False):
    stage = get_context().get_stage()
    link_path = f"{prim_path}/{link_name}"
    prim = stage.GetPrimAtPath(link_path)
    
    if verbose:
        print(f"Looking for link: {link_name} at path: {link_path}")
        print(f"Prim exists: {prim.IsValid() if prim else False}")
    
    if prim and prim.IsValid():
        try:
            xform = UsdGeom.Xformable(prim)
            transform_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            
            translation = transform_matrix.ExtractTranslation()
            position = np.array([translation[0], translation[1], translation[2]])
            
            rotation = transform_matrix.ExtractRotationQuat()
            orientation = np.array([rotation.GetImaginary()[0], 
                                   rotation.GetImaginary()[1], 
                                   rotation.GetImaginary()[2], 
                                   rotation.GetReal()])
            
            if verbose:
                print(f"Found valid pose for {link_name}: pos={position}, orient={orientation}")
            return position, orientation
        except Exception as e:
            if verbose:
                print(f"Error getting pose for {link_name}: {e}")
    else:
        if verbose:
            print(f"Link {link_name} not found at {link_path}")
    
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


def setup_wrist_camera_viewport():
    """Setup viewport for wrist camera to ensure it's visible"""
    try:
        import omni.kit.viewport.utility as viewport_utils
        
        viewport_api = viewport_utils.get_viewport_interface()
        if viewport_api:
            viewport_window = viewport_utils.get_active_viewport_window()
            if viewport_window:
                viewport_window.set_active_camera(wrist_camera.prim_path)
                print(f"Set wrist camera as active camera: {wrist_camera.prim_path}")
            else:
                print("No active viewport window found")
        else:
            print("Viewport API not available")
    except Exception as e:
        print(f"Failed to setup wrist camera viewport: {e}")

setup_wrist_camera_viewport()

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
    
    if step_count == 60:
        update_wrist_camera_position(verbose=True)
    
    
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
    
    if step_count % 120 == 0 and step_count > 0 and wrist_camera is not None:
        try:
            cam_pos, cam_orient = wrist_camera.get_world_pose()
            wrist_pos, wrist_orient = get_link_world_pose("wrist")
            
            print(f"Step {step_count}:")
            print(f"  Wrist link position: {wrist_pos}")
            print(f"  Camera sensor position: {cam_pos}")
            print(f"  Camera sensor orientation: {cam_orient}")
            
            print("---")
        except Exception as e:
            print(f"Debug error: {e}")
    
    step_count += 1

simulation_app.close()
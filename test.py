import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False, 
    "load_stage_on_start": False 
})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
from omni.usd import get_context
import numpy as np
from omni.isaac.sensor import Camera
import omni.replicator.core as rep

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
    prim_path=f"{prim_path}/wrist_camera",
    name="wrist_camera",
    position=np.array([0, 0, 0]), 
    orientation=np.array([1, 0, 0, 0]), 
    frequency=30,
    resolution=(640, 480)
)

top_camera = Camera(
    prim_path=f"{prim_path}/top_camera", 
    name="top_camera",
    position=np.array([0, 0, 0]), 
    orientation=np.array([1, 0, 0, 0]), 
    frequency=30,
    resolution=(640, 480)
)

side_camera = Camera(
    prim_path=f"{prim_path}/side_camera",
    name="side_camera", 
    position=np.array([0, 0, 0]), 
    orientation=np.array([1, 0, 0, 0]), 
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
            
        try:
            robot_prim = world.stage.GetPrimAtPath(prim_path)
            if robot_prim:
                print(f"Step {step_count}: Intended shoulder rotation: {rotation_angle:.2f} radians ({np.degrees(rotation_angle):.1f} degrees)")
        except Exception as e:
            pass
    
    step_count += 1

simulation_app.close()
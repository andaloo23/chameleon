import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False, 
    "load_stage_on_start": False 
})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.usd import get_context
import numpy as np
from omni.isaac.sensor import Camera
from robot import SO100Robot

get_context().new_stage()

current_dir = os.path.dirname(os.path.abspath(__file__))

world = World(stage_units_in_meters=1.0)

world.scene.add_default_ground_plane()

urdf_path = os.path.join(current_dir, "so100.urdf")

# Create the robot object
robot = SO100Robot(world, urdf_path)

cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="collision_cube",
    position=np.array([0.5, -0.5, 0]),
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

import time
time.sleep(1.0)

# Create and setup the wrist camera
robot.create_wrist_camera()
robot.update_wrist_camera_position(verbose=False)

top_cam_pos = np.array([0, -0.75, 8.0])  
top_cam_orient = np.array([-np.sqrt(0.25), -np.sqrt(0.25), -np.sqrt(0.25), np.sqrt(0.25)])

side_cam_pos = np.array([6.0, -0.5, 0.5])
side_cam_orient = np.array([0, 0, 0, 1])


top_camera.set_world_pose(position=top_cam_pos, orientation=top_cam_orient)
side_camera.set_world_pose(position=side_cam_pos, orientation=side_cam_orient)

step_count = 0

while simulation_app.is_running():
    world.step(render=True)
    step_count += 1

    if step_count % 30 == 0:
        # Generate and send random joint values
        random_joint_positions = robot.get_random_joint_positions()
        robot.set_joint_positions(random_joint_positions)

simulation_app.close()
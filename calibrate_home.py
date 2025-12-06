
import time
import numpy as np
from load_scene import IsaacPickPlaceEnv

def calibrate():
    print("[info] Initializing environment...")
    env = IsaacPickPlaceEnv(headless=True, capture_images=False)
    env.reset()

    # grasp configuration
    # IMPORTANT: Must respect URDF limits!
    #   shoulder_lift: 0 to 3.5 (POSITIVE ONLY! 0=high, 3.5=low)
    #   elbow_flex: -3.14 to 0 (NEGATIVE ONLY! 0=straight, -3.14=bent)
    grasp_config = np.array([
        0.0,    # shoulder_pan
        2.2,    # shoulder_lift (positive! higher value = lower position)
        -1.2,   # elbow_flex (negative! bent elbow)
        0.5,    # wrist_flex
        0.0,    # wrist_roll
        0.0,    # gripper
    ], dtype=np.float32)

    print(f"[info] Moving to grasp configuration: {grasp_config}")
    
    for _ in range(120):
        env.step(grasp_config, render=False)

    if env.robot.wrist_camera:
        cam_pos, cam_orient = env.robot.wrist_camera.get_world_pose()
        print(f"\n[RESULT] Wrist Camera Position (xyz): {cam_pos}")
        print(f"[RESULT] Wrist Camera Orientation (quat): {cam_orient}")
    else:
        print("[error] Wrist camera not found on robot.")

    if env.robot_articulation:
        base_pos, base_rot = env.robot_articulation.get_world_pose()
        print(f"[RESULT] Robot Base Position (xyz): {base_pos}")
    
    if env.cube:
        cube_pos, _ = env.cube.get_world_pose()
        print(f"[RESULT] Current Cube Position (xyz): {cube_pos}")
        
        if env.robot.wrist_camera:
             dist = np.linalg.norm(np.array(cam_pos) - np.array(cube_pos))
             print(f"[RESULT] Distance from Camera to Cube: {dist:.4f} meters")

    env.shutdown()

if __name__ == "__main__":
    calibrate()


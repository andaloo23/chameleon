import numpy as np
from load_scene import IsaacPickPlaceEnv
import time

def test_kinematics():
    # Initialize environment in headless mode for quick checking, 
    # or non-headless to see it.
    env = IsaacPickPlaceEnv(headless=False)
    env.reset()
    
    robot = env.robot
    print("\n--- Testing Preset Positions ---")
    for pid in ["1", "2", "3", "4"]:
        print(f"Moving to Preset {pid}...")
        robot.move_to_preset(pid)
        for _ in range(30): env.world.step(render=True)
        
        pos = robot.get_robot().get_joint_positions()
        print(f"Preset {pid} joint positions (rad): {pos}")
        time.sleep(0.5)

    print("\n--- Testing Cartesian IK ---")
    # Try a reachable point
    # X=150, Z=200
    target_x, target_z = 150.0, 200.0
    print(f"Moving to Cartesian: X={target_x}, Z={target_z}")
    success = robot.move_to_cartesian(target_x, target_z)
    if success:
        for _ in range(60): env.world.step(render=True)
        # In a real test, we would use get_world_pose() of the gripper to verify accuracy
        lp, lo = env._get_link_poses()
        gripper_pos = lp[env._gripper_link_idx] if env._gripper_link_idx is not None else "Unknown"
        print(f"Actual Gripper World Position: {gripper_pos}")
        print(f"Expected Z (approx): {target_z / 1000.0}m")
    else:
        print("Cartesian move failed (invalid target).")

    env.shutdown()

if __name__ == "__main__":
    test_kinematics()

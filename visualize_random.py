"""
Visualization script: Run one episode with random joint actions.

Usage:
    ~/isaacsim/python.sh visualize_random.py
"""

import numpy as np
from ppo_env import PPOEnv


def run_random_episode(max_steps: int = 200):
    """Run a single episode with random actions for visualization."""
    
    print("=" * 60)
    print("Random Joint Visualization")
    print("=" * 60)
    
    # Create environment with rendering enabled (headless=False)
    env = PPOEnv(headless=False, max_steps=max_steps, delta_scale=0.1)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 60)
    
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    total_reward = 0.0
    min_gripper_cube_dist = float("inf")
    min_gripper_width = float("inf")
    
    for step in range(max_steps):
        # Random action in [-1, 1]
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Track statistics
        task_state = info.get("task_state", {})
        gripper_cube_dist = task_state.get("gripper_cube_distance")
        gripper_joint = task_state.get("gripper_joint")
        
        if gripper_cube_dist is not None:
            min_gripper_cube_dist = min(min_gripper_cube_dist, gripper_cube_dist)
        if gripper_joint is not None:
            min_gripper_width = min(min_gripper_width, gripper_joint)
        
        # Print every 20 steps
        if (step + 1) % 20 == 0:
            flags = info.get("milestone_flags", {})
            print(f"Step {step+1:3d} | "
                  f"Dist: {gripper_cube_dist:.3f}m | "
                  f"Gripper: {gripper_joint:.3f} | "
                  f"Reward: {reward:.2f} | "
                  f"Reached: {flags.get('reached', False)}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            break
    
    # Final summary
    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"Total Steps: {step + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Min Gripper-Cube Distance: {min_gripper_cube_dist:.4f}m")
    print(f"Min Gripper Width: {min_gripper_width:.4f}")
    
    flags = info.get("milestone_flags", {})
    print(f"Milestones: Reached={flags.get('reached')}, "
          f"Controlled={flags.get('controlled')}, "
          f"Lifted={flags.get('lifted')}, "
          f"Success={flags.get('success')}")
    
    env.close()


if __name__ == "__main__":
    run_random_episode(max_steps=200)

"""
Visualization script: Run one episode with smooth random motion.

Usage:
    ~/isaacsim/python.sh visualize_random.py
"""

import numpy as np
from ppo_env import PPOEnv


def run_random_episode(max_steps: int = 300):
    """Run a single episode with smooth random motion for visualization."""
    
    print("=" * 60)
    print("Smooth Random Joint Visualization")
    print("=" * 60)
    
    # Create environment with rendering enabled (headless=False)
    # Use large delta_scale for visible motion
    env = PPOEnv(headless=False, max_steps=max_steps, delta_scale=0.5)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 60)
    
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    total_reward = 0.0
    min_gripper_cube_dist = float("inf")
    min_gripper_width = float("inf")
    
    # Use smooth correlated random motion instead of pure noise
    # Initialize a persistent action that drifts slowly
    current_action = np.zeros(6, dtype=np.float32)
    
    for step in range(max_steps):
        # Smooth random walk: drift current action with some momentum
        # Add small random delta to current action (correlated noise)
        drift = np.random.uniform(-0.3, 0.3, size=6).astype(np.float32)
        current_action = 0.9 * current_action + 0.1 * drift
        
        # Every 30 steps, pick a new random direction bias
        if step % 30 == 0:
            # Bias toward a random direction
            target_action = np.random.uniform(-1.0, 1.0, size=6).astype(np.float32)
            current_action = 0.5 * current_action + 0.5 * target_action
            print(f"  -> New direction at step {step}")
        
        # Clip to valid range
        action = np.clip(current_action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Track statistics
        task_state = info.get("task_state", {})
        gripper_cube_dist = task_state.get("gripper_cube_distance")
        gripper_width = task_state.get("gripper_width")  # Physical distance between jaws
        
        if gripper_cube_dist is not None:
            min_gripper_cube_dist = min(min_gripper_cube_dist, gripper_cube_dist)
        if gripper_width is not None:
            min_gripper_width = min(min_gripper_width, gripper_width)
        
        # Print every 30 steps
        if (step + 1) % 30 == 0:
            flags = info.get("milestone_flags", {})
            print(f"Step {step+1:3d} | "
                  f"Dist: {gripper_cube_dist:.3f}m | "
                  f"GripperWidth: {gripper_width:.4f if gripper_width else 'N/A'}m | "
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
    run_random_episode(max_steps=300)

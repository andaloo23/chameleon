#!/usr/bin/env python3
# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
Test script for visualizing the Isaac Lab pick-and-place environment.

Usage:
    # From Isaac Lab environment (with isaaclab in path):
    python tests/visualize_isaac_lab.py
    
    # Or with --headless for CI testing
    python tests/visualize_isaac_lab.py --headless
"""

from __future__ import annotations

import argparse
import sys

def main():
    # Parse arguments before imports (Isaac Lab convention)
    parser = argparse.ArgumentParser(description="Visualize Isaac Lab Pick-and-Place Environment")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of steps to run")
    args = parser.parse_args()
    
    # Launch Isaac Lab application
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    
    # Now we can import other modules
    import torch
    
    # Add parent directory to path for imports
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from lab.pick_place_env import PickPlaceEnv
    from lab.pick_place_env_cfg import PickPlaceEnvCfg
    
    print(f"\n{'='*60}")
    print("Isaac Lab Pick-and-Place Environment Visualization")
    print(f"{'='*60}\n")
    
    # Create configuration
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    
    print(f"Configuration:")
    print(f"  - Num environments: {cfg.scene.num_envs}")
    print(f"  - Action space: {cfg.action_space}")
    print(f"  - Observation space: {cfg.observation_space}")
    print(f"  - Episode length: {cfg.episode_length_s}s")
    print()
    
    # Create environment
    print("Creating environment...")
    env = PickPlaceEnv(cfg)
    print(f"Environment created with {env.num_envs} parallel environments")
    print(f"Device: {env.device}")
    print()
    
    # Reset environment
    print("Resetting environment...")
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print(f"Initial observation shape: {obs.shape}")
    print()
    
    # Run random actions
    print(f"Running {args.num_steps} steps with random actions...")
    total_reward = torch.zeros(env.num_envs, device=env.device)
    success_count = 0
    
    for step in range(args.num_steps):
        # Sample random actions
        actions = torch.rand(env.num_envs, 6, device=env.device) * 2 - 1  # [-1, 1]
        
        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        
        # Check for successes
        if terminated.any():
            success_count += terminated.sum().item()
            print(f"  Step {step}: {terminated.sum().item()} environment(s) terminated (success)")
        
        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{args.num_steps} - Mean reward: {total_reward.mean().item():.3f}")
        
        # Reset terminated environments
        done = terminated | truncated
        if done.any():
            # Note: DirectRLEnv handles auto-reset internally
            pass
    
    print()
    print(f"{'='*60}")
    print("Summary:")
    print(f"  - Total steps: {args.num_steps}")
    print(f"  - Total successes: {success_count}")
    print(f"  - Mean cumulative reward: {total_reward.mean().item():.3f}")
    print(f"  - Max cumulative reward: {total_reward.max().item():.3f}")
    print(f"{'='*60}\n")
    
    # Cleanup
    print("Closing environment...")
    env.close()
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()

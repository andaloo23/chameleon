#!/usr/bin/env python3
# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
Visualize curriculum mode (lift-only) in slow motion.

Use this to verify the robot starts with the cube gripped and that the cube
teleports to the gripper each step. Run with visualization (no --headless).

Usage:
    python tests/visualize_curriculum.py
    python tests/visualize_curriculum.py --step-delay 0.2   # Slower
    python tests/visualize_curriculum.py --headless         # CI / no display
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize curriculum mode (lift-only) in slow motion"
    )
    parser.add_argument("--headless", action="store_true", help="Run headless (no window)")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs (1 for easy viewing)")
    parser.add_argument("--num-steps", type=int, default=300, help="Number of steps to run")
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.1,
        help="Seconds to wait between steps (0.1 = slow motion, 0 = real-time)",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=30,
        help="Frames of zero action at start to inspect initial grip pose",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum (for comparison)",
    )
    parser.add_argument(
        "--pause-at-start",
        type=float,
        default=2.0,
        help="Seconds to pause at start before stepping (to inspect initial pose)",
    )
    args = parser.parse_args()

    # Isaac Lab requires AppLauncher before other imports
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import torch

    from lab.pick_place_env import PickPlaceEnv
    from lab.pick_place_env_cfg import PickPlaceEnvCfg

    print("\n" + "=" * 60)
    print("Curriculum Mode Visualization (Slow Motion)")
    print("=" * 60 + "\n")

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum_lift_only = not args.no_curriculum

    print("Configuration:")
    print(f"  - Num envs: {cfg.scene.num_envs}")
    print(f"  - Curriculum (lift-only): {cfg.curriculum_lift_only}")
    print(f"  - Step delay: {args.step_delay}s")
    print(f"  - Hold frames (zero action): {args.hold_frames}")
    print(f"  - Pause at start: {args.pause_at_start}s")
    print()

    env = PickPlaceEnv(cfg)
    obs_dict, _ = env.reset()
    print("Environment reset. Initial state (before first step):")
    if cfg.curriculum_lift_only:
        mid = 0.5 * (env._gripper_tip_pos + env._jaw_tip_pos)
        cube_pos = env.cube.data.root_pos_w
        print(f"  - Gripper midpoint: {mid[0].cpu().numpy()}")
        print(f"  - Cube position:    {cube_pos[0].cpu().numpy()}")
        dist = (mid - cube_pos).norm(dim=-1)[0].item()
        print(f"  - Distance (midpoint to cube): {dist:.4f}m")
        print("  (After first step, cube teleports to gripper midpoint.)")
    print()

    total_reward = torch.zeros(env.num_envs, device=env.device)

    # Pause at start so you can inspect initial pose (cube at curriculum_cube_xy)
    if args.pause_at_start > 0 and not args.headless:
        print(f"Pausing {args.pause_at_start}s — inspect initial pose (cube at curriculum_cube_xy)...")
        for _ in range(int(args.pause_at_start * 60)):  # ~60 updates/sec
            simulation_app.update()
            time.sleep(1.0 / 60.0)

    for step in range(args.num_steps):
        # Zero action for first N frames to inspect initial grip
        if step < args.hold_frames:
            actions = torch.zeros(env.num_envs, 6, device=env.device)
        else:
            # Gentle lift: positive action on shoulder_lift (index 1) to raise arm
            actions = torch.zeros(env.num_envs, 6, device=env.device)
            actions[:, 1] = 0.3  # Slight upward motion

        obs_dict, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward

        if step == 0:
            print("Step 0: Check that cube is at/near gripper fingertips.")
        elif step == args.hold_frames:
            print(f"Step {args.hold_frames}: Starting gentle lift (action[1]=0.3).")

        # Update display
        simulation_app.update()

        if args.step_delay > 0:
            time.sleep(args.step_delay)

    print("\n" + "=" * 60)
    print("Done. What to verify:")
    print("  1. At reset: gripper at curriculum_joint_pos, cube at curriculum_cube_xy")
    print("  2. After first step: cube teleports to gripper midpoint (robot 'gripping')")
    print("  3. During hold: cube stays at gripper (zero action)")
    print("  4. During lift: cube follows gripper upward")
    print("=" * 60 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

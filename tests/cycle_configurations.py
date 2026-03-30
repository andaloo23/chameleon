# Copyright (c) 2024, Chameleon Project
# SPDX-License-Identifier: MIT

"""
Cycle through all cup height and cube size configurations.

Resets the environment to each cup height variant sequentially,
holding each for ~3 seconds of simulation time. Cube size is
randomized independently at each reset. Prints dimensions to console.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cycle through cup/cube configurations")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from lab.pick_place_env import PickPlaceEnv
from lab.pick_place_env_cfg import PickPlaceEnvCfg


def main():
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 9999.0  # don't auto-terminate

    env = PickPlaceEnv(cfg)

    # Steps per 3-second hold
    steps_per_config = int(3.0 / (cfg.sim.dt * cfg.decimation))
    print(f"Steps per config: {steps_per_config} ({cfg.sim.dt * cfg.decimation:.4f}s per step)")

    num_cup_variants = len(cfg.cup_height_variants)
    num_cube_variants = len(cfg.cube_size_variants)
    print(f"Cup variants: {cfg.cup_height_variants}")
    print(f"Cube variants: {num_cube_variants} variants")

    obs, _ = env.reset()

    step = 0
    cup_variant_idx = 0
    env._debug_force_cup_variant = 0  # start with first cup

    # Force reset to initial cup variant
    env._reset_idx(torch.zeros(1, dtype=torch.long, device=env.device))

    zero_actions = torch.zeros(1, cfg.action_space, device=env.device)

    while simulation_app.is_running():
        # Check if it's time to switch to next cup variant
        if step > 0 and step % steps_per_config == 0:
            cup_variant_idx = (cup_variant_idx + 1) % num_cup_variants
            env._debug_force_cup_variant = cup_variant_idx
            env._reset_idx(torch.zeros(1, dtype=torch.long, device=env.device))

            # Print new configuration
            cup_h = cfg.cup_height_variants[cup_variant_idx]
            cube_dims = env._cube_dims[0].cpu().tolist()
            cube_variant = env._active_cube_variant[0].item()
            print(
                f"[step {step:6d}] Cup #{cup_variant_idx}: height={cup_h*100:.1f}cm | "
                f"Cube #{cube_variant}: "
                f"{cube_dims[0]*200:.1f}x{cube_dims[1]*200:.1f}x{cube_dims[2]*200:.1f} mm"
                f" (full dims)"
            )

        obs, rewards, terminated, truncated, info = env.step(zero_actions)
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

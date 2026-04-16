"""
Verify camera framing by saving sample frames to disk.

Run from repo root:
    python tests/check_cameras.py

Saves camera_check_third_person_envN.png and camera_check_wrist_envN.png
for the first --n-envs environments. Inspect with any image viewer.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--n-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20, help="Steps to run before capturing (lets robot settle)")
parser.add_argument("--out-dir", type=str, default="camera_check")
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import torch
from PIL import Image

from lab.pick_place_env import PickPlaceEnv
from lab.pick_place_env_cfg import PickPlaceEnvCfg

cfg = PickPlaceEnvCfg()
cfg.enable_cameras = True
cfg.scene.num_envs = args.n_envs

env = PickPlaceEnv(cfg)
env.reset()

# Run a few steps so the scene is fully settled before capturing
for _ in range(args.steps):
    actions = torch.zeros(args.n_envs, cfg.action_space, device=env.device)
    env.step(actions)

images = env.update_cameras()

os.makedirs(args.out_dir, exist_ok=True)
for name, frames in images.items():
    # frames: [N, H, W, 4] uint8 (RGBA)
    for i in range(min(args.n_envs, frames.shape[0])):
        frame = frames[i].cpu().numpy()
        img = Image.fromarray(frame[..., :3])  # drop alpha
        path = os.path.join(args.out_dir, f"{name}_env{i}.png")
        img.save(path)
        print(f"Saved {path}")

env.close()
simulation_app.close()

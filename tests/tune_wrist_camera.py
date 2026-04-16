#!/usr/bin/env python3
"""
Interactively tune the wrist camera offset.

Runs 1 env with cameras enabled. Move the robot joints into a useful pose,
then adjust the wrist camera position until the framing looks right.
Saves a frame to disk on demand so you can scp it and inspect it.

Joint Controls:
  Q/A: Shoulder Pan
  W/S: Shoulder Lift
  E/D: Elbow Flex
  R/V: Wrist Flex
  T/G: Wrist Roll

Wrist Camera Offset Controls (env-local frame of gripper link):
  1/2: X  -/+
  3/4: Y  -/+
  5/6: Z  -/+
  7/8: Step size  -/+

  P:    Print current offset + save frame to wrist_cam_check.png
  ESC:  Exit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import torch
import carb
import omni.appwindow
from PIL import Image

from lab.pick_place_env import PickPlaceEnv
from lab.pick_place_env_cfg import PickPlaceEnvCfg


def main():
    cfg = PickPlaceEnvCfg()
    cfg.enable_cameras = True
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0

    env = PickPlaceEnv(cfg)
    env.reset()

    joint_targets = env.robot.data.joint_pos[0].clone()
    STEP = 0.05

    # Current wrist camera offset (x, y, z in gripper-local frame)
    wrist_offset = list(cfg.camera_wrist_pos)
    cam_step = 0.005

    input_state = {"running": True}

    joint_map = {
        carb.input.KeyboardInput.Q: (0, +1),
        carb.input.KeyboardInput.A: (0, -1),
        carb.input.KeyboardInput.W: (1, +1),
        carb.input.KeyboardInput.S: (1, -1),
        carb.input.KeyboardInput.E: (2, +1),
        carb.input.KeyboardInput.D: (2, -1),
        carb.input.KeyboardInput.R: (3, +1),
        carb.input.KeyboardInput.V: (3, -1),
        carb.input.KeyboardInput.T: (4, +1),
        carb.input.KeyboardInput.G: (4, -1),
    }

    def _rebuild_wrist_camera():
        """Destroy and recreate the wrist TiledCamera with the updated offset."""
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        import isaaclab.sim as sim_utils

        env.camera_wrist._sensor_prims = []   # detach old prims
        try:
            env.camera_wrist.__del__()
        except Exception:
            pass

        wx, wy, wz = wrist_offset
        env.camera_wrist = TiledCamera(TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/gripper/wrist_cam",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            data_types=["rgb"],
            width=cfg.camera_width,
            height=cfg.camera_height,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(float(wx), float(wy), float(wz)),
                rot=(0.707, -0.707, 0.0, 0.0),
                convention="opengl",
            ),
        ))

    def _save_frame(tag: str = "wrist"):
        images = env.update_cameras()
        rgb = images[tag][0].cpu().numpy()
        path = f"{tag}_cam_check.png"
        Image.fromarray(rgb[..., :3]).save(path)
        print(f"[SAVE] Wrote {path}")

    def on_key(event):
        nonlocal cam_step
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        k = event.input
        if k == carb.input.KeyboardInput.ESCAPE:
            input_state["running"] = False

        elif k in joint_map:
            idx, d = joint_map[k]
            joint_targets[idx] += d * STEP

        elif k in (carb.input.KeyboardInput.KEY_1, carb.input.KeyboardInput.KEY_2,
                   carb.input.KeyboardInput.KEY_3, carb.input.KeyboardInput.KEY_4,
                   carb.input.KeyboardInput.KEY_5, carb.input.KeyboardInput.KEY_6):
            axis_map = {
                carb.input.KeyboardInput.KEY_1: (0, -1),
                carb.input.KeyboardInput.KEY_2: (0, +1),
                carb.input.KeyboardInput.KEY_3: (1, -1),
                carb.input.KeyboardInput.KEY_4: (1, +1),
                carb.input.KeyboardInput.KEY_5: (2, -1),
                carb.input.KeyboardInput.KEY_6: (2, +1),
            }
            axis, direction = axis_map[k]
            wrist_offset[axis] += direction * cam_step
            print(f"[CAM] wrist_pos = ({wrist_offset[0]:.4f}, {wrist_offset[1]:.4f}, {wrist_offset[2]:.4f})  step={cam_step:.4f}")
            _rebuild_wrist_camera()
            _save_frame("wrist")

        elif k == carb.input.KeyboardInput.KEY_7:
            cam_step = max(0.001, cam_step / 2)
            print(f"[CAM] step -> {cam_step:.4f}")
        elif k == carb.input.KeyboardInput.KEY_8:
            cam_step = min(0.05, cam_step * 2)
            print(f"[CAM] step -> {cam_step:.4f}")

        elif k == carb.input.KeyboardInput.P:
            print(f"[CAM] wrist_pos = ({wrist_offset[0]:.4f}, {wrist_offset[1]:.4f}, {wrist_offset[2]:.4f})")
            _save_frame("wrist")
            _save_frame("third_person")

        return True

    input_interface = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    _sub = input_interface.subscribe_to_keyboard_events(keyboard, on_key)

    print("\nJoint Controls:  Q/A W/S E/D R/V T/G")
    print("Wrist Cam Offset: 1/2=X  3/4=Y  5/6=Z  7/8=step")
    print("P: print offset + save frames   ESC: exit\n")
    print(f"Starting wrist offset: {wrist_offset}")
    print("-" * 50)

    while input_state["running"] and simulation_app.is_running():
        current_pos = env.robot.data.joint_pos[0]
        delta = (joint_targets - current_pos) / env.cfg.action_scale
        action = torch.clamp(delta, -1.0, 1.0).unsqueeze(0)
        env.step(action)
        simulation_app.update()

    env.close()
    simulation_app.close()
    print(f"\nFinal wrist_pos: camera_wrist_pos = {tuple(round(v, 4) for v in wrist_offset)}")


if __name__ == "__main__":
    main()

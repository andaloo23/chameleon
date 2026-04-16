#!/usr/bin/env python3
"""
Interactively tune the wrist camera position and orientation.

Joint Controls:
  Q/A: Shoulder Pan       W/S: Shoulder Lift
  E/D: Elbow Flex         R/V: Wrist Flex
  T/G: Wrist Roll         Y: Gripper open   U: Gripper close

Wrist Camera Translation (gripper-local frame):
  1/2: X -/+    3/4: Y -/+    5/6: Z -/+
  7/8: translation step -/+

Wrist Camera Rotation (euler degrees, applied on top of base Rx-90):
  I/K: Rx -/+    O/L: Ry -/+    F/H: Rz -/+
  9/0: rotation step -/+

  P:   Print current pos+rot and save wrist_cam_check.png + third_person_cam_check.png
  ESC: Exit and print final values to paste into cfg
"""
import math
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


def euler_to_quat(rx_deg, ry_deg, rz_deg):
    """XYZ intrinsic euler angles (degrees) → (w, x, y, z) quaternion."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return (w, x, y, z)


def main():
    cfg = PickPlaceEnvCfg()
    cfg.enable_cameras = True
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0

    env = PickPlaceEnv(cfg)
    env.reset()

    joint_targets = env.robot.data.joint_pos[0].clone()
    JOINT_STEP = 0.05

    # Translation state (gripper-local x, y, z)
    wrist_pos = list(cfg.camera_wrist_pos)
    pos_step = 0.005

    # Rotation state as euler degrees. Base orientation (0.707,-0.707,0,0) = Rx(-90).
    wrist_rot_euler = [-90.0, 0.0, 0.0]  # rx, ry, rz in degrees
    rot_step = 5.0  # degrees per keypress

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

    def _update_wrist_prim():
        import isaaclab.sim.utils.stage as stage_utils
        from pxr import UsdGeom, Gf

        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper/wrist_cam")
        if not prim or not prim.IsValid():
            print("[WARN] wrist_cam prim not found")
            return

        xf = UsdGeom.Xformable(prim)

        # Build a 4x4 matrix from current pos + euler rotation.
        # Using MakeMatrixXform avoids fighting whatever op types Isaac Lab created.
        w, x, y, z = euler_to_quat(*wrist_rot_euler)
        rot = Gf.Rotation(Gf.Quatd(float(w), float(x), float(y), float(z)))
        mat = Gf.Matrix4d()
        mat.SetRotate(rot)
        tx, ty, tz = wrist_pos
        mat.SetTranslateOnly(Gf.Vec3d(tx, ty, tz))

        xf.ClearXformOpOrder()
        xf.MakeMatrixXform().Set(mat)

    def _save_frame(tag: str = "wrist"):
        images = env.update_cameras()
        rgb = images[tag][0].cpu().numpy()
        path = f"{tag}_cam_check.png"
        Image.fromarray(rgb[..., :3]).save(path)
        print(f"[SAVE] {path}")

    def _print_state():
        q = euler_to_quat(*wrist_rot_euler)
        print(f"[CAM] pos=({wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f})  "
              f"rot_euler=({wrist_rot_euler[0]:.1f}°, {wrist_rot_euler[1]:.1f}°, {wrist_rot_euler[2]:.1f}°)  "
              f"quat=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")

    def on_key(event):
        nonlocal pos_step, rot_step
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        k = event.input
        if k == carb.input.KeyboardInput.ESCAPE:
            input_state["running"] = False

        # Gripper
        elif k == carb.input.KeyboardInput.Y:
            joint_targets[5] = torch.clamp(joint_targets[5] + 0.1, -0.5, 1.5)
        elif k == carb.input.KeyboardInput.U:
            joint_targets[5] = -0.2

        # Arm joints
        elif k in joint_map:
            idx, d = joint_map[k]
            joint_targets[idx] += d * JOINT_STEP

        # Translation
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
            wrist_pos[axis] += direction * pos_step
            _update_wrist_prim()
            _print_state()
            _save_frame("wrist")

        elif k == carb.input.KeyboardInput.KEY_7:
            pos_step = max(0.001, pos_step / 2)
            print(f"[CAM] pos_step -> {pos_step:.4f}m")
        elif k == carb.input.KeyboardInput.KEY_8:
            pos_step = min(0.05, pos_step * 2)
            print(f"[CAM] pos_step -> {pos_step:.4f}m")

        # Rotation
        elif k == carb.input.KeyboardInput.I:
            wrist_rot_euler[0] += rot_step
            _update_wrist_prim(); _print_state(); _save_frame("wrist")
        elif k == carb.input.KeyboardInput.K:
            wrist_rot_euler[0] -= rot_step
            _update_wrist_prim(); _print_state(); _save_frame("wrist")
        elif k == carb.input.KeyboardInput.O:
            wrist_rot_euler[1] += rot_step
            _update_wrist_prim(); _print_state(); _save_frame("wrist")
        elif k == carb.input.KeyboardInput.L:
            wrist_rot_euler[1] -= rot_step
            _update_wrist_prim(); _print_state(); _save_frame("wrist")
        elif k == carb.input.KeyboardInput.F:
            wrist_rot_euler[2] += rot_step
            _update_wrist_prim(); _print_state(); _save_frame("wrist")
        elif k == carb.input.KeyboardInput.H:
            wrist_rot_euler[2] -= rot_step
            _update_wrist_prim(); _print_state(); _save_frame("wrist")

        elif k == carb.input.KeyboardInput.KEY_9:
            rot_step = max(1.0, rot_step / 2)
            print(f"[CAM] rot_step -> {rot_step:.1f}°")
        elif k == carb.input.KeyboardInput.KEY_0:
            rot_step = min(45.0, rot_step * 2)
            print(f"[CAM] rot_step -> {rot_step:.1f}°")

        elif k == carb.input.KeyboardInput.P:
            _print_state()
            _save_frame("wrist")
            _save_frame("third_person")

        return True

    input_interface = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    _sub = input_interface.subscribe_to_keyboard_events(keyboard, on_key)

    print("\nJoints:      Q/A W/S E/D R/V T/G  |  Y=open  U=close")
    print("Translation: 1/2=X  3/4=Y  5/6=Z  |  7/8=pos step")
    print("Rotation:    I/K=Rx  O/L=Ry  F/H=Rz  |  9/0=rot step")
    print("P: save frames   ESC: exit\n")
    _print_state()
    print("-" * 60)

    while input_state["running"] and simulation_app.is_running():
        current_pos = env.robot.data.joint_pos[0]
        delta = (joint_targets - current_pos) / env.cfg.action_scale
        action = torch.clamp(delta, -1.0, 1.0).unsqueeze(0)
        env.step(action)
        simulation_app.update()

    env.close()
    simulation_app.close()

    q = euler_to_quat(*wrist_rot_euler)
    print(f"\n── Paste into pick_place_env_cfg.py ──")
    print(f"camera_wrist_pos: tuple = {tuple(round(v, 4) for v in wrist_pos)}")
    print(f"\n── Paste into _create_camera_sensors (OffsetCfg rot=) ──")
    print(f"rot=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}),")


if __name__ == "__main__":
    main()

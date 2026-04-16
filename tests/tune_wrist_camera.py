#!/usr/bin/env python3
"""
Interactively tune wrist and top-down cameras.

TAB: toggle active camera (wrist / top)

Joint Controls:
  Q/A: Shoulder Pan    W/S: Shoulder Lift
  E/D: Elbow Flex      R/V: Wrist Flex
  T/B: Wrist Roll      Y: Gripper open   U: Gripper close

Active Camera Translation:
  1/2: X -/+    3/4: Y -/+    5/6: Z -/+
  7/8: translation step -/+

Active Camera Rotation (euler degrees):
  I/K: Rx -/+    O/L: Ry -/+    F/H: Rz -/+
  9/0: rotation step -/+

  P:   Print current values + save both frames
  ESC: Exit and print final cfg values
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
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    return (
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    )


def _set_prim_matrix(prim_path, pos, euler):
    import isaaclab.sim.utils.stage as stage_utils
    from pxr import UsdGeom, Gf

    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[WARN] prim not found: {prim_path}")
        return
    w, x, y, z = euler_to_quat(*euler)
    rot = Gf.Rotation(Gf.Quatd(float(w), float(x), float(y), float(z)))
    mat = Gf.Matrix4d()
    mat.SetRotate(rot)
    mat.SetTranslateOnly(Gf.Vec3d(*pos))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.MakeMatrixXform().Set(mat)


def main():
    cfg = PickPlaceEnvCfg()
    cfg.enable_cameras = True
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0

    env = PickPlaceEnv(cfg)
    env.reset()

    joint_targets = env.robot.data.joint_pos[0].clone()
    JOINT_STEP = 0.05

    # Per-camera state: [pos, euler_rot]
    # Top camera: seed from cfg eye/target → convert target to euler
    cameras = {
        "wrist": {
            "prim": "/World/envs/env_0/Robot/gripper/wrist_cam",
            "save_tag": "wrist",
            "pos": list(cfg.camera_wrist_pos),
            "euler": [-90.0, 0.0, 0.0],
        },
        "top": {
            "prim": "/World/envs/env_0/third_person_cam",
            "save_tag": "third_person",
            "pos": list(cfg.camera_third_person_eye),
            "euler": [-180.0, 0.0, 0.0],  # straight down, adjust from here
        },
    }
    active = "wrist"
    pos_step = 0.005
    rot_step = 5.0

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
        carb.input.KeyboardInput.B: (4, -1),
    }

    def _update():
        cam = cameras[active]
        _set_prim_matrix(cam["prim"], cam["pos"], cam["euler"])

    def _save_frame(tag):
        images = env.update_cameras()
        rgb = images[tag][0].cpu().numpy()
        path = f"{tag}_cam_check.png"
        Image.fromarray(rgb[..., :3]).save(path)
        print(f"[SAVE] {path}")

    def _print_state():
        cam = cameras[active]
        q = euler_to_quat(*cam["euler"])
        p = cam["pos"]
        e = cam["euler"]
        print(f"[{active.upper()}] pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
              f"euler=({e[0]:.1f}°, {e[1]:.1f}°, {e[2]:.1f}°)  "
              f"quat=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")

    def on_key(event):
        nonlocal active, pos_step, rot_step
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        k = event.input
        cam = cameras[active]

        if k == carb.input.KeyboardInput.ESCAPE:
            input_state["running"] = False

        elif k == carb.input.KeyboardInput.TAB:
            active = "top" if active == "wrist" else "wrist"
            print(f"[MODE] Active camera: {active.upper()}")
            _print_state()

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
            cam["pos"][axis] += direction * pos_step
            _update(); _print_state(); _save_frame(cam["save_tag"])

        elif k == carb.input.KeyboardInput.KEY_7:
            pos_step = max(0.001, pos_step / 2)
            print(f"[CAM] pos_step -> {pos_step:.4f}m")
        elif k == carb.input.KeyboardInput.KEY_8:
            pos_step = min(0.10, pos_step * 2)
            print(f"[CAM] pos_step -> {pos_step:.4f}m")

        # Rotation
        elif k == carb.input.KeyboardInput.I:
            cam["euler"][0] += rot_step
            _update(); _print_state(); _save_frame(cam["save_tag"])
        elif k == carb.input.KeyboardInput.K:
            cam["euler"][0] -= rot_step
            _update(); _print_state(); _save_frame(cam["save_tag"])
        elif k == carb.input.KeyboardInput.O:
            cam["euler"][1] += rot_step
            _update(); _print_state(); _save_frame(cam["save_tag"])
        elif k == carb.input.KeyboardInput.L:
            cam["euler"][1] -= rot_step
            _update(); _print_state(); _save_frame(cam["save_tag"])
        elif k == carb.input.KeyboardInput.F:
            cam["euler"][2] += rot_step
            _update(); _print_state(); _save_frame(cam["save_tag"])
        elif k == carb.input.KeyboardInput.H:
            cam["euler"][2] -= rot_step
            _update(); _print_state(); _save_frame(cam["save_tag"])

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

    print("\nTAB: toggle active camera (wrist / top)")
    print("Joints:      Q/A W/S E/D R/V T/B  |  Y=open  U=close")
    print("Translation: 1/2=X  3/4=Y  5/6=Z  |  7/8=pos step")
    print("Rotation:    I/K=Rx  O/L=Ry  F/H=Rz  |  9/0=rot step")
    print("P: save both frames   ESC: exit\n")
    print(f"Active camera: {active.upper()}")
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

    print("\n── Paste into pick_place_env_cfg.py ──")
    for name, cam in cameras.items():
        q = euler_to_quat(*cam["euler"])
        p = cam["pos"]
        print(f"\n# {name}")
        if name == "wrist":
            print(f"camera_wrist_pos: tuple = {tuple(round(v, 4) for v in p)}")
        else:
            print(f"camera_third_person_eye: tuple = {tuple(round(v, 4) for v in p)}")
        print(f"# rot=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})  "
              f"euler=({cam['euler'][0]:.1f}, {cam['euler'][1]:.1f}, {cam['euler'][2]:.1f})")


if __name__ == "__main__":
    main()

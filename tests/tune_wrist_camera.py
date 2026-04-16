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

Active Camera Rotation (incremental, world-space axes — no gimbal lock):
  I/K: world X -/+    O/L: world Y -/+    F/H: world Z -/+
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


def quat_mul(a, b):
    """Multiply two (w,x,y,z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    return (w/n, x/n, y/n, z/n)


def axis_angle_to_quat(ax, ay, az, angle_deg):
    """Unit axis (ax,ay,az) + angle in degrees → (w,x,y,z)."""
    a = math.radians(angle_deg) / 2
    s = math.sin(a)
    return (math.cos(a), ax*s, ay*s, az*s)


def euler_to_quat(rx_deg, ry_deg, rz_deg):
    """XYZ intrinsic euler (degrees) → (w,x,y,z). Used only for wrist seed."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cx, sx = math.cos(rx/2), math.sin(rx/2)
    cy, sy = math.cos(ry/2), math.sin(ry/2)
    cz, sz = math.cos(rz/2), math.sin(rz/2)
    return (
        cx*cy*cz + sx*sy*sz,
        sx*cy*cz - cx*sy*sz,
        cx*sy*cz + sx*cy*sz,
        cx*cy*sz - sx*sy*cz,
    )


def _read_prim_state(prim_path):
    """Read local pos and quat (w,x,y,z) from a prim as Isaac Lab set it up."""
    import isaaclab.sim.utils.stage as stage_utils
    from pxr import UsdGeom, Gf

    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return [0.0, 0.0, 0.0], (1.0, 0.0, 0.0, 0.0)

    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}

    pos = [0.0, 0.0, 0.0]
    quat = (1.0, 0.0, 0.0, 0.0)

    if "xformOp:transform" in ops:
        mat = ops["xformOp:transform"].Get()
        t = mat.ExtractTranslation()
        pos = [t[0], t[1], t[2]]
        q = mat.ExtractRotation().GetQuat()
        im = q.GetImaginary()
        quat = (q.GetReal(), im[0], im[1], im[2])
    else:
        if "xformOp:translate" in ops:
            t = ops["xformOp:translate"].Get()
            pos = [t[0], t[1], t[2]]
        for op_name in ("xformOp:orient", "xformOp:orientf"):
            if op_name in ops:
                q = ops[op_name].Get()
                im = q.GetImaginary()
                quat = (q.GetReal(), im[0], im[1], im[2])
                break

    return pos, quat


def _set_prim_matrix(prim_path, pos, quat):
    import isaaclab.sim.utils.stage as stage_utils
    from pxr import UsdGeom, Gf

    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"[WARN] prim not found: {prim_path}")
        return
    w, x, y, z = quat
    # Build matrix directly from quaternion to avoid Gf.Rotation constructor ambiguity
    q2 = [w*w, x*x, y*y, z*z]
    mat = Gf.Matrix4d(
        q2[0]+q2[1]-q2[2]-q2[3], 2*(x*y+w*z),           2*(x*z-w*y),           0,
        2*(x*y-w*z),           q2[0]-q2[1]+q2[2]-q2[3], 2*(y*z+w*x),           0,
        2*(x*z+w*y),           2*(y*z-w*x),           q2[0]-q2[1]-q2[2]+q2[3], 0,
        pos[0],                pos[1],                pos[2],                    1,
    )
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

    # Read actual transforms Isaac Lab put on the prims — don't guess/seed them.
    wrist_prim = "/World/envs/env_0/Robot/gripper/wrist_cam"
    top_prim   = "/World/envs/env_0/third_person_cam"
    wrist_pos_init, wrist_quat_init = _read_prim_state(wrist_prim)
    top_pos_init,   top_quat_init   = _read_prim_state(top_prim)

    cameras = {
        "wrist": {
            "prim": wrist_prim,
            "save_tag": "wrist",
            "pos": wrist_pos_init,
            "quat": wrist_quat_init,
        },
        "top": {
            "prim": top_prim,
            "save_tag": "third_person",
            "pos": top_pos_init,
            "quat": top_quat_init,
        },
    }
    active = "wrist"
    pos_step = 0.005
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
        carb.input.KeyboardInput.B: (4, -1),
    }

    def _update():
        cam = cameras[active]
        _set_prim_matrix(cam["prim"], cam["pos"], cam["quat"])

    def _rotate(ax, ay, az, sign):
        """Apply an incremental world-space rotation to the active camera."""
        cam = cameras[active]
        dq = axis_angle_to_quat(ax, ay, az, sign * rot_step)
        cam["quat"] = quat_normalize(quat_mul(dq, cam["quat"]))  # world-space: pre-multiply
        _update()
        _print_state()
        _save_frame(cam["save_tag"])

    def _save_frame(tag):
        images = env.update_cameras()
        rgb = images[tag][0].cpu().numpy()
        path = f"{tag}_cam_check.png"
        Image.fromarray(rgb[..., :3]).save(path)
        print(f"[SAVE] {path}")

    def _print_state():
        cam = cameras[active]
        w, x, y, z = cam["quat"]
        p = cam["pos"]
        print(f"[{active.upper()}] pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})  "
              f"quat=({w:.4f}, {x:.4f}, {y:.4f}, {z:.4f})")

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

        # Rotation — incremental world-space, no gimbal lock
        elif k == carb.input.KeyboardInput.I:  _rotate(1, 0, 0, +1)
        elif k == carb.input.KeyboardInput.K:  _rotate(1, 0, 0, -1)
        elif k == carb.input.KeyboardInput.O:  _rotate(0, 1, 0, +1)
        elif k == carb.input.KeyboardInput.L:  _rotate(0, 1, 0, -1)
        elif k == carb.input.KeyboardInput.F:  _rotate(0, 0, 1, +1)
        elif k == carb.input.KeyboardInput.H:  _rotate(0, 0, 1, -1)

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
    print("Rotation:    I/K=worldX  O/L=worldY  F/H=worldZ  |  9/0=rot step")
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

    print("\n── Paste into pick_place_env_cfg.py / _create_camera_sensors ──")
    for name, cam in cameras.items():
        w, x, y, z = cam["quat"]
        p = cam["pos"]
        print(f"\n# {name}")
        if name == "wrist":
            print(f"camera_wrist_pos: tuple = {tuple(round(v, 4) for v in p)}")
        else:
            print(f"camera_third_person_eye: tuple = {tuple(round(v, 4) for v in p)}")
        print(f"rot=({w:.4f}, {x:.4f}, {y:.4f}, {z:.4f})")


if __name__ == "__main__":
    main()

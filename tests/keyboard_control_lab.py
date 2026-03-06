#!/usr/bin/env python3
"""
Keyboard control script for Isaac Lab pick-and-place environment.

Joint Controls:
  Q/A: Shoulder Pan
  W/S: Shoulder Lift
  E/D: Elbow Flex
  R/V: Wrist Flex
  T/G: Wrist Roll
  Y/U: Gripper (open/close)

Sphere Offset Controls:
  TAB:  Toggle active sphere (Green=fixed / Red=moving)
  1/2:  Adjust X offset (-/+)
  3/4:  Adjust Y offset (-/+)
  5/6:  Adjust Z offset (-/+)
  7/8:  Decrease/Increase step size
  P:    Print current offsets

  ESC: Exit
"""
import argparse
import sys
import os
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Keyboard control for Isaac Lab environment")
    parser.add_argument("--headless", action="store_true", help="Run headless (no window)")
    args = parser.parse_args()
    
    # Isaac Lab requires AppLauncher before any other imports
    from isaaclab.app import AppLauncher
    
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    
    # Now import Isaac Lab modules
    import torch
    import carb
    import omni.appwindow
    
    from lab.pick_place_env import PickPlaceEnv
    from lab.pick_place_env_cfg import PickPlaceEnvCfg
    
    print("=" * 60)
    print("Isaac Lab Keyboard Control - Detector Test")
    print("=" * 60)
    
    # Create environment with 1 env for testing
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 3600.0  # 1 hour - effectively no timeout for keyboard control
    env = PickPlaceEnv(cfg)
    
    # Reset environment
    obs_dict, info = env.reset()
    print("[INFO] Environment ready")
    
    # Get initial joint positions
    joint_targets = env.robot.data.joint_pos[0].clone()
    
    # Movement parameters
    STEP_SIZE = 0.05  # radians per keypress
    GRIPPER_STEP = 0.1
    
    # Sphere offset tuning state
    sphere_step = 0.005  # meters per keypress for sphere offset
    active_sphere = "green"  # "green" (fixed/gripper) or "red" (moving/jaw)
    
    # State tracking
    input_state = {"is_running": True}
    
    # Detector state tracking (to print only on transitions)
    detector_state = {
        "reached": False,
        "grasped": False,
        "droppable": False,
        "in_cup": False,
    }
    
    # Key mapping: key -> (joint_index, direction)
    mapping = {
        carb.input.KeyboardInput.Q: (0, 1),   # shoulder_pan
        carb.input.KeyboardInput.A: (0, -1),
        carb.input.KeyboardInput.W: (1, 1),   # shoulder_lift
        carb.input.KeyboardInput.S: (1, -1),
        carb.input.KeyboardInput.E: (2, 1),   # elbow_flex
        carb.input.KeyboardInput.D: (2, -1),
        carb.input.KeyboardInput.R: (3, 1),   # wrist_flex
        carb.input.KeyboardInput.V: (3, -1),
        carb.input.KeyboardInput.T: (4, 1),   # wrist_roll
        carb.input.KeyboardInput.G: (4, -1),
    }
    
    # Gripper limits
    GRIPPER_OPEN = 1.5
    GRIPPER_CLOSE = -0.2
    
    def on_keyboard_event(event):
        nonlocal joint_targets, active_sphere, sphere_step
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.ESCAPE:
                input_state["is_running"] = False
            elif event.input == carb.input.KeyboardInput.Y:
                # Continuous: Gripper incremental open
                joint_targets[5] = torch.clamp(joint_targets[5] + GRIPPER_STEP, float(GRIPPER_CLOSE), float(GRIPPER_OPEN))
                print(f"[GRIPPER] Opening -> target={joint_targets[5]:.2f}")
            elif event.input == carb.input.KeyboardInput.U:
                # Discrete: Gripper snap close
                joint_targets[5] = GRIPPER_CLOSE
                print(f"[GRIPPER] Snap Close -> target={GRIPPER_CLOSE}")
            elif event.input in mapping:
                idx, direction = mapping[event.input]
                joint_targets[idx] += direction * STEP_SIZE
            
            # --- Sphere offset controls ---
            elif event.input == carb.input.KeyboardInput.TAB:
                active_sphere = "red" if active_sphere == "green" else "green"
                label = "Green (Fixed)" if active_sphere == "green" else "Red (Moving)"
                print(f"[SPHERE] Active: {label}")
            elif event.input == carb.input.KeyboardInput.P:
                g = env.tip_offset_gripper.cpu().numpy()
                j = env.tip_offset_jaw.cpu().numpy()
                print(f"[OFFSETS] Green: X={g[0]:.4f} Y={g[1]:.4f} Z={g[2]:.4f}")
                print(f"[OFFSETS] Red:   X={j[0]:.4f} Y={j[1]:.4f} Z={j[2]:.4f}")
            elif event.input in (carb.input.KeyboardInput.KEY_1, carb.input.KeyboardInput.KEY_2,
                                 carb.input.KeyboardInput.KEY_3, carb.input.KeyboardInput.KEY_4,
                                 carb.input.KeyboardInput.KEY_5, carb.input.KeyboardInput.KEY_6):
                # Select which offset tensor to modify
                offset = env.tip_offset_gripper if active_sphere == "green" else env.tip_offset_jaw
                axis_map = {
                    carb.input.KeyboardInput.KEY_1: (0, -1),  # X-
                    carb.input.KeyboardInput.KEY_2: (0, +1),  # X+
                    carb.input.KeyboardInput.KEY_3: (1, -1),  # Y-
                    carb.input.KeyboardInput.KEY_4: (1, +1),  # Y+
                    carb.input.KeyboardInput.KEY_5: (2, -1),  # Z-
                    carb.input.KeyboardInput.KEY_6: (2, +1),  # Z+
                }
                axis, direction = axis_map[event.input]
                offset[axis] += direction * sphere_step
                axis_name = "XYZ"[axis]
                label = "Green" if active_sphere == "green" else "Red"
                print(f"[SPHERE {label}] {axis_name}={offset[axis]:.4f}  (step={sphere_step:.4f})")
            elif event.input == carb.input.KeyboardInput.KEY_7:
                sphere_step = max(0.001, sphere_step / 2.0)
                print(f"[SPHERE] Step size: {sphere_step:.4f}")
            elif event.input == carb.input.KeyboardInput.KEY_8:
                sphere_step = min(0.05, sphere_step * 2.0)
                print(f"[SPHERE] Step size: {sphere_step:.4f}")
        return True
    
    # Register keyboard listener
    input_interface = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    _sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print("\nJoint Controls:")
    print("  Shoulder Pan:  Q / A")
    print("  Shoulder Lift: W / S")
    print("  Elbow Flex:    E / D")
    print("  Wrist Flex:    R / V")
    print("  Wrist Roll:    T / G")
    print("  Gripper:       Y (open) / U (close)")
    print("\nSphere Controls:")
    print("  TAB:  Toggle active sphere (Green/Red)")
    print("  1/2:  X offset  -/+")
    print("  3/4:  Y offset  -/+")
    print("  5/6:  Z offset  -/+")
    print("  7/8:  Step size -/+")
    print("  P:    Print offsets")
    print("\n  ESC to Exit")
    print("-" * 50)
    
    REACH_THRESHOLD = 0.10  # meters
    warmup_frames = 60  # Skip detector checks during scene stabilization
    frame_count = 0
    last_print_time = 0.0  # For 1-second rate-limiting of metrics output
    
    # Track minimum distance and ft_rew sum during pre-grasp phase
    min_dl = float('inf')
    min_dr = float('inf')
    ft_rew_sum = 0.0
    
    try:
        while input_state["is_running"] and simulation_app.is_running():
            # Create action tensor (delta from current to target)
            current_pos = env.robot.data.joint_pos[0]
            
            # Correct action scaling: env.step(action) does current_pos + action * action_scale
            # So action = (target_pos - current_pos) / action_scale
            # We also clamp to [-1, 1] to stay within standard action bounds
            delta_action = (joint_targets - current_pos) / env.cfg.action_scale
            delta_action = torch.clamp(delta_action, -1.0, 1.0).unsqueeze(0)
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(delta_action)
            
            # Reset local targets if environment resets to prevent drift
            if terminated.item() or truncated.item():
                joint_targets = env.robot.data.default_joint_pos[0].clone()
                print("[INFO] Environment reset - targets synchronized to home.")

            # Update simulation app to process events and render
            simulation_app.update()
            
            frame_count += 1
            if frame_count < warmup_frames:
                continue  # Skip detector checks during warmup
            
            # Get detector states from task_state
            task_state = info.get("task_state", {})
            
            def _extract_val(x):
                if hasattr(x, "item"):
                    return x[0].item() if getattr(x, "dim", lambda: 0)() > 0 else x.item()
                return float(x)
            
            # Get distance for reach detection
            gripper_cube_dist = task_state.get("gripper_cube_distance")
            if gripper_cube_dist is not None:
                # Handle tensor or scalar
                if hasattr(gripper_cube_dist, "item"):
                    dist = gripper_cube_dist[0].item() if gripper_cube_dist.dim() > 0 else gripper_cube_dist.item()
                else:
                    dist = gripper_cube_dist
                
                reached_now = dist < REACH_THRESHOLD
                if reached_now and not detector_state["reached"]:
                    print(f"\n[REACHED] Gripper close to cube (dist={dist:.3f}m < {REACH_THRESHOLD}m)")
                elif not reached_now and detector_state["reached"]:
                    print(f"\n[REACHED OFF] Gripper moved away (dist={dist:.3f}m)")
                detector_state["reached"] = reached_now

            # Zone-entry detection for fingertips (left/right face zones)
            left_ok = task_state.get("left_zone_ok")
            right_ok = task_state.get("right_zone_ok")
            
            if left_ok is not None:
                val = left_ok[0].item() if left_ok.dim() > 0 else left_ok.item()
                if val and not detector_state.get("left_zone_ok", False):
                    print("\n[LEFT ZONE] ✅ Fixed jaw (Green zone) ENTERED left face zone")
                elif not val and detector_state.get("left_zone_ok", False):
                    print("\n[LEFT ZONE] ❌ Fixed jaw (Green zone) LEFT left face zone")
                detector_state["left_zone_ok"] = val
            
            if right_ok is not None:
                val = right_ok[0].item() if right_ok.dim() > 0 else right_ok.item()
                if val and not detector_state.get("right_zone_ok", False):
                    print("\n[RIGHT ZONE] ✅ Moving jaw (Red zone) ENTERED right face zone")
                elif not val and detector_state.get("right_zone_ok", False):
                    print("\n[RIGHT ZONE] ❌ Moving jaw (Red zone) LEFT right face zone")
                detector_state["right_zone_ok"] = val
            
            # Check grasped
            is_grasped = task_state.get("is_grasped")
            if is_grasped is not None:
                if hasattr(is_grasped, "item"):
                    grasped = is_grasped[0].item() if is_grasped.dim() > 0 else is_grasped.item()
                else:
                    grasped = bool(is_grasped)
                
                if grasped and not detector_state["grasped"]:
                    # Print cube-local debug info right before grasp
                    dbg_ql = task_state.get("dbg_q_left")
                    dbg_qr = task_state.get("dbg_q_right")
                    dbg_ax = task_state.get("dbg_axis_local")
                    dbg_sl = task_state.get("dbg_sign_left")
                    if dbg_ql is not None:
                        ql = dbg_ql[0].cpu().numpy()
                        qr = dbg_qr[0].cpu().numpy()
                        ax = dbg_ax[0].cpu().numpy()
                        sl = dbg_sl[0].item()
                        sr = -sl
                        print(f"\n--- PRE-GRASP DEBUG (cube local) ---")
                        print(f"  q_left  (fixed jaw): [{ql[0]:+.4f}, {ql[1]:+.4f}, {ql[2]:+.4f}]")
                        print(f"  q_right (moving jaw): [{qr[0]:+.4f}, {qr[1]:+.4f}, {qr[2]:+.4f}]")
                        print(f"  |q_left|:  [{abs(ql[0]):.4f}, {abs(ql[1]):.4f}, {abs(ql[2]):.4f}]")
                        print(f"  |q_right|: [{abs(qr[0]):.4f}, {abs(qr[1]):.4f}, {abs(qr[2]):.4f}]")
                        print(f"  axis_local: [{ax[0]:.0f}, {ax[1]:.0f}, {ax[2]:.0f}]")
                        
                        # Show raw distances to both faces for each tip
                        dl_p = _extract_val(task_state.get("d_L_pos"))
                        dl_n = _extract_val(task_state.get("d_L_neg"))
                        dr_p = _extract_val(task_state.get("d_R_pos"))
                        dr_n = _extract_val(task_state.get("d_R_neg"))
                        print(f"  EucDist Fixed Jaw  to +/-face centers: [+Face={dl_p:.4f}, -Face={dl_n:.4f}]")
                        print(f"  EucDist Moving Jaw to +/-face centers: [+Face={dr_p:.4f}, -Face={dr_n:.4f}]")
                        print(f"---")
                    print("[GRASPED] Cube is being held!")
                    print(f"[GRASPED] Total ft_rew accumulated this approach: {ft_rew_sum:.4f}")
                elif not grasped and detector_state["grasped"]:
                    cube_z = env.cube.data.root_pos_w[0, 2].item()
                    print(f"\n[GRASPED OFF] Cube released (cube_z={cube_z:.3f}m)")
                    # Reset minimums and ft_rew sum for next grasp attempt
                    min_dl = float('inf')
                    min_dr = float('inf')
                    ft_rew_sum = 0.0
                detector_state["grasped"] = grasped
            
            # Check droppable (cube over cup)
            is_droppable = task_state.get("is_droppable")
            if is_droppable is not None:
                if hasattr(is_droppable, "item"):
                    droppable = is_droppable[0].item() if is_droppable.dim() > 0 else is_droppable.item()
                else:
                    droppable = bool(is_droppable)
                
                if droppable and not detector_state["droppable"]:
                    print("\n[DROPPABLE] Cube is over the cup!")
                elif not droppable and detector_state["droppable"]:
                    print("\n[DROPPABLE OFF] Cube moved away from cup")
                detector_state["droppable"] = droppable
            
            # Check in_cup
            is_in_cup = task_state.get("is_in_cup")
            if is_in_cup is not None:
                if hasattr(is_in_cup, "item"):
                    in_cup = is_in_cup[0].item() if is_in_cup.dim() > 0 else is_in_cup.item()
                else:
                    in_cup = bool(is_in_cup)
                
                if in_cup and not detector_state["in_cup"]:
                    print("\n[IN CUP] *** SUCCESS! Cube is in the cup! ***")
                elif not in_cup and detector_state["in_cup"]:
                    print("\n[IN CUP OFF] Cube fell out of cup")
                detector_state["in_cup"] = in_cup
            
            # Print distance + fingertip reward metrics once per second
            dl = task_state.get("d_left")
            dr = task_state.get("d_right")
            rg = task_state.get("reach_gate")
            ft_rew = task_state.get("fingertip_step_reward")
            
            if dl is not None and dr is not None:
                dl_val  = _extract_val(dl)
                dr_val  = _extract_val(dr)
                rg_val  = _extract_val(rg) if rg is not None else 0.0
                gc_val  = _extract_val(gripper_cube_dist) if gripper_cube_dist is not None else -1.0
                rew_val = _extract_val(ft_rew) if ft_rew is not None else 0.0
                
                # Accumulate ft_rew every frame (pre-grasp only)
                if not detector_state.get("grasped", False):
                    ft_rew_sum += rew_val
                
                # Update minimums only when not grasped
                if not detector_state.get("grasped", False):
                    min_dl = min(min_dl, dl_val)
                    min_dr = min(min_dr, dr_val)
                
                now = time.time()
                if now - last_print_time >= 1.0:
                    last_print_time = now
                    dl_print  = f"{min_dl:.4f}" if min_dl != float('inf') else "inf"
                    dr_print  = f"{min_dr:.4f}" if min_dr != float('inf') else "inf"
                    grasped_str = "GRASPED" if detector_state.get("grasped") else "pre-grasp"
                    print(f"[{grasped_str}]  dist={gc_val:.4f}  "
                          f"fcL={dl_val:.4f} (min={dl_print})  "
                          f"fcR={dr_val:.4f} (min={dr_print})  "
                          f"RG={rg_val:.3f}  "
                          f"ft_rew={rew_val:.5f}")

                    # --- Grasp condition breakdown (only useful pre-grasp) ---
                    if not detector_state.get("grasped", False):
                        gd = env.grasp_detector
                        left_ok  = env._fixed_tip_in_left_zone[0].item()
                        right_ok = env._moving_tip_in_right_zone[0].item()
                        f_frames = gd.in_zone_frames[0].item()
                        print(f"  [GRASP DBG] "
                              f"L_in_zone={left_ok}  "
                              f"R_in_zone={right_ok}  "
                              f"→ frames={f_frames}/{env.cfg.grasp_frames_to_grasp}")
                        # Print zone sub-components for diagnosis
                        d_fix = _extract_val(task_state.get("dbg_d_fixed"))
                        d_mov = _extract_val(task_state.get("dbg_d_moving"))
                        z_r   = _extract_val(task_state.get("dbg_zone_radius"))
                        print(f"  [ZONE DBG] zone_radius={z_r:.3f}  "
                              f"fcL={d_fix:.4f} ({'IN' if d_fix < z_r else 'OUT'})  "
                              f"fcR={d_mov:.4f} ({'IN' if d_mov < z_r else 'OUT'})")
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[INFO] Shutting down...")
        try:
            input_interface.unsubscribe_to_keyboard_events(keyboard, _sub)
        except Exception:
            pass
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

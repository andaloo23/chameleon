#!/usr/bin/env python3
"""
Keyboard control script for Isaac Lab pick-and-place environment.

Tests the various detectors:
- Reach detector (gripper close to cube)
- Grasp detector
- Droppable detector (cube over cup)
- In-cup detector

Controls:
  Q/A: Shoulder Pan
  W/S: Shoulder Lift
  E/D: Elbow Flex
  R/V: Wrist Flex
  T/G: Wrist Roll
  Y/U: Gripper (open/close)
  ESC: Exit
"""
import argparse
import sys
import os

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
        nonlocal joint_targets
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
                step = STEP_SIZE
                joint_targets[idx] += direction * step
        return True
    
    # Register keyboard listener
    input_interface = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    _sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    
    print("\nControls:")
    print("  Shoulder Pan:  Q / A")
    print("  Shoulder Lift: W / S")
    print("  Elbow Flex:    E / D")
    print("  Wrist Flex:    R / V")
    print("  Wrist Roll:    T / G")
    print("  Gripper:       Y (open) / U (close)")
    print("  Press ESC to Exit")
    print("\nDetector events will print below:")
    print("-" * 40)
    
    REACH_THRESHOLD = 0.10  # meters
    warmup_frames = 60  # Skip detector checks during scene stabilization
    frame_count = 0
    
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
            if terminated[0] or truncated[0]:
                joint_targets = env.robot.data.default_joint_pos[0].clone()
                print("[INFO] Environment reset - targets synchronized to home.")

            # Update simulation app to process events and render
            simulation_app.update()
            
            frame_count += 1
            if frame_count < warmup_frames:
                continue  # Skip detector checks during warmup
            
            # Get detector states from task_state
            task_state = info.get("task_state", {})
            
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
                    print(f"[REACHED] Gripper close to cube (dist={dist:.3f}m < {REACH_THRESHOLD}m)")
                elif not reached_now and detector_state["reached"]:
                    print(f"[REACHED OFF] Gripper moved away (dist={dist:.3f}m)")
                detector_state["reached"] = reached_now

            # Per-jaw tip distance debug
            gripper_tip = task_state.get("gripper_tip_pos")
            jaw_tip = task_state.get("jaw_tip_pos")
            cube_pos = task_state.get("cube_pos")
            
            if gripper_tip is not None and jaw_tip is not None and cube_pos is not None:
                # Calculate XYZ deltas from TIPS to cube center
                # Left (Fixed)
                gl_xyz = (gripper_tip[0] - cube_pos[0]).cpu().numpy()
                # Right (Moving)
                gr_xyz = (jaw_tip[0] - cube_pos[0]).cpu().numpy()
                
                if frame_count % 10 == 0:
                    print(f"Dist L-Tip | X:{gl_xyz[0]:.3f} Y:{gl_xyz[1]:.3f} Z:{gl_xyz[2]:.3f} | "
                          f"R-Tip | X:{gr_xyz[0]:.3f} Y:{gr_xyz[1]:.3f} Z:{gr_xyz[2]:.3f}")
            
            # Check grasped
            is_grasped = task_state.get("is_grasped")
            if is_grasped is not None:
                if hasattr(is_grasped, "item"):
                    grasped = is_grasped[0].item() if is_grasped.dim() > 0 else is_grasped.item()
                else:
                    grasped = bool(is_grasped)
                
                if grasped and not detector_state["grasped"]:
                    print("[GRASPED] Cube is being held!")
                elif not grasped and detector_state["grasped"]:
                    cube_z = env.cube.data.root_pos_w[0, 2].item()
                    print(f"[GRASPED OFF] Cube released (cube_z={cube_z:.3f}m)")
                detector_state["grasped"] = grasped
            
            # Check droppable (cube over cup)
            is_droppable = task_state.get("is_droppable")
            if is_droppable is not None:
                if hasattr(is_droppable, "item"):
                    droppable = is_droppable[0].item() if is_droppable.dim() > 0 else is_droppable.item()
                else:
                    droppable = bool(is_droppable)
                
                if droppable and not detector_state["droppable"]:
                    print("[DROPPABLE] Cube is over the cup!")
                elif not droppable and detector_state["droppable"]:
                    print("[DROPPABLE OFF] Cube moved away from cup")
                detector_state["droppable"] = droppable
            
            # Check in_cup
            is_in_cup = task_state.get("is_in_cup")
            if is_in_cup is not None:
                if hasattr(is_in_cup, "item"):
                    in_cup = is_in_cup[0].item() if is_in_cup.dim() > 0 else is_in_cup.item()
                else:
                    in_cup = bool(is_in_cup)
                
                if in_cup and not detector_state["in_cup"]:
                    print("[IN CUP] *** SUCCESS! Cube is in the cup! ***")
                elif not in_cup and detector_state["in_cup"]:
                    print("[IN CUP OFF] Cube fell out of cup")
                detector_state["in_cup"] = in_cup
            
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

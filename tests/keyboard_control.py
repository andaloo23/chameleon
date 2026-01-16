import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from load_scene import IsaacPickPlaceEnv

def main():
    print("--- Starting Keyboard Control for SO-100 ---")
    
    # Initialize environment FIRST to ensure SimulationApp is running
    # This is required for omni.* and carb imports to work
    env = IsaacPickPlaceEnv(headless=False)
    
    import carb
    import omni.appwindow
    
    # Reset to starting state
    print("[INFO] Resetting environment...")
    obs = env.reset()
    
    # Wait a moment for scene to stabilize
    for _ in range(60): env.world.step(render=True)
    
    # Get initial joint positions
    joint_positions = np.array(env.robot_articulation.get_joint_positions(), dtype=float)
    
    # Movement parameters
    STEP_SIZE = np.deg2rad(2.0)
    GRIPPER_STEP = 0.05
    
    # Key mapping
    # joint_idx -> (plus_key, minus_key)
    # 0: shoulder_pan (Q/A)
    # 1: shoulder_lift (W/S)
    # 2: elbow_flex (E/D)
    # 3: wrist_flex (R/F)
    # 4: wrist_roll (T/G)
    # 5: gripper (Y/H)
    
    input_state = {
        "is_running": True
    }
    
    mapping = {
        carb.input.KeyboardInput.Q: (0, 1),   # index, direction
        carb.input.KeyboardInput.A: (0, -1),
        carb.input.KeyboardInput.W: (1, 1),
        carb.input.KeyboardInput.S: (1, -1),
        carb.input.KeyboardInput.E: (2, 1),
        carb.input.KeyboardInput.D: (2, -1),
        carb.input.KeyboardInput.R: (3, 1),
        carb.input.KeyboardInput.V: (3, -1),
        carb.input.KeyboardInput.T: (4, 1),
        carb.input.KeyboardInput.G: (4, -1),
        carb.input.KeyboardInput.Y: (5, 1),
        carb.input.KeyboardInput.U: (5, -1),
    }

    def on_keyboard_event(event):
        nonlocal joint_positions
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.ESCAPE:
                input_state["is_running"] = False
            elif event.input == carb.input.KeyboardInput.U:
                # Close gripper with continuous pressure (target beyond physical limit)
                joint_positions[5] = -0.5
            elif event.input in mapping:
                idx, direction = mapping[event.input]
                step = GRIPPER_STEP if idx == 5 else STEP_SIZE
                joint_positions[idx] += direction * step
                # Clip immediately to prevent invalid targets
                joint_positions = env._clip_action(joint_positions)
        return True

    # Register keyboard listener
    input_interface = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    _sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("\nControls (Discrete Steps):")
    print("  Shoulder Pan:  Q / A")
    print("  Shoulder Lift: W / S")
    print("  Elbow Flex:    E / D")
    print("  Wrist Flex:    R / V")
    print("  Wrist Roll:    T / G")
    print("  Gripper:       Y / U")
    print("  Press ESC to Exit")

    try:
        while input_state["is_running"]:
            # Step simulation with current targets
            # This maintains the robot's pose and updates physics/rendering
            env.step(joint_positions, render=True)
            
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[INFO] Shutting down...")
        try:
            input_interface.unsubscribe_to_keyboard_events(keyboard, _sub)
        except Exception: pass
        
        # Explicit shutdown sequence to prevent crashes
        try:
            env.shutdown()
        except Exception as e:
            print(f"[WARN] Shutdown error: {e}")

if __name__ == "__main__":
    main()

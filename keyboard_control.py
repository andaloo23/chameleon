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
        "plus": [False] * 6,
        "minus": [False] * 6
    }
    
    mapping = {
        carb.input.KeyboardInput.Q: (0, "plus"),
        carb.input.KeyboardInput.A: (0, "minus"),
        carb.input.KeyboardInput.W: (1, "plus"),
        carb.input.KeyboardInput.S: (1, "minus"),
        carb.input.KeyboardInput.E: (2, "plus"),
        carb.input.KeyboardInput.D: (2, "minus"),
        carb.input.KeyboardInput.R: (3, "plus"),
        carb.input.KeyboardInput.F: (3, "minus"),
        carb.input.KeyboardInput.I: (3, "plus"), # Alternative for F
        carb.input.KeyboardInput.K: (3, "minus"), # Alternative for F
        carb.input.KeyboardInput.T: (4, "plus"),
        carb.input.KeyboardInput.G: (4, "minus"),
        carb.input.KeyboardInput.Y: (5, "plus"),
        carb.input.KeyboardInput.H: (5, "minus"),
    }

    def on_keyboard_event(event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in mapping:
                idx, field = mapping[event.input]
                input_state[field][idx] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in mapping:
                idx, field = mapping[event.input]
                input_state[field][idx] = False
        return True

    # Register keyboard listener
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    _sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("\nControls:")
    print("  Shoulder Pan:  Q / A")
    print("  Shoulder Lift: W / S")
    print("  Elbow Flex:    E / D")
    print("  Wrist Flex:    R / F")
    print("  Wrist Roll:    T / G")
    print("  Gripper:       Y / H")
    print("  Press ESC to Exit")

    try:
        is_running = True
        while is_running:
            # Check for ESC to exit
            if keyboard.get_key_value(carb.input.KeyboardInput.ESCAPE):
                is_running = False
                break
            
            changed = False
            for i in range(6):
                step = GRIPPER_STEP if i == 5 else STEP_SIZE
                if input_state["plus"][i]:
                    joint_positions[i] += step
                    changed = True
                if input_state["minus"][i]:
                    joint_positions[i] -= step
                    changed = True
            
            # Step simulation
            # Use env.step to handle kinematics, physics, and rendering
            env.step(joint_positions, render=True)
            
            # Update local current positions to prevent drift if physics restricts movement
            # Actually, we want to accumulate 'targets' so we can override physics
            # But let's stay within limits
            joint_positions = env._clip_action(joint_positions)
            
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[INFO] Shutting down...")
        carb.input.acquire_input_interface().unsubscribe_to_keyboard_events(keyboard, _sub)
        env.close()

if __name__ == "__main__":
    main()

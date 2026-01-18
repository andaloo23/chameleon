import numpy as np
from load_scene import IsaacPickPlaceEnv
import time

def main():
    print("--- Starting Pick-and-Place Kinematics Verification ---")
    
    # Initialize environment
    # Keep it windowed so we can see the action
    env = IsaacPickPlaceEnv(headless=False, capture_images=True)
    
    # Reset to starting state
    print("[INFO] Resetting environment...")
    obs = env.reset()
    
    # Wait a moment for scene to stabilize
    for _ in range(60): env.world.step(render=True)
    
    # Execute the automated pick-up sequence
    try:
        env.pick_up_cube()
        
        # Hold the lifted position for a few seconds to verify
        print("[INFO] Holding object for 5 seconds...")
        for _ in range(300): env.world.step(render=True)
        
    except Exception as e:
        print(f"[ERROR] Pick-up sequence failed: {e}")
    finally:
        print("[INFO] Shutting down...")
        env.close()

if __name__ == "__main__":
    main()

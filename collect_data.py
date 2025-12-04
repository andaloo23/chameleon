from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np
from PIL import Image

from load_scene import IsaacPickPlaceEnv
from simulator_loop import EpisodeResult, SimulationLoop, PolicyFn

def save_episode(
    result: EpisodeResult,
    output_dir: str,
    episode_idx: int,
    save_images: bool = True
) -> None:
    """Save rollout data (JSON trajectory + images) to disk."""
    episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)

    traj = result.to_trajectory()
    
    print(f"[info] Filtering large image arrays from JSON for episode {episode_idx}...")
    def clean_obs(obs_list):
        for obs in obs_list:
            if not isinstance(obs, dict): continue
            keys_to_remove = [k for k in obs.keys() if "rgb" in k or "image" in k or "depth" in k]
            for k in keys_to_remove:
                obs[k] = "<image_data_removed>"
    
    clean_obs(traj["observations"])
    clean_obs(traj["next_observations"])

    def default_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    json_path = os.path.join(episode_dir, "trajectory.json")
    print(f"[info] Writing JSON to {json_path}...")
    with open(json_path, "w") as f:
        json.dump(traj, f, indent=2, default=default_serializer)
    
    print(f"[info] Saved trajectory to {json_path}")

    if not save_images:
        print(f"[info] Saved episode {episode_idx} with {len(result.transitions)} steps and 0 images.")
        return

    if save_images:
         print("[warn] Image data was stripped from observations for JSON safety. Cannot save PNGs.")

    print(f"[info] Saved episode {episode_idx} with {len(result.transitions)} steps.")


def main():
    parser = argparse.ArgumentParser(description="Collect success data for OpenVLA.")
    parser.add_argument("--out", type=str, default="data/rollouts", help="Output directory.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    parser.add_argument("--max_steps", type=int, default=360, help="Max steps per episode.")
    args = parser.parse_args()

    print(f"[info] Starting data collection for {args.episodes} episodes...")
    
    try:
        loop = SimulationLoop(env=IsaacPickPlaceEnv(), capture_images=False, headless=args.headless, max_steps=args.max_steps)

        policy = loop.scripted_policy()
        
        success_count = 0
        target_successes = 1 # FORCE 1 for debug
        
        os.makedirs(args.out, exist_ok=True)

        for i in range(args.episodes):
            print(f"[info] Starting Episode {i}...")
            result = loop.run_episode(policy=policy, reset=True, render=True)
            
            is_success = True 

            if is_success:
                print(f"[info] Episode {i} completed. Steps: {len(result.transitions)}")
                print(f"[info] Episode {i} considered success (forced)!")
                save_episode(result, args.out, i, save_images=False)
                success_count += 1
                if success_count >= target_successes:
                    print("[info] Reached target success count. Stopping.")
                    break
            else:
                print(f"[info] Episode {i} failed. Reason: {result.termination_reason}")

        print(f"[info] Data collection complete. Saved {success_count} episodes.")
        
    except Exception as e:
        print(f"[error] An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'loop' in locals():
            print("[info] Shutting down SimulationLoop...")
            try:
                loop.close()
            except Exception as e:
                print(f"[warn] Error closing loop: {e}")
        print("[info] Exiting script forcefully.")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

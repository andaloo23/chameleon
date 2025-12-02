
import os
import time
import json
import numpy as np
from simulator_loop import SimulationLoop, EpisodeResult
from image_utils import write_png

DATA_DIR = "collected_data"

def save_episode(episode_idx: int, result: EpisodeResult):
    """Save the episode data to disk."""
    if not result.success:
        print(f"[info] Episode {episode_idx} failed, skipping save.")
        return

    ep_dir = os.path.join(DATA_DIR, f"episode_{episode_idx}")
    os.makedirs(ep_dir, exist_ok=True)

    traj_data = {
        "actions": [],
        "rewards": [],
        "dones": [],
        "infos": [],
        "joint_positions": [],
        "images": []
    }

    print(f"[info] Saving episode {episode_idx} with {len(result.transitions)} steps...")

    for t_idx, transition in enumerate(result.transitions):
        # Save Images
        obs = transition.observation
        step_image_info = {}
        
        for key, value in obs.items():
            if key.endswith("_rgb") and value is not None:
                img_filename = f"step_{t_idx}_{key}.png"
                img_path = os.path.join(ep_dir, img_filename)
                write_png(img_path, value)
                step_image_info[key] = img_filename

        # Append data
        traj_data["images"].append(step_image_info)
        traj_data["actions"].append(transition.action.tolist())
        traj_data["rewards"].append(transition.reward)
        traj_data["dones"].append(transition.done)
        
        if "joint_positions" in obs:
             traj_data["joint_positions"].append(obs["joint_positions"].tolist())

    # Save metadata
    with open(os.path.join(ep_dir, "trajectory.json"), "w") as f:
        json.dump(traj_data, f, indent=2)

def collect():
    os.makedirs(DATA_DIR, exist_ok=True)

    with SimulationLoop(max_steps=360, headless=True, capture_images=True, image_interval=1) as loop:
        policy_fn = loop.scripted_policy()

        success_count = 0
        target_successes = 5

        ep_idx = 0
        while success_count < target_successes:
            print(f"--- Starting Episode {ep_idx} ---")
            result = loop.run_episode(policy=policy_fn)
            
            if result.success:
                save_episode(success_count, result)
                success_count += 1
                print(f"--- Episode {ep_idx} SUCCESS ({success_count}/{target_successes}) ---")
            else:
                print(f"--- Episode {ep_idx} FAILED ---")
            
            ep_idx += 1

if __name__ == "__main__":
    collect()


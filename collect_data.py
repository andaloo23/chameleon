
import os
import time
import json
import numpy as np
from simulator_loop import SimulationLoop, EpisodeResult
from image_utils import write_png

DATA_DIR = "collected_data"

def save_episode(episode_idx: int, result: EpisodeResult):
    """Save the episode data to disk."""
    # DEBUG: Save EVERYTHING to diagnose failure
    is_success = True 
    
    if not is_success:
        print(f"[info] Episode {episode_idx} failed (not lifted), skipping save.")
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
    if result.termination_reason:
        print(f"[info] Termination reason: {result.termination_reason}")

    for t_idx, transition in enumerate(result.transitions):
        traj_data["actions"].append(transition.action.tolist())
        traj_data["rewards"].append(transition.reward)
        traj_data["dones"].append(transition.done)

        if transition.info:
             clean_info = {}
             for k, v in transition.info.items():
                 if isinstance(v, dict):
                     clean_info[k] = v
                 elif isinstance(v, (bool, int, float, str)):
                     clean_info[k] = v
                 elif isinstance(v, (list, tuple)):
                     clean_info[k] = v
             traj_data["infos"].append(clean_info)
        else:
             traj_data["infos"].append({})
        
        obs = transition.observation
        if "joint_positions" in obs:
             traj_data["joint_positions"].append(obs["joint_positions"].tolist())
        
        traj_data["images"].append({})

    json_path = os.path.join(ep_dir, "trajectory.json")
    with open(json_path, "w") as f:
        json.dump(traj_data, f, indent=2)

    for t_idx, transition in enumerate(result.transitions):
        obs = transition.observation
        step_image_info = {}
        
        for key, value in obs.items():
            if key.endswith("_rgb") and value is not None:
                try:
                    img_filename = f"step_{t_idx}_{key}.png"
                    img_path = os.path.join(ep_dir, img_filename)
                    write_png(img_path, value)
                    step_image_info[key] = img_filename
                except Exception as e:
                    print(f"[warn] Failed to save image {key} at step {t_idx}: {e}")

        traj_data["images"][t_idx] = step_image_info

    with open(json_path, "w") as f:
        json.dump(traj_data, f, indent=2)

def collect():
    os.makedirs(DATA_DIR, exist_ok=True)

    with SimulationLoop(max_steps=360, headless=True, capture_images=True, image_interval=1) as loop:
        policy_fn = loop.scripted_policy()

        success_count = 0
        target_successes = 1 # Just run 1 for debug

        ep_idx = 0
        while success_count < target_successes:
            print(f"--- Starting Episode {ep_idx} ---")
            # FORCE RENDER=TRUE so cameras update even in headless mode
            result = loop.run_episode(policy=policy_fn, render=True)
            
            # ALWAYS SAVE
            save_episode(ep_idx, result)
            success_count += 1
            
            ep_idx += 1

if __name__ == "__main__":
    collect()


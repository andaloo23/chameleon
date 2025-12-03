import numpy as np
import os
import json
from simulator_loop import SimulationLoop
from load_scene import IsaacPickPlaceEnv
import image_utils
import time

def save_episode(episode_idx, steps, obs_list, action_list, reward_list, info_list, success):
    data_dir = "data/rollouts"
    os.makedirs(data_dir, exist_ok=True)
    
    episode_dir = os.path.join(data_dir, f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)
    
    traj_data = []
    for i in range(len(steps)):
        info_step = info_list[i]
        serializable_info = {}
        for k, v in info_step.items():
            if isinstance(v, np.ndarray):
                serializable_info[k] = v.tolist()
            else:
                serializable_info[k] = v
                
        traj_data.append({
            "step": steps[i],
            "action": action_list[i].tolist(),
            "reward": float(reward_list[i]),
            "info": serializable_info
        })
    
    json_path = os.path.join(episode_dir, "trajectory.json")
    with open(json_path, "w") as f:
        json.dump(traj_data, f, indent=2)
    print(f"[info] Saved trajectory to {json_path}")

    images_saved = 0
    for i, obs in enumerate(obs_list):
        step_num = steps[i]
        for key, value in obs.items():
            if "rgb" in key:
                try:
                    img_array = np.array(value)
                    
                    if len(img_array.shape) == 3 and img_array.shape[0] > 10 and img_array.shape[1] > 10:
                        image_name = f"{key}_{step_num:04d}.png"
                        image_path = os.path.join(episode_dir, image_name)
                        image_utils.write_png(image_path, img_array)
                        images_saved += 1
                    else:
                        pass
                except Exception as e:
                    pass

    print(f"[info] Saved episode {episode_idx} with {len(steps)} steps and {images_saved} images.")

def main():
    print("[info] Initializing SimulationLoop...")
    loop = SimulationLoop(capture_images=False)
    
    target_successes = 1  # Just 1 for debugging
    success_count = 0
    episode_count = 0
    
    policy = loop.scripted_policy()
    
    while success_count < target_successes:
        print(f"[info] Starting episode {episode_count}...")
        
        try:
            result = loop.run_episode(policy=policy, render=True)
        except Exception as e:
            print(f"[error] Episode {episode_count} crashed: {e}")
            raise
        
        print(f"[info] Episode {episode_count} completed. Steps: {len(result.transitions)}")

        is_success = True 
        
        if is_success:
            print(f"[info] Episode {episode_count} considered success (forced)!")
            
            steps = []
            obs_list = []
            action_list = []
            reward_list = []
            info_list = []
            
            for i, transition in enumerate(result.transitions):
                steps.append(i)
                obs_list.append(transition.observation)
                action_list.append(transition.action)
                reward_list.append(transition.reward)
                info = transition.info if transition.info else {}
                info_list.append(info)
            
            save_episode(success_count, steps, obs_list, action_list, reward_list, info_list, is_success)
            success_count += 1
        else:
            print(f"[info] Episode {episode_count} failed.")
            
        episode_count += 1
        if episode_count >= 5: 
            break
            
    print("[info] Closing SimulationLoop...")
    loop.close()

if __name__ == "__main__":
    main()

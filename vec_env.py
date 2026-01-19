"""
Vectorized environment wrapper for parallel PPO training.

Supports running N environments in parallel using multiprocessing.
Note: Isaac Sim GPU memory limits parallel instances to ~4-8 on typical hardware.
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
import queue


def _worker_process(
    env_id: int,
    pipe: mp.connection.Connection,
    headless: bool,
    max_steps: int,
    random_seed: Optional[int],
):
    """Worker process that runs a single environment instance."""
    # Import inside worker to avoid issues with Isaac Sim
    from ppo_env import PPOEnv
    
    seed = random_seed + env_id if random_seed is not None else None
    env = PPOEnv(headless=headless, max_steps=max_steps, random_seed=seed)
    
    while True:
        try:
            cmd, data = pipe.recv()
            
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                pipe.send((obs, reward, terminated, truncated, info))
                
            elif cmd == "reset":
                obs, info = env.reset()
                pipe.send((obs, info))
                
            elif cmd == "close":
                env.close()
                pipe.close()
                break
                
            elif cmd == "get_spaces":
                pipe.send((env.observation_space, env.action_space))
                
        except EOFError:
            break
        except Exception as e:
            print(f"[Worker {env_id}] Error: {e}")
            break


class VecEnv:
    """
    Vectorized environment that runs N parallel PPOEnv instances.
    
    Uses multiprocessing to run environments in separate processes.
    Each env has its own Isaac Sim instance.
    """
    
    def __init__(
        self,
        n_envs: int = 4,
        headless: bool = True,
        max_steps: int = 500,
        random_seed: Optional[int] = None,
    ):
        self.n_envs = n_envs
        self.headless = headless
        self.max_steps = max_steps
        self.random_seed = random_seed
        
        # Create worker processes
        self.pipes = []
        self.processes = []
        
        print(f"[VecEnv] Starting {n_envs} parallel environments...")
        
        for i in range(n_envs):
            parent_pipe, child_pipe = mp.Pipe()
            process = mp.Process(
                target=_worker_process,
                args=(i, child_pipe, headless, max_steps, random_seed),
                daemon=True,
            )
            process.start()
            self.pipes.append(parent_pipe)
            self.processes.append(process)
            print(f"[VecEnv] Started env {i+1}/{n_envs}")
        
        # Get observation and action spaces from first env
        self.pipes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.pipes[0].recv()
        
        print(f"[VecEnv] All {n_envs} environments ready")
    
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments. Returns (obs_batch, info_list)."""
        for pipe in self.pipes:
            pipe.send(("reset", None))
        
        obs_list = []
        info_list = []
        
        for pipe in self.pipes:
            obs, info = pipe.recv()
            obs_list.append(obs)
            info_list.append(info)
        
        return np.stack(obs_list), info_list
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with given actions.
        
        Args:
            actions: (n_envs, action_dim) array of actions
            
        Returns:
            obs: (n_envs, obs_dim) observations
            rewards: (n_envs,) rewards
            terminated: (n_envs,) terminated flags
            truncated: (n_envs,) truncated flags
            infos: list of info dicts
        """
        # Send actions to all workers
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", actions[i]))
        
        # Collect results
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for pipe in self.pipes:
            obs, reward, terminated, truncated, info = pipe.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (
            np.stack(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list,
        )
    
    def step_async(self, actions: np.ndarray):
        """Send actions to all workers asynchronously."""
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", actions[i]))
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Wait for all workers to complete step."""
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for pipe in self.pipes:
            obs, reward, terminated, truncated, info = pipe.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (
            np.stack(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list,
        )
    
    def close(self):
        """Close all environments."""
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        print("[VecEnv] All environments closed")


# Simpler single-env wrapper that mimics VecEnv interface for compatibility
class DummyVecEnv:
    """Single environment wrapped to match VecEnv interface."""
    
    def __init__(
        self,
        headless: bool = True,
        max_steps: int = 500,
        random_seed: Optional[int] = None,
    ):
        from ppo_env import PPOEnv
        
        self.n_envs = 1
        self._env = PPOEnv(headless=headless, max_steps=max_steps, random_seed=random_seed)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        obs, info = self._env.reset()
        return obs[np.newaxis, ...], [info]
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs, reward, terminated, truncated, info = self._env.step(actions[0])
        return (
            obs[np.newaxis, ...],
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            [info],
        )
    
    def close(self):
        self._env.close()

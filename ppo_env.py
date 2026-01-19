"""
PPO-ready environment wrapper for the pick-and-place task.

Provides:
- Flattened 21-dim observation vector
- 6-dim delta joint action space
- Milestone flags in info dict

No external gym/gymnasium dependency required.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from load_scene import IsaacPickPlaceEnv


@dataclass
class Box:
    """Simple Box space mimicking gym.spaces.Box."""
    low: float
    high: float
    shape: Tuple[int, ...]
    dtype: np.dtype = np.float32
    
    def sample(self) -> np.ndarray:
        """Sample random action from space."""
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)
    
    def __repr__(self):
        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"


class PPOEnv:
    """
    Gym-compatible wrapper for IsaacPickPlaceEnv.
    
    Observation (21-dim):
        - joint_positions (6)
        - joint_velocities (6)
        - gripper_pos (3)
        - cube_pos (3)
        - cup_pos (3)
    
    Action (6-dim):
        - Delta joint commands scaled by delta_scale
    """
    
    def __init__(
        self,
        headless: bool = True,
        max_steps: int = 500,
        delta_scale: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        self.headless = headless
        self.max_steps = max_steps
        self.delta_scale = delta_scale
        
        # Create underlying Isaac Sim environment
        self._env = IsaacPickPlaceEnv(
            headless=headless,
            capture_images=False,
            random_seed=random_seed,
        )
        
        # Observation space: 21-dim flattened vector
        # joint_pos(6) + joint_vel(6) + gripper_pos(3) + cube_pos(3) + cup_pos(3)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32,
        )
        
        # Action space: delta joint commands (6 joints)
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        
        # Get joint limits for clamping
        self._joint_limits = self._env.robot.joint_limits
        self._joint_names = self._env.robot.joint_names
        
        self._current_step = 0
        self._current_joints = None
    
    def _flatten_obs(self, obs: dict) -> np.ndarray:
        """Flatten observation dict to 21-dim vector."""
        joint_pos = np.array(obs.get("joint_positions", np.zeros(6)), dtype=np.float32)
        joint_vel = np.array(obs.get("joint_velocities", np.zeros(6)), dtype=np.float32)
        gripper_pos = obs.get("gripper_pos")
        cube_pos = np.array(obs.get("cube_pos", np.zeros(3)), dtype=np.float32)
        cup_pos = np.array(obs.get("cup_pos", np.zeros(3)), dtype=np.float32)
        
        # Handle None values
        if gripper_pos is None:
            gripper_pos = np.zeros(3, dtype=np.float32)
        else:
            gripper_pos = np.array(gripper_pos, dtype=np.float32)
        
        return np.concatenate([
            joint_pos[:6],
            joint_vel[:6],
            gripper_pos[:3],
            cube_pos[:3],
            cup_pos[:3],
        ]).astype(np.float32)
    
    def _get_milestone_flags(self, info: dict) -> dict:
        """Extract and augment milestone flags from info."""
        stage_flags = info.get("stage_flags", {})
        task_state = info.get("task_state", {})
        
        # Get detector states
        gripper_detector = self._env.gripper_detector
        
        # Compute milestone flags
        gripper_cube_dist = task_state.get("gripper_cube_distance", float("inf"))
        cube_height = task_state.get("cube_height", 0.0)
        
        flags = {
            "reached": gripper_cube_dist < 0.20 if gripper_cube_dist is not None else False,
            "controlled": bool(stage_flags.get("grasped", False)),
            "lifted": cube_height > 0.03 if cube_height is not None else False,
            "above_cup": bool(getattr(gripper_detector, "is_droppable_range", False) if gripper_detector else False),
            "released": bool(stage_flags.get("droppable_reached", False)),
            "success": bool(stage_flags.get("success", False)),
        }
        return flags
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation."""
        if seed is not None:
            np.random.seed(seed)
        
        obs_dict = self._env.reset(render=not self.headless)
        self._current_step = 0
        self._current_joints = np.array(obs_dict.get("joint_positions", np.zeros(6)), dtype=np.float32)
        
        obs = self._flatten_obs(obs_dict)
        info = {"milestone_flags": self._get_milestone_flags({})}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute delta action and return (obs, reward, terminated, truncated, info).
        
        Args:
            action: 6-dim delta joint commands in [-1, 1], scaled by delta_scale
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Convert delta action to absolute joint positions
        delta = action * self.delta_scale
        target_joints = self._current_joints + delta
        
        # Clamp to joint limits
        for i, name in enumerate(self._joint_names[:6]):
            lower, upper = self._joint_limits[name]
            target_joints[i] = np.clip(target_joints[i], lower, upper)
        
        # Step underlying environment
        obs_dict, reward, done, info = self._env.step(target_joints, render=not self.headless)
        
        self._current_step += 1
        self._current_joints = np.array(obs_dict.get("joint_positions", target_joints), dtype=np.float32)
        
        # Get milestone flags
        milestone_flags = self._get_milestone_flags(info)
        info["milestone_flags"] = milestone_flags
        
        # Flatten observation
        obs = self._flatten_obs(obs_dict)
        
        # Determine termination
        terminated = done or milestone_flags["success"]
        truncated = self._current_step >= self.max_steps
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render is handled by Isaac Sim internally."""
        pass
    
    def close(self):
        """Clean up environment."""
        self._env.shutdown()


def make_ppo_env(headless: bool = True, max_steps: int = 500, delta_scale: float = 0.05):
    """Factory function to create PPOEnv."""
    return PPOEnv(headless=headless, max_steps=max_steps, delta_scale=delta_scale)


if __name__ == "__main__":
    # Quick test
    env = PPOEnv(headless=False, max_steps=100)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial milestone flags: {info.get('milestone_flags')}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, flags={info.get('milestone_flags')}")
        if terminated or truncated:
            break
    
    env.close()

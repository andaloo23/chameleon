"""
Minimal PPO training script for the pick-and-place task in Isaac Lab.

Uses a tiny MLP policy and trains on the PickPlaceEnv across parallel GPU environments.
Outputs statistics: avg reward, % reached, % grasped, % lifted, % success.
"""

import argparse
import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class RunningObsNormalizer:
    """Welford online running mean/variance for observation normalization."""

    def __init__(self, obs_dim: int, device: str = "cpu", clip: float = 10.0):
        self.mean = torch.zeros(obs_dim, device=device)
        self.var = torch.ones(obs_dim, device=device)
        self.count = 1e-4  # avoid div-by-zero
        self.clip = clip

    def update(self, x: torch.Tensor):
        """Update running stats with a batch [N, obs_dim]."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize observations to ~zero-mean, unit-variance."""
        return torch.clamp((x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip)


def _extract_obs(obs):
    """Extract observation tensor from Isaac Lab dict format.
    
    Returns: 1D numpy array suitable for training
    """
    # Isaac Lab format - extract policy tensor
    policy_obs = obs.get("policy", obs.get("obs", None))
    if policy_obs is None:
        raise ValueError(f"Unknown observation dict keys: {obs.keys()}")
    
    # Handle batched tensor (take first env for single-env training inference if needed)
    if hasattr(policy_obs, "cpu"):
        if policy_obs.dim() > 1:
            policy_obs = policy_obs[0]  # Take first environment for metrics
        return policy_obs.cpu().numpy()
    return np.array(policy_obs)


class TinyMLP(nn.Module):
    """
    MLP policy for PPO with separate actor/critic heads.
    
    Input: obs_dim observation
    Output: act_dim action mean + act_dim action log_std
    """
    
    def __init__(self, obs_dim: int = 21, act_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Policy head (actor) — own hidden layer for specialization
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden_dim, act_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Value head (critic) — own hidden layer for specialization
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action mean and value estimate."""
        features = self.shared(obs)
        action_mean = self.policy_mean(self.actor_head(features))
        value = self.value(self.critic_head(features))
        
        # Clamp log_std: floor at -0.3 (std>=0.74) prevents entropy collapse,
        # ceiling at 0.5 prevents excess noise
        with torch.no_grad():
            self.policy_log_std.clamp_(-0.3, 0.5)
            
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.policy_log_std)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
        
        return action, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log prob and entropy for given actions."""
        action_mean, value = self.forward(obs)
        action_std = torch.exp(self.policy_log_std)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """Simple buffer for storing rollout data."""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
    
    def add(self, obs, action, reward, value, done, log_prob):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.log_probs.clear()
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages and returns."""
        advantages = []
        returns = []
        
        gae = 0.0
        next_value = last_value
        
        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            next_value = self.values[t]
        
        return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)
    
    def get_tensors(self):
        """Convert buffer to tensors."""
        return (
            torch.tensor(np.array(self.observations), dtype=torch.float32),
            torch.tensor(np.array(self.actions), dtype=torch.float32),
            torch.tensor(np.array(self.log_probs), dtype=torch.float32),
        )


def ppo_update(
    policy: TinyMLP,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.001,
    n_epochs: int = 5,
    batch_size: int = 1024,
):
    """Perform PPO update. Returns dict of metrics."""
    # Get device from policy
    device = next(policy.parameters()).device
    
    returns, advantages = buffer.compute_returns_and_advantages(last_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    obs_tensor, actions_tensor, old_log_probs_tensor = buffer.get_tensors()
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    values_tensor = torch.tensor(np.array(buffer.values), dtype=torch.float32)
    
    # Move all tensors to policy device
    obs_tensor = obs_tensor.to(device)
    actions_tensor = actions_tensor.to(device)
    old_log_probs_tensor = old_log_probs_tensor.to(device)
    returns_tensor = returns_tensor.to(device)
    advantages_tensor = advantages_tensor.to(device)
    values_tensor = values_tensor.to(device)
    
    n_samples = len(buffer.observations)
    indices = np.arange(n_samples)
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_fraction = 0.0
    n_updates = 0
    
    kl_brake_triggered = False
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            batch_obs = obs_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            
            # Evaluate actions
            log_probs, entropy, values = policy.evaluate_actions(batch_obs, batch_actions)
            
            # Policy loss (clipped surrogate)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, batch_returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                clip_fraction = (torch.abs(ratio - 1.0) > clip_eps).float().mean().item()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_approx_kl += approx_kl
            total_clip_fraction += clip_fraction
            n_updates += 1
            
            # KL Safety Brake: if KL divergence is too high, stop updating to prevent collapse
            if approx_kl > 0.05:
                print(f"      [KL Brake] Early stopping at epoch {epoch+1}, batch {start} | KL: {approx_kl:.4f} > 0.05")
                kl_brake_triggered = True
                break
        
        if kl_brake_triggered:
            break
    
    # Compute explained variance
    with torch.no_grad():
        explained_var = 1 - (returns_tensor - values_tensor).var() / (returns_tensor.var() + 1e-8)
        explained_var = explained_var.item()
    
    metrics = {
        "loss": total_loss / max(n_updates, 1),
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
        "approx_kl": total_approx_kl / max(n_updates, 1),
        "clip_fraction": total_clip_fraction / max(n_updates, 1),
        "explained_variance": explained_var,
    }
    return metrics


def collect_rollout(
    env,
    policy: TinyMLP,
    buffer: RolloutBuffer,
    n_steps: int = 2048,
    num_envs: int = 1,
    state: Dict = None,
) -> Tuple[float, List[float], List[Dict], float, List[float], List[float], Dict]:
    """Collect rollout using Isaac Lab parallel environments.
    
    Returns:
        (last_value, episode_rewards, episode_flags, mean_action_mag, ep_min_dists, ep_final_dists, next_state)
    """
    device = env.device
    
    # Initialize or restore state
    if state is None or "obs_dict" not in state:
        obs_dict, info = env.reset()
        current_rewards = torch.zeros(num_envs, device=device)
        min_dists = torch.full((num_envs,), float("inf"), device=device)
        final_dists = torch.zeros(num_envs, device=device)
        
        # Latched milestone flags for each environment
        ever_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_droppable = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Per-env fingertip in-region accumulators (sum of per-step fracs; divide by episode len)
        sum_left_in_region  = torch.zeros(num_envs, device=device)
        sum_right_in_region = torch.zeros(num_envs, device=device)
        ep_steps = torch.zeros(num_envs, device=device)
        
        penalties = {
            "action": torch.zeros(num_envs, device=device),
            "drop": torch.zeros(num_envs, device=device),
            "cup": torch.zeros(num_envs, device=device),
            "self": torch.zeros(num_envs, device=device),
        }
    else:
        obs_dict = state["obs_dict"]
        current_rewards = state["current_rewards"]
        min_dists = state["min_dists"]
        final_dists = state["final_dists"]
        ever_reached = state["ever_reached"]
        ever_grasped = state["ever_grasped"]
        ever_lifted = state["ever_lifted"]
        ever_droppable = state["ever_droppable"]
        ever_success = state["ever_success"]
        prev_cube_z = state.get("prev_cube_z", torch.zeros_like(ever_grasped))
        penalties = state["penalties"]
        info = state["info"]
        ep_steps = state.get("ep_steps", torch.zeros(num_envs, device=device))

    episode_rewards, episode_flags, episode_min_dists, episode_final_dists = [], [], [], []
    action_magnitudes = []
    
    steps_per_env = max(1, n_steps // num_envs)
    
    # Get obs normalizer from state if available
    obs_normalizer = state.get("obs_normalizer", None) if state else None

    for _ in range(steps_per_env):
        policy_obs = obs_dict["policy"]
        if not policy_obs.is_cuda and device.type == "cuda":
            policy_obs = policy_obs.to(device)

        # Update and apply observation normalization
        if obs_normalizer is not None:
            obs_normalizer.update(policy_obs)
            policy_obs_norm = obs_normalizer.normalize(policy_obs)
        else:
            policy_obs_norm = policy_obs
            
        with torch.no_grad():
            action_mean, values = policy.forward(policy_obs_norm)
            action_std = torch.exp(policy.policy_log_std)
            dist = Normal(action_mean, action_std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            values = values.squeeze(-1)
            
        action_magnitudes.append(actions.abs().mean().item())
        
        # Step environment
        next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
        dones = terminated | truncated
        
        current_rewards += rewards
        
        task_state = info.get("task_state", {})
        if "gripper_cube_distance" in task_state:
            d = task_state["gripper_cube_distance"]
            min_dists = torch.minimum(min_dists, d)
            final_dists = d.clone()
            ever_reached |= (d < 0.15)
            
        # Update milestone flags using the latched state from the environment
        milestones = info.get("milestone_flags", {})
        ever_grasped |= milestones.get("grasped", torch.zeros_like(ever_grasped))
        ever_lifted |= milestones.get("lifted", torch.zeros_like(ever_lifted))
        ever_droppable |= milestones.get("droppable", torch.zeros_like(ever_droppable))
        ever_success |= milestones.get("success", torch.zeros_like(ever_success))
        
        # Penalties breakdown
        if "penalties" in task_state:
            p = task_state["penalties"]
            penalties["action"] += p.get("action_cost", 0.0)
            penalties["drop"] += p.get("drop_penalty", 0.0)
            penalties["cup"] += p.get("cup_collision", 0.0)
            penalties["self"] += p.get("self_collision", 0.0)

        ep_steps += 1.0
            
        # Store in buffer
        po_np = policy_obs_norm.cpu().numpy()
        ac_np = actions.cpu().numpy()
        re_np = rewards.cpu().numpy()
        va_np = values.cpu().numpy()
        do_np = dones.cpu().numpy()
        lp_np = log_probs.cpu().numpy()
        
        for i in range(num_envs):
            buffer.add(po_np[i], ac_np[i], re_np[i], va_np[i], do_np[i], lp_np[i])
            
            if do_np[i]:
                episode_rewards.append(current_rewards[i].item())
                episode_min_dists.append(min_dists[i].item())
                episode_final_dists.append(final_dists[i].item())
                episode_flags.append({
                    "reached": ever_reached[i].item(),
                    "grasped": ever_grasped[i].item(),
                    "lifted": ever_lifted[i].item(),
                    "droppable": ever_droppable[i].item(),
                    "success": ever_success[i].item(),
                })

                # Reset per-env
                current_rewards[i] = 0.0
                min_dists[i] = float("inf")
                ever_reached[i], ever_grasped[i], ever_lifted[i], ever_droppable[i], ever_success[i] = False, False, False, False, False
                ep_steps[i] = 0.0
                for k in penalties: penalties[k][i] = 0.0
                
        obs_dict = next_obs_dict
        
    # Last value for GAE
    with torch.no_grad():
        last_obs = obs_dict["policy"].to(device)
        if obs_normalizer is not None:
            last_obs = obs_normalizer.normalize(last_obs)
        _, last_values = policy.forward(last_obs)
        last_value = last_values[0].item()
        
    next_state = {
        "obs_dict": obs_dict, "current_rewards": current_rewards, "min_dists": min_dists,
        "final_dists": final_dists, "ever_reached": ever_reached, "ever_grasped": ever_grasped,
        "ever_lifted": ever_lifted, "ever_droppable": ever_droppable, "ever_success": ever_success, 
        "prev_cube_z": getattr(env, "_prev_cube_z", torch.zeros_like(ever_grasped)),
        "penalties": penalties, "info": info,
        "ep_steps": ep_steps,
        "obs_normalizer": obs_normalizer,
    }
    
    return last_value, episode_rewards, episode_flags, np.mean(action_magnitudes), episode_min_dists, episode_final_dists, next_state




def _collect_rollout_single(env, policy: TinyMLP, buffer: RolloutBuffer, n_steps: int):
    """Single-env rollout collection for legacy compatibility."""
    obs, info = env.reset()
    episode_rewards = []
    episode_flags = []
    episode_min_dists = []
    episode_final_dists = []
    current_episode_reward = 0.0
    current_episode_penalties = {
        "action": 0.0,
        "drop": 0.0,
        "cup": 0.0,
        "self": 0.0
    }
    
    # Per-episode tracking (latched flags)
    min_gripper_cube_dist = float("inf")
    final_gripper_cube_dist = float("inf")
    min_gripper_width = float("inf")
    ever_reached = False
    ever_grasped = False
    ever_droppable = False
    ever_success = False
    episode_count = 0
    step_count = 0
    first_episode_debug = True
    
    # Reward component sums per episode
    sum_approach = 0.0
    sum_action_cost = 0.0
    sum_joint_limit = 0.0
    sum_self_keepout = 0.0
    
    # Action magnitude tracking
    action_magnitudes = []
    
    for _ in range(n_steps):
        obs_array = _extract_obs(obs)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action, value = policy.get_action(obs_tensor)
            action = action.squeeze(0).numpy()
            value = value.item()
            
            # Compute log prob for the action
            action_mean, _ = policy.forward(obs_tensor)
            action_std = torch.exp(policy.policy_log_std)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(torch.tensor(action)).sum().item()
        
        # Track action magnitude
        action_magnitudes.append(np.abs(action).mean())
        
        # Handle action format: Isaac Lab expects batched tensor, legacy expects numpy
        if isinstance(obs, dict):
            action_input = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                action_input = action_input.cuda()
        else:
            action_input = action
        
        next_obs, reward, terminated, truncated, info = env.step(action_input)
        
        # Extract scalars from potential tensors
        if hasattr(reward, "item"):
            reward = reward[0].item() if reward.dim() > 0 else reward.item()
        if hasattr(terminated, "item"):
            terminated = terminated[0].item() if terminated.dim() > 0 else terminated.item()
        if hasattr(truncated, "item"):
            truncated = truncated[0].item() if truncated.dim() > 0 else truncated.item()
        
        done = terminated or truncated
        
        # Track per-step statistics from info
        task_state = info.get("task_state", {}) if isinstance(info, dict) else {}
        reward_components = info.get("reward_components", {}) if isinstance(info, dict) else {}
        gripper_cube_dist = task_state.get("gripper_cube_distance")
        gripper_width = task_state.get("gripper_width")
        
        # Get minimum distance to body links
        d_base = task_state.get("gripper_base_distance", float("inf"))
        d_shoulder = task_state.get("gripper_shoulder_distance", float("inf"))
        d_upper_arm = task_state.get("gripper_upper_arm_distance", float("inf"))
        d_min_body = min(d_base, d_shoulder, d_upper_arm)
        
        # Track reward component sums
        sum_approach += reward_components.get("approach_shaping", 0.0)
        sum_action_cost += reward_components.get("action_cost", 0.0)
        sum_joint_limit += reward_components.get("joint_limit_penalty", 0.0)
        sum_self_keepout += reward_components.get("self_collision_penalty", 0.0)
        
        # Track cumulative penalties for breakdown logging
        current_episode_penalties["action"] += reward_components.get("action_cost", 0.0)
        current_episode_penalties["drop"] += reward_components.get("drop_penalty", 0.0)
        current_episode_penalties["cup"] += reward_components.get("cup_collision_penalty", 0.0)
        current_episode_penalties["self"] += reward_components.get("self_collision_penalty", 0.0)
        
        if gripper_cube_dist is not None:
            min_gripper_cube_dist = min(min_gripper_cube_dist, gripper_cube_dist)
            final_gripper_cube_dist = gripper_cube_dist
            if gripper_cube_dist < 0.15:
                ever_reached = True
        
        if gripper_width is not None:
            min_gripper_width = min(min_gripper_width, gripper_width)

        # Update latched flags from task_state
        if task_state.get("is_grasped", False): ever_grasped = True
        if task_state.get("is_droppable", False): ever_droppable = True
        if task_state.get("is_in_cup", False): ever_success = True
        
        buffer.add(obs_array, action, reward, value, done, log_prob)
        current_episode_reward += reward
        step_count += 1
        
        if done:
            episode_count += 1
            episode_rewards.append(current_episode_reward)
            # Prepare milestone flags for logging (interpret task_state for Isaac Lab)
            # Both reached and grasped are needed for Iter summary
            m_flags = info.get("milestone_flags", {})
            if not m_flags and "task_state" in info:
                # Isaac Lab fallback - use latched flags
                m_flags = {
                    "reached": ever_reached,
                    "grasped": ever_grasped or (min_gripper_width < 0.03), # Heuristic for grasp
                    "lifted": ever_droppable,
                    "success": ever_success
                }
            
            # Ensure keys are unified for summary
            if "controlled" in m_flags:
                m_flags["grasped"] = m_flags.get("grasped") or m_flags["controlled"]
            
            # Handle potential tensors in task_state
            for k, v in m_flags.items():
                if hasattr(v, "item"): m_flags[k] = bool(v.item())
            
            episode_flags.append(m_flags)
            episode_min_dists.append(min_gripper_cube_dist)
            episode_final_dists.append(final_gripper_cube_dist)
            
            msg = (f"  [Ep {episode_count}] "
                  f"MinDist: {min_gripper_cube_dist:.3f}m | "
                  f"GripperWidth: {min_gripper_width:.4f}m | "
                  f"Reward: {current_episode_reward:.2f} | "
                  f"P: Action:{current_episode_penalties['action']:.2f} "
                  f"Drop:{current_episode_penalties['drop']:.2f} "
                  f"Cup:{current_episode_penalties['cup']:.2f} "
                  f"Self:{current_episode_penalties['self']:.2f}")
            print(msg)
            
            if first_episode_debug:
                first_episode_debug = False
                print("  [First episode debug complete]")
            
            # Reset tracking
            current_episode_reward = 0.0
            for k in current_episode_penalties:
                current_episode_penalties[k] = 0.0
            min_gripper_cube_dist = float("inf")
            final_gripper_cube_dist = float("inf")
            min_gripper_width = float("inf")
            ever_reached = False
            ever_grasped = False
            ever_droppable = False
            ever_success = False
            sum_approach = 0.0
            sum_action_cost = 0.0
            sum_joint_limit = 0.0
            sum_self_keepout = 0.0
            step_count = 0
            obs, info = env.reset()
        else:
            obs = next_obs
    
    # Get last value for GAE computation
    with torch.no_grad():
        obs_array = _extract_obs(obs)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        _, last_value = policy.get_action(obs_tensor)
        last_value = last_value.item()
    
    mean_action_mag = np.mean(action_magnitudes) if action_magnitudes else 0.0
    return last_value, episode_rewards, episode_flags, mean_action_mag, episode_min_dists, episode_final_dists


def train_ppo(
    num_episodes: int = 1000,
    max_steps: int = 500,
    headless: bool = True,
    learning_rate: float = 3e-4,
    rollout_steps: int = 2048,
    log_interval: int = 10,
    n_envs: int = 1,
):
    """
    Train PPO on the pick-and-place task using Isaac Lab.
    
    Args:
        num_episodes: Target number of episodes to collect
        max_steps: Max steps per episode
        headless: Run simulation headless
        learning_rate: Learning rate for optimizer
        rollout_steps: Steps per rollout before update
        log_interval: Episodes between logging
        n_envs: Number of parallel environments
    """
    print("=" * 60)
    print("PPO Training for Pick-and-Place Task (Isaac Lab)")
    print("=" * 60)
    
    # Create environment(s)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    
    from lab.pick_place_env import PickPlaceEnv
    from lab.pick_place_env_cfg import PickPlaceEnvCfg
    
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = n_envs
    cfg.episode_length_s = max_steps / 60.0  # Convert steps to seconds
    env = PickPlaceEnv(cfg)
    print(f"Using Isaac Lab with {n_envs} GPU-parallel environments")
    
    # Isaac Lab uses gymnasium-style API with dict observations
    obs_dim = cfg.observation_space
    act_dim = cfg.action_space
    
    print(f"Observation space: {obs_dim}")
    print(f"Action space: {act_dim}")
    
    # Create policy and optimizer
    hidden_dim = 256
    policy = TinyMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim)
    
    if torch.cuda.is_available():
        policy = policy.cuda()
        print("Policy moved to GPU")
    
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    print(f"Policy: MLP({obs_dim} -> {hidden_dim}x2 -> actor:{hidden_dim} / critic:{hidden_dim} -> {act_dim})")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters())}")
    print("=" * 60)
    
    # Training loop
    buffer = RolloutBuffer()
    all_rewards = deque(maxlen=100)
    all_flags = deque(maxlen=100)
    total_episodes = 0
    iteration = 0
    
    # Metrics tracking for plotting - now per-episode for smoother curves
    metrics_history = {
        # Per-episode data (logged for each episode)
        "episode_num": [],      # Episode index
        "reward": [],           # Episode reward
        "min_dist": [],         # Min gripper-cube distance
        "reached": [],          # Boolean: did gripper reach cube
        "grasped": [],          # Boolean: did gripper grasp cube  
        "lifted": [],           # Boolean: was cube lifted off ground
        "droppable": [],         # Boolean: was cube aligned over cup
        "success": [],          # Boolean: was cube placed in cup
        
        # Per-iteration PPO metrics (for training diagnostics)
        "iter_episode": [],     # Episode count at iteration
        "entropy": [],
        "approx_kl": [],
        "value_loss": [],
    }
    
    # State for batched rollout persistence (Isaac Lab only)
    obs_normalizer = RunningObsNormalizer(obs_dim, device=env.device)
    rollout_state = {"obs_normalizer": obs_normalizer}
    
    while total_episodes < num_episodes:
        iteration += 1
        buffer.clear()
        
        last_value, episode_rewards, episode_flags, mean_action_mag, ep_min_dists, ep_final_dists, rollout_state = collect_rollout(
            env, policy, buffer, n_steps=rollout_steps, num_envs=n_envs, state=rollout_state
        )
        
        # Update policy
        metrics = ppo_update(
            policy, optimizer, buffer, last_value,
            n_epochs=5, batch_size=1024
        )
        
        # Track statistics
        all_rewards.extend(episode_rewards)
        all_flags.extend(episode_flags)
        total_episodes += len(episode_rewards)
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        
        # Store per-episode metrics for plotting
        for i, (reward, min_d, flags) in enumerate(zip(episode_rewards, ep_min_dists, episode_flags)):
            ep_idx = total_episodes - len(episode_rewards) + i + 1
            metrics_history["episode_num"].append(ep_idx)
            metrics_history["reward"].append(reward)
            metrics_history["min_dist"].append(min_d if min_d != float("inf") else 0.0)
            metrics_history["reached"].append(flags.get("reached", False))
            metrics_history["grasped"].append(flags.get("grasped", False))
            metrics_history["lifted"].append(flags.get("lifted", False))
            metrics_history["droppable"].append(flags.get("droppable", False))
            metrics_history["success"].append(flags.get("success", False))
        
        # Store per-iteration PPO metrics
        metrics_history["iter_episode"].append(total_episodes)
        metrics_history["entropy"].append(metrics["entropy"])
        metrics_history["approx_kl"].append(metrics["approx_kl"])
        metrics_history["value_loss"].append(metrics["value_loss"])
        
        # Log progress
        if len(episode_flags) > 0:  # Only log when episodes completed
            # Count milestone percentages from THIS rollout's episodes (not stale deque)
            n_ep_this_iter = len(episode_flags)
            if n_ep_this_iter > 0:
                pct_reached = 100 * sum(1 for f in episode_flags if f.get("reached")) / n_ep_this_iter
                pct_grasped = 100 * sum(1 for f in episode_flags if f.get("grasped")) / n_ep_this_iter
                pct_lifted = 100 * sum(1 for f in episode_flags if f.get("lifted")) / n_ep_this_iter
                pct_droppable = 100 * sum(1 for f in episode_flags if f.get("droppable")) / n_ep_this_iter
                pct_success = 100 * sum(1 for f in episode_flags if f.get("success")) / n_ep_this_iter
                iter_avg_reward = np.mean(episode_rewards)
            else:
                pct_reached = pct_grasped = pct_lifted = pct_droppable = pct_success = 0.0
                iter_avg_reward = 0.0
            
            last_ep_reward = episode_rewards[-1] if episode_rewards else 0.0
            print(f"[Iter {iteration:4d}] Ep: {total_episodes:5d} ({n_ep_this_iter} this iter) | "
                  f"Reward: {iter_avg_reward:7.2f} | "
                  f"Last: {last_ep_reward:7.2f} | "
                  f"Entropy: {metrics['entropy']:.3f} | "
                  f"KL: {metrics['approx_kl']:.4f}")
            print(f"           R:{pct_reached:4.1f}% G:{pct_grasped:4.1f}% "
                  f"L:{pct_lifted:4.1f}% D:{pct_droppable:4.1f}% S:{pct_success:4.1f}%")
    
    # Final statistics — use full metrics_history for accurate start-to-finish report
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    n_total = len(metrics_history["reward"])
    if n_total > 0:
        # Overall stats
        avg_reward_all = np.mean(metrics_history["reward"])
        last_reward = metrics_history["reward"][-1]
        
        # Compute percentages over ALL episodes
        pct_reached_all   = 100 * sum(metrics_history["reached"])   / n_total
        pct_grasped_all   = 100 * sum(metrics_history["grasped"])   / n_total
        pct_lifted_all    = 100 * sum(metrics_history["lifted"])    / n_total
        pct_droppable_all = 100 * sum(metrics_history["droppable"]) / n_total
        pct_success_all   = 100 * sum(metrics_history["success"])   / n_total
        
        # Start vs finish comparison (first 10% vs last 10%)
        window = max(1, n_total // 10)
        
        def pct(flags, start, end):
            sl = flags[start:end]
            return 100 * sum(sl) / max(1, len(sl))
        
        print(f"Total episodes: {n_total}")
        print(f"Average Episode Reward (all): {avg_reward_all:.2f}")
        print(f"Last Episode Reward: {last_reward:.2f}")
        print()
        print(f"{'Milestone':<12} {'Overall':>8}  {'First 10%':>10}  {'Last 10%':>10}  {'Delta':>8}")
        print("-" * 54)
        for key, label in [("reached", "Reached"), ("grasped", "Grasped"),
                           ("lifted", "Lifted"), ("droppable", "Droppable"),
                           ("success", "Success")]:
            p_all   = pct(metrics_history[key], 0, n_total)
            p_start = pct(metrics_history[key], 0, window)
            p_end   = pct(metrics_history[key], n_total - window, n_total)
            delta   = p_end - p_start
            sign    = "+" if delta >= 0 else ""
            print(f"{label:<12} {p_all:>7.1f}%  {p_start:>9.1f}%  {p_end:>9.1f}%  {sign}{delta:>6.1f}%")
    
    # Plot metrics
    plot_training_metrics(metrics_history)
    
    env.close()
    
    return policy


def plot_training_metrics(metrics: Dict, smoothing_window: int = 20):
    """Plot training metrics vs episode number with moving average smoothing."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for headless
        import matplotlib.pyplot as plt
        
        def moving_avg(data, window):
            """Compute moving average with given window size."""
            if len(data) < window:
                window = max(1, len(data))
            cumsum = np.cumsum(np.insert(data, 0, 0))
            return (cumsum[window:] - cumsum[:-window]) / window
        
        def moving_avg_bool(data, window):
            """Compute moving average of boolean array (gives percentage)."""
            return moving_avg(np.array(data, dtype=float) * 100, window)
        
        episodes = np.array(metrics["episode_num"])
        if len(episodes) < 2:
            print("[WARN] Not enough episodes for plotting")
            return
        
        w = min(smoothing_window, len(episodes) // 2)  # Adaptive window
        ep_smoothed = episodes[w-1:]  # Align with smoothed data
        
        fig, axes = plt.subplots(4, 2, figsize=(12, 13))
        fig.suptitle(f"PPO Training Metrics (smoothing window={w})", fontsize=14)
        
        # Plot 1: Episode Reward (smoothed)
        reward_smoothed = moving_avg(np.array(metrics["reward"]), w)
        axes[0, 0].plot(ep_smoothed, reward_smoothed, 'b-', linewidth=2)
        axes[0, 0].fill_between(ep_smoothed, reward_smoothed * 0.9, reward_smoothed * 1.1, alpha=0.2)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Episode Reward (Smoothed)")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Min Distance (smoothed)
        min_dist_smoothed = moving_avg(np.array(metrics["min_dist"]), w)
        axes[0, 1].plot(ep_smoothed, min_dist_smoothed, 'purple', linewidth=2)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Min Distance (m)")
        axes[0, 1].set_title("Min Gripper-Cube Distance (Smoothed)")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: % Reached
        reached_smoothed = moving_avg_bool(metrics["reached"], w)
        axes[1, 0].plot(ep_smoothed, reached_smoothed, 'g-', linewidth=2)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("% Reached")
        axes[1, 0].set_title("Success Rate: Reached Cube")
        axes[1, 0].set_ylim(0, 105)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: % Grasped
        grasped_smoothed = moving_avg_bool(metrics["grasped"], w)
        axes[1, 1].plot(ep_smoothed, grasped_smoothed, 'orange', linewidth=2)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("% Grasped")
        axes[1, 1].set_title("Success Rate: Grasped Cube")
        axes[1, 1].set_ylim(0, 105)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: % Lifted
        lifted_smoothed = moving_avg_bool(metrics["lifted"], w)
        axes[2, 0].plot(ep_smoothed, lifted_smoothed, 'm-', linewidth=2)
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].set_ylabel("% Lifted")
        axes[2, 0].set_title("Success Rate: Lifted Cube")
        axes[2, 0].set_ylim(0, 105)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: % Droppable
        droppable_smoothed = moving_avg_bool(metrics["droppable"], w)
        axes[2, 1].plot(ep_smoothed, droppable_smoothed, 'c-', linewidth=2)
        axes[2, 1].set_xlabel("Episode")
        axes[2, 1].set_ylabel("% Droppable")
        axes[2, 1].set_title("Success Rate: Aligned Above Cup")
        axes[2, 1].set_ylim(0, 105)
        axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 7: % Success
        success_smoothed = moving_avg_bool(metrics["success"], w)
        axes[3, 0].plot(ep_smoothed, success_smoothed, 'r-', linewidth=2)
        axes[3, 0].set_xlabel("Episode")
        axes[3, 0].set_ylabel("% Success")
        axes[3, 0].set_title("Success Rate: Placed in Cup")
        axes[3, 0].set_ylim(0, 105)
        axes[3, 0].grid(True, alpha=0.3)
        
        # Plot 8: Entropy (training diagnostic)
        if metrics.get("iter_episode") and metrics.get("entropy"):
            axes[3, 1].plot(metrics["iter_episode"], metrics["entropy"], 'k-', linewidth=1, alpha=0.7)
            axes[3, 1].set_xlabel("Episode")
            axes[3, 1].set_ylabel("Entropy")
            axes[3, 1].set_title("Policy Entropy")
            axes[3, 1].grid(True, alpha=0.3)
        else:
            axes[3, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig("ppo_training_metrics.png", dpi=150)
        print(f"\n[INFO] Training metrics plot saved to: ppo_training_metrics.png")
        plt.close()
        
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
    except Exception as e:
        print(f"[WARN] Failed to create plots: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on pick-and-place task (Isaac Lab)")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rollout-steps", type=int, default=131072, help="Steps per rollout")
    parser.add_argument("--n-envs", type=int, default=1024, help="Number of parallel environments")
    
    args = parser.parse_args()
    
    print("[INFO] Using Isaac Lab backend for GPU-accelerated training")
    print(f"[INFO] Parallel environments: {args.n_envs}")
    print(f"[INFO] Total steps per iteration: {args.rollout_steps} ({args.rollout_steps // args.n_envs} steps/env)")
    
    train_ppo(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        learning_rate=args.lr,
        rollout_steps=args.rollout_steps,
        n_envs=args.n_envs,
    )


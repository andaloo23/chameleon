"""
Minimal PPO training script for the pick-and-place task.

Uses a tiny MLP policy and trains on the PPOEnv wrapper.
Outputs statistics: avg reward, % reached, % controlled, % lifted.
"""

import argparse
import numpy as np
from collections import deque
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class TinyMLP(nn.Module):
    """
    Tiny MLP policy for PPO.
    
    Input: 21-dim observation
    Output: 6-dim action mean + 6-dim action log_std
    """
    
    def __init__(self, obs_dim: int = 21, act_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(hidden_dim, act_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action mean and value estimate."""
        features = self.shared(obs)
        action_mean = self.policy_mean(features)
        value = self.value(features)
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
    entropy_coef: float = 0.01,
    n_epochs: int = 4,
    batch_size: int = 64,
):
    """Perform PPO update."""
    returns, advantages = buffer.compute_returns_and_advantages(last_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    obs_tensor, actions_tensor, old_log_probs_tensor = buffer.get_tensors()
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    
    n_samples = len(buffer.observations)
    indices = np.arange(n_samples)
    
    total_loss = 0.0
    n_updates = 0
    
    for _ in range(n_epochs):
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
            
            total_loss += loss.item()
            n_updates += 1
    
    return total_loss / max(n_updates, 1)


def collect_rollout(env, policy: TinyMLP, buffer: RolloutBuffer, n_steps: int = 2048):
    """Collect rollout data from environment."""
    obs, info = env.reset()
    episode_rewards = []
    episode_flags = []
    current_episode_reward = 0.0
    
    # Per-episode tracking
    min_gripper_cube_dist = float("inf")
    final_gripper_cube_dist = float("inf")
    min_gripper_width = float("inf")
    episode_count = 0
    
    for _ in range(n_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action, value = policy.get_action(obs_tensor)
            action = action.squeeze(0).numpy()
            value = value.item()
            
            # Compute log prob for the action
            action_mean, _ = policy.forward(obs_tensor)
            action_std = torch.exp(policy.policy_log_std)
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(torch.tensor(action)).sum().item()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track per-step statistics from info
        task_state = info.get("task_state", {})
        gripper_cube_dist = task_state.get("gripper_cube_distance")
        gripper_joint = task_state.get("gripper_joint")
        
        if gripper_cube_dist is not None:
            min_gripper_cube_dist = min(min_gripper_cube_dist, gripper_cube_dist)
            final_gripper_cube_dist = gripper_cube_dist
        
        if gripper_joint is not None:
            min_gripper_width = min(min_gripper_width, gripper_joint)
        
        buffer.add(obs, action, reward, value, done, log_prob)
        current_episode_reward += reward
        
        if done:
            episode_count += 1
            episode_rewards.append(current_episode_reward)
            episode_flags.append(info.get("milestone_flags", {}))
            
            # Print per-episode statistics
            print(f"  [Episode {episode_count}] "
                  f"MinDist: {min_gripper_cube_dist:.4f}m | "
                  f"FinalDist: {final_gripper_cube_dist:.4f}m | "
                  f"MinGripperWidth: {min_gripper_width:.4f} | "
                  f"Reward: {current_episode_reward:.2f}")
            
            # Reset tracking for next episode
            current_episode_reward = 0.0
            min_gripper_cube_dist = float("inf")
            final_gripper_cube_dist = float("inf")
            min_gripper_width = float("inf")
            obs, info = env.reset()
        else:
            obs = next_obs
    
    # Get last value for GAE computation
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        _, last_value = policy.get_action(obs_tensor)
        last_value = last_value.item()
    
    return last_value, episode_rewards, episode_flags


def train_ppo(
    num_episodes: int = 1000,
    max_steps: int = 500,
    headless: bool = True,
    learning_rate: float = 3e-4,
    rollout_steps: int = 2048,
    log_interval: int = 10,
):
    """
    Train PPO on the pick-and-place task.
    
    Args:
        num_episodes: Target number of episodes to collect
        max_steps: Max steps per episode
        headless: Run simulation headless
        learning_rate: Learning rate for optimizer
        rollout_steps: Steps per rollout before update
        log_interval: Episodes between logging
    """
    from ppo_env import PPOEnv
    
    print("=" * 60)
    print("PPO Training for Pick-and-Place Task")
    print("=" * 60)
    
    # Create environment
    env = PPOEnv(headless=headless, max_steps=max_steps)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create policy and optimizer
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = TinyMLP(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=64)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    print(f"Policy: TinyMLP({obs_dim} -> 64 -> 64 -> {act_dim})")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters())}")
    print("=" * 60)
    
    # Training loop
    buffer = RolloutBuffer()
    all_rewards = deque(maxlen=100)
    all_flags = deque(maxlen=100)
    total_episodes = 0
    iteration = 0
    
    while total_episodes < num_episodes:
        iteration += 1
        buffer.clear()
        
        # Collect rollout
        last_value, episode_rewards, episode_flags = collect_rollout(
            env, policy, buffer, n_steps=rollout_steps
        )
        
        # Update policy
        loss = ppo_update(policy, optimizer, buffer, last_value)
        
        # Track statistics
        all_rewards.extend(episode_rewards)
        all_flags.extend(episode_flags)
        total_episodes += len(episode_rewards)
        
        # Log progress
        if iteration % log_interval == 0 or total_episodes >= num_episodes:
            avg_reward = np.mean(all_rewards) if all_rewards else 0.0
            
            # Count milestone percentages
            n_episodes = len(all_flags)
            if n_episodes > 0:
                pct_reached = 100 * sum(1 for f in all_flags if f.get("reached")) / n_episodes
                pct_controlled = 100 * sum(1 for f in all_flags if f.get("controlled")) / n_episodes
                pct_lifted = 100 * sum(1 for f in all_flags if f.get("lifted")) / n_episodes
                pct_success = 100 * sum(1 for f in all_flags if f.get("success")) / n_episodes
            else:
                pct_reached = pct_controlled = pct_lifted = pct_success = 0.0
            
            print(f"[Iter {iteration:4d}] Episodes: {total_episodes:5d} | "
                  f"Avg Reward: {avg_reward:7.2f} | Loss: {loss:.4f}")
            print(f"           Reached: {pct_reached:5.1f}% | Controlled: {pct_controlled:5.1f}% | "
                  f"Lifted: {pct_lifted:5.1f}% | Success: {pct_success:5.1f}%")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    n_episodes = len(all_flags)
    if n_episodes > 0:
        avg_reward = np.mean(all_rewards)
        pct_reached = 100 * sum(1 for f in all_flags if f.get("reached")) / n_episodes
        pct_controlled = 100 * sum(1 for f in all_flags if f.get("controlled")) / n_episodes
        pct_lifted = 100 * sum(1 for f in all_flags if f.get("lifted")) / n_episodes
        pct_success = 100 * sum(1 for f in all_flags if f.get("success")) / n_episodes
        
        print(f"Average Episode Reward: {avg_reward:.2f}")
        print(f"% Reached:   {pct_reached:.1f}%")
        print(f"% Controlled: {pct_controlled:.1f}%")
        print(f"% Lifted:    {pct_lifted:.1f}%")
        print(f"% Success:   {pct_success:.1f}%")
    
    env.close()
    
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on pick-and-place task")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Steps per rollout")
    
    args = parser.parse_args()
    
    train_ppo(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        learning_rate=args.lr,
        rollout_steps=args.rollout_steps,
    )

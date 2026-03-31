"""
Visualize a trained PPO policy on the pick-and-place task.

Loads ppo_checkpoint_final.pt (or a specified checkpoint) and runs the policy
for N episodes with the Isaac Lab viewer enabled.

Run from repo root:
    python tests/visualize.py
    python tests/visualize.py --n 20 --delay 0.05 --deterministic
"""

import argparse
import os
import sys
import time

# Ensure repo root is on path so `lab` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple


# ── Policy (must match train_ppo.py exactly) ──────────────────────────────────

class TinyMLP(nn.Module):
    def __init__(self, obs_dim: int = 25, act_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.policy_mean = nn.Linear(hidden_dim, act_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        action_mean = self.policy_mean(self.actor_head(features))
        value = self.value(self.critic_head(features))
        with torch.no_grad():
            self.policy_log_std.clamp_(-0.15, 0.0)
        return action_mean, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        action_mean, value = self.forward(obs)
        if deterministic:
            return action_mean, value
        dist = Normal(action_mean, torch.exp(self.policy_log_std))
        return dist.sample(), value


class RunningObsNormalizer:
    def __init__(self, mean: torch.Tensor, var: torch.Tensor, count: float, clip: float = 10.0):
        self.mean = mean
        self.var = var
        self.count = count
        self.clip = clip

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize trained PPO policy")
    parser.add_argument("--checkpoint", type=str, default="ppo_checkpoint_final.pt",
                        help="Path to checkpoint file (default: ppo_checkpoint_final.pt)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of episodes to run (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode (default: 500, matches training episode length)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use mean actions instead of sampling")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Sleep (seconds) between steps to slow down playback. "
                             "e.g. --delay 0.05 for ~20 steps/sec (default: 0 = full speed)")
    parser.add_argument("--render-every", type=int, default=1,
                        help="Only render every N steps (default: 1). "
                             "Increase to speed up playback, e.g. --render-every 5")
    args = parser.parse_args()

    # Isaac Lab must be launched before importing env
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    from isaaclab.utils.logger import configure_logging
    configure_logging(save_logs_to_file=False)

    from lab.pick_place_env import PickPlaceEnv
    from lab.pick_place_env_cfg import PickPlaceEnvCfg

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args.n_envs

    policy = TinyMLP(obs_dim=cfg.observation_space, act_dim=cfg.action_space, hidden_dim=256)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()

    obs_normalizer = RunningObsNormalizer(
        mean=ckpt["obs_normalizer_mean"].to(device),
        var=ckpt["obs_normalizer_var"].to(device),
        count=ckpt["obs_normalizer_count"],
    )

    trained_episodes = ckpt.get("total_episodes", "unknown")
    step_delay = f"{args.delay*1000:.0f}ms/step" if args.delay > 0 else "full speed"
    print(f"[INFO] Checkpoint trained for {trained_episodes} episodes")
    print(f"[INFO] {args.n} episodes | deterministic={args.deterministic} | {step_delay}")

    # ── Create environment (with viewer) ─────────────────────────────────────
    env = PickPlaceEnv(cfg)

    # ── Episode loop ──────────────────────────────────────────────────────────
    results = {"reached": 0, "grasped": 0, "lifted": 0, "droppable": 0, "success": 0}
    total_reward = 0.0

    for ep in range(args.n):
        obs_dict, _ = env.reset()
        obs = obs_normalizer.normalize(obs_dict["policy"].to(device))

        ep_reward = 0.0
        ever_reached = False
        ever_grasped   = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
        ever_lifted    = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
        ever_droppable = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
        ever_success   = torch.zeros(args.n_envs, dtype=torch.bool, device=device)

        # Log active variant for this episode
        cube_var_idx = env._active_cube_variant[0].item()
        cup_var_idx  = env._active_cup_variant[0].item()
        cube_dims = cfg.cube_size_variants[cube_var_idx]
        cup_h     = cfg.cup_height_variants[cup_var_idx]
        print(f"\n[Ep {ep+1}/{args.n}] cube={tuple(round(v*100,2) for v in cube_dims)}cm  cup_h={cup_h*100:.1f}cm", flush=True)
        prev_flags = ""
        for step in range(args.max_steps):
            with torch.no_grad():
                action, _ = policy.get_action(obs, deterministic=args.deterministic)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_normalizer.normalize(obs_dict["policy"].to(device))
            ep_reward += reward[0].item()

            task_state = info.get("task_state", {})
            if "gripper_cube_distance" in task_state:
                ever_reached |= task_state["gripper_cube_distance"][0].item() < 0.15

            milestones = info.get("milestone_flags", {})
            ever_grasped   |= milestones.get("grasped",   torch.zeros_like(ever_grasped))
            ever_lifted    |= milestones.get("lifted",    torch.zeros_like(ever_lifted))
            ever_droppable |= milestones.get("droppable", torch.zeros_like(ever_droppable))
            ever_success   |= milestones.get("success",   torch.zeros_like(ever_success))

            # Per-step status line (overwrite in place)
            dist    = task_state.get("gripper_cube_distance", None)
            is_g    = task_state.get("is_grasped", None)
            cube_z  = task_state.get("cube_pos", None)
            g_width = task_state.get("gripper_width", None)
            dist_str    = f"dist={dist[0].item():.3f}" if dist is not None else ""
            height_str  = f"z={cube_z[0, 2].item():.3f}" if cube_z is not None else ""
            gripper_str = f"grip={g_width[0].item():.3f}" if g_width is not None else ""
            grasped_str = f"{'GRASPED' if (is_g is not None and is_g[0].item()) else '       '}"
            flags_str   = ("R" if ever_reached else ".") + \
                          ("G" if ever_grasped[0].item() else ".") + \
                          ("L" if ever_lifted[0].item() else ".") + \
                          ("D" if ever_droppable[0].item() else ".") + \
                          ("S" if ever_success[0].item() else ".")
            line = f"  s{step:>3d} | r={reward[0].item():+7.2f} | {dist_str} | {height_str} | {gripper_str} | {grasped_str} | {flags_str}"
            print(f"\r{line:<90}", end="", flush=True)
            # Print new line when a new milestone is reached
            if flags_str != prev_flags:
                print(flush=True)
                prev_flags = flags_str

            if terminated[0].item() or truncated[0].item():
                break

            if step % args.render_every == 0:
                simulation_app.update()

            if args.delay > 0:
                time.sleep(args.delay)

        if ever_reached:             results["reached"] += 1
        if ever_grasped[0].item():   results["grasped"] += 1
        if ever_lifted[0].item():    results["lifted"] += 1
        if ever_droppable[0].item(): results["droppable"] += 1
        if ever_success[0].item():   results["success"] += 1
        total_reward += ep_reward

        flags = ("R" if ever_reached else "") + \
                ("G" if ever_grasped[0].item() else "") + \
                ("L" if ever_lifted[0].item() else "") + \
                ("D" if ever_droppable[0].item() else "") + \
                ("S" if ever_success[0].item() else "")
        print(f"[Ep {ep+1:3d}/{args.n}] Reward: {ep_reward:7.1f} | {flags or '-'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"RESULTS ({args.n} episodes)")
    print("=" * 50)
    print(f"Avg reward:  {total_reward / args.n:.1f}")
    for key, count in results.items():
        print(f"{key.capitalize():<12} {count}/{args.n}  ({100*count/args.n:.0f}%)")

    env.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Minimal Isaac Lab test - bypasses our custom environment.
Tests if Isaac Lab itself works before testing our environment.

Usage:
    ~/IsaacLab/isaaclab.sh -p tests/debug_isaac_lab.py
"""

from __future__ import annotations
import argparse

# Parse args FIRST (Isaac Lab requirement)
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, unknown = parser.parse_known_args()

print("=" * 50)
print("STEP 1: Launching AppLauncher...")
print("=" * 50)

# Try different import patterns
try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

launcher = AppLauncher(headless=args.headless)
sim_app = launcher.app

print("AppLauncher ready!")
print()

# Now import torch and other modules
import torch
print("=" * 50)
print("STEP 2: Testing imports...")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Try importing Isaac Lab modules
try:
    import isaaclab.sim as sim_utils
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    print("Isaac Lab imports: OK (isaaclab.* pattern)")
except ImportError:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sim import SimulationCfg
    from omni.isaac.lab.utils import configclass
    print("Isaac Lab imports: OK (omni.isaac.lab.* pattern)")

print()
print("=" * 50)
print("STEP 3: Creating minimal environment...")
print("=" * 50)

@configclass
class MinimalEnvCfg(DirectRLEnvCfg):
    """Absolutely minimal config."""
    decimation = 2
    episode_length_s = 5.0
    action_space = 1
    observation_space = 1
    state_space = 0
    
    sim: SimulationCfg = SimulationCfg(dt=1/60)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=2.0)


class MinimalEnv(DirectRLEnv):
    """Minimal environment - just ground plane."""
    
    cfg: MinimalEnvCfg
    
    def _setup_scene(self):
        print("  _setup_scene: Adding ground plane...")
        try:
            from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        except ImportError:
            from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        print("  _setup_scene: Cloning environments...")
        self.scene.clone_environments(copy_from_source=False)
        print("  _setup_scene: Done!")
    
    def _pre_physics_step(self, actions):
        pass
    
    def _apply_action(self):
        pass
    
    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}
    
    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_dones(self):
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               self.episode_length_buf >= self.max_episode_length - 1
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)


print("Creating MinimalEnv...")
cfg = MinimalEnvCfg()
env = MinimalEnv(cfg)
print(f"Environment created! num_envs={env.num_envs}, device={env.device}")

print()
print("=" * 50)
print("STEP 4: Testing reset and step...")
print("=" * 50)

obs, _ = env.reset()
print(f"Reset OK, obs shape: {obs['policy'].shape}")

for i in range(5):
    actions = torch.zeros(env.num_envs, 1, device=env.device)
    obs, reward, term, trunc, info = env.step(actions)
    print(f"Step {i+1}: reward={reward.mean().item():.3f}")

print()
print("=" * 50)
print("STEP 5: Cleanup...")
print("=" * 50)
env.close()
sim_app.close()
print("SUCCESS! Isaac Lab is working.")

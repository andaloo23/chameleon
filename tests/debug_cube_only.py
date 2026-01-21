#!/usr/bin/env python3
"""
Standalone test for Robot + Cube ONLY (no prior environment creation).

Usage:
    ~/IsaacLab/isaaclab.sh -p tests/debug_cube_only.py --headless
"""

from __future__ import annotations
import argparse
import os
import sys

# Parse args FIRST
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, unknown = parser.parse_known_args()

print("=" * 60)
print("Robot + Cube STANDALONE Test")
print("=" * 60)

# Launch app
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=args.headless)
sim_app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, UrdfFileCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

# Get URDF path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "..", "so100.urdf")

print(f"\nURDF path: {URDF_PATH}")
print(f"URDF exists: {os.path.exists(URDF_PATH)}")

if not os.path.exists(URDF_PATH):
    print("ERROR: URDF not found!")
    sim_app.close()
    sys.exit(1)
print()


@configclass
class TestEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = 6
    observation_space = 9  # 6 joints + 3 cube xyz
    state_space = 0
    
    sim: SimulationCfg = SimulationCfg(dt=1/120)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=2.0)
    
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UrdfFileCfg(
            asset_path=URDF_PATH,
            fix_base=True,
            joint_drive=UrdfFileCfg.JointDriveCfg(
                gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None,
                    damping=None,
                ),
            ),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )
    
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.15, 0.02)),
    )


class TestEnv(DirectRLEnv):
    cfg: TestEnvCfg
    
    def __init__(self, cfg, **kwargs):
        print(">>> __init__ starting...")
        super().__init__(cfg, **kwargs)
        print(">>> __init__ complete!")
    
    def _setup_scene(self):
        import time
        t0 = time.time()
        print(f"  [{time.time()-t0:.2f}s] _setup_scene starting...")
        
        print(f"  [{time.time()-t0:.2f}s] Creating Articulation...")
        self.robot = Articulation(self.cfg.robot_cfg)
        print(f"  [{time.time()-t0:.2f}s] Articulation created!")
        
        print(f"  [{time.time()-t0:.2f}s] Creating RigidObject (cube)...")
        self.cube = RigidObject(self.cfg.cube_cfg)
        print(f"  [{time.time()-t0:.2f}s] RigidObject created!")
        
        print(f"  [{time.time()-t0:.2f}s] Adding ground...")
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        print(f"  [{time.time()-t0:.2f}s] Ground added!")
        
        print(f"  [{time.time()-t0:.2f}s] Cloning environments...")
        self.scene.clone_environments(copy_from_source=False)
        print(f"  [{time.time()-t0:.2f}s] Cloning done!")
        
        print(f"  [{time.time()-t0:.2f}s] Registering with scene...")
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        print(f"  [{time.time()-t0:.2f}s] Registration done!")
        
        print(f"  [{time.time()-t0:.2f}s] Adding light...")
        sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
        print(f"  [{time.time()-t0:.2f}s] _setup_scene COMPLETE!")
    
    def _pre_physics_step(self, actions):
        pass
    
    def _apply_action(self):
        pass
    
    def _get_observations(self):
        return {"policy": torch.cat([
            self.robot.data.joint_pos,
            self.cube.data.root_pos_w,
        ], dim=1)}
    
    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_dones(self):
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               self.episode_length_buf >= self.max_episode_length - 1
    
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)


print("Creating environment...")
try:
    cfg = TestEnvCfg()
    env = TestEnv(cfg)
    print(f"SUCCESS! num_envs={env.num_envs}")
    
    print("\nResetting...")
    obs, _ = env.reset()
    print(f"Reset OK, obs shape: {obs['policy'].shape}")
    
    print("\nStepping...")
    for i in range(5):
        actions = torch.zeros(env.num_envs, 6, device=env.device)
        obs, reward, term, trunc, info = env.step(actions)
        print(f"  Step {i+1}: reward={reward.mean().item():.3f}")
    
    print("\nClosing...")
    env.close()
    print("PASSED!")
    
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

sim_app.close()

import functools
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from load_scene import IsaacPickPlaceEnv

PolicyFn = Callable[[Dict[str, Any], int], np.ndarray]


@dataclass
class EpisodeResult:
    """Stores the results of a single simulation episode."""
    transitions: List[Dict[str, Any]]
    success: bool
    termination_reason: Optional[str] = None
    final_info: Dict[str, Any] = field(default_factory=dict)

    def to_trajectory(self) -> Dict[str, List[Any]]:
        """Convert transitions to a trajectory dictionary format."""
        traj = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "dones": [],
            "infos": [],
        }
        for t in self.transitions:
            traj["observations"].append(t["observation"])
            traj["actions"].append(t["action"])
            traj["rewards"].append(t["reward"])
            traj["next_observations"].append(t["next_observation"])
            traj["dones"].append(t["done"])
            traj["infos"].append(t["info"])
        return traj


def _default_random_policy(env: IsaacPickPlaceEnv, observation: Dict[str, Any], step: int) -> np.ndarray:
    """Default random policy fallback."""
    return env.robot.get_random_joint_positions()


class SimulationLoop:
    """Main simulation loop for data collection."""

    def __init__(
        self,
        env: Optional[IsaacPickPlaceEnv] = None,
        headless: bool = False,
        max_steps: int = 500,
        capture_images: bool = False,
        image_interval: int = 3,
        random_seed: Optional[int] = None,
        grasp_mode: str = "weld",
    ) -> None:
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("`max_steps` must be greater than zero.")
        grasp_mode = (grasp_mode or "sticky").strip().lower()
        if grasp_mode not in {"sticky", "physics", "weld"}:
            raise ValueError("grasp_mode must be 'sticky', 'physics', or 'weld'")
        self.grasp_mode = grasp_mode

        if env is None:
            self.env = IsaacPickPlaceEnv(
                headless=headless,
                capture_images=capture_images,
                image_interval=image_interval,
                random_seed=random_seed,
                grasp_mode=self.grasp_mode,
            )
            self._owns_env = True
        else:
            self.env = env
            self._owns_env = False
            env_mode = getattr(self.env, "grasp_mode", None)
            if env_mode is not None and env_mode != self.grasp_mode:
                print(f"[warn] SimulationLoop grasp_mode={self.grasp_mode} but env.grasp_mode={env_mode}; using env value.")
                self.grasp_mode = env_mode

        self.random_policy = functools.partial(_default_random_policy, self.env)

    def scripted_policy(self) -> PolicyFn:
        """IK-based scripted policy for the SO100 arm with corrected kinematics."""

        def _policy(observation: Dict[str, Any], step: int) -> np.ndarray:
            # Persistent state for the policy
            if not hasattr(_policy, "_state"):
                _policy._state = "APPROACH" # APPROACH -> GRASP -> LIFT
                _policy._waypoints = {}
                _policy._last_q = None

            cube_pos = observation.get("cube_pos")
            if cube_pos is None:
                return self.random_policy(observation, step)

            # Initialize waypoints at step 0
            if not _policy._waypoints or step == 0:
                print(f"\n[RMPFLOW] ========== Planning for cube at {cube_pos} ==========")
                robot_base_pos, _ = self.env.robot_articulation.get_world_pose()
                
                # Pre-grasp: 20cm above cube
                target_pre = np.array(cube_pos) + np.array([0, 0, 0.20])
                # Grasp: 2cm above cube center (accounting for jaw length)
                target_grasp = np.array(cube_pos) + np.array([0, 0, 0.02])
                # Lift: 30cm above table
                target_lift = np.array(cube_pos) + np.array([0, 0, 0.30])
                
                _policy._waypoints = {
                    "pre": target_pre,
                    "grasp": target_grasp,
                    "lift": target_lift
                }
                _policy._state = "APPROACH"
                print(f"[RMPFLOW] State: APPROACH | Target: {target_pre}")

            # State Transitions
            reward_engine = getattr(self.env, "reward_engine", None)
            stage_flags = getattr(reward_engine, "stage_flags", {}) if reward_engine is not None else {}
            grasped_flag = bool(stage_flags.get("grasped"))
            
            ee_pos, _ = self.env.robot_articulation.get_world_pose() # This is base pose, we need EE
            # Actually, RMPflow handles the EE pose internally, we just provide the target.
            
            # Use step counts as a simple fallback for state transitions if needed
            if _policy._state == "APPROACH" and step > 250:
                _policy._state = "GRASP"
                print(f"[RMPFLOW] State: GRASP | Target: {_policy._waypoints['grasp']}")
            elif _policy._state == "GRASP" and (grasped_flag or step > 600):
                _policy._state = "LIFT"
                print(f"[RMPFLOW] State: LIFT | Target: {_policy._waypoints['lift']}")

            # Compute RMP Action
            target_pos = _policy._waypoints.get("pre" if _policy._state == "APPROACH" else 
                                               "grasp" if _policy._state == "GRASP" else "lift")
            
            # Use a slightly downward orientation for the gripper
            # RMPflow orientation is [w, x, y, z]
            target_orient = np.array([0.707, 0, 0.707, 0]) # Pointing down
            
            next_q = self.env.compute_rmp_action(target_pos, target_orient)
            
            # Handle gripper manually
            if _policy._state == "APPROACH":
                next_q[5] = 1.2 # Wide open
            elif _policy._state == "GRASP":
                if step > 450 or grasped_flag:
                    next_q[5] = 0.05 # Close
                else:
                    next_q[5] = 1.2 # Stay open until near
            else:
                next_q[5] = 0.05 # Keep closed
                
            return next_q

        return _policy

    def run_episode(
        self,
        policy: Optional[PolicyFn] = None,
        reset: bool = True,
        render: bool = True,
    ) -> EpisodeResult:
        """Run a single episode of data collection."""
        if policy is None:
            policy = self.scripted_policy()

        observation = self.env.reset(render=render) if reset else self.env._get_observation()
        transitions = []
        success = False
        termination_reason = None
        final_info = {}

        try:
            for step in range(self.max_steps):
                action = policy(observation, step)
                next_observation, reward, done, info = self.env.step(action, render=render)

                transitions.append({
                    "observation": observation,
                    "action": action,
                    "reward": reward,
                    "next_observation": next_observation,
                    "done": done,
                    "info": info,
                })

                observation = next_observation
                final_info = info
                
                if done:
                    # Success flag is inside stage_flags
                    success = bool(info.get("stage_flags", {}).get("success", False))
                    termination_reason = info.get("termination_reason", "unknown")
                    print(f"Episode finished after {step+1} steps. Reason: {termination_reason} Success: {success}")
                    break
        except Exception as e:
            print(f"[error] An error occurred: {e}")
            import traceback
            traceback.print_exc()
            termination_reason = f"error: {str(e)}"
        finally:
            pass

        return EpisodeResult(
            transitions=transitions,
            success=success,
            termination_reason=termination_reason,
            final_info=final_info
        )

    def close(self) -> None:
        """Clean up the environment."""
        self.env.shutdown()


def run_collection(num_episodes: int = 5, headless: bool = False):
    """Run multiple episodes of data collection."""
    loop = SimulationLoop(headless=headless, capture_images=True)
    policy = loop.scripted_policy()

    for i in range(num_episodes):
        print(f"Starting episode {i+1}/{num_episodes}...")
        loop.run_episode(policy=policy, reset=True, render=True)

    loop.close()


if __name__ == "__main__":
    run_collection()

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
        """MoveIt-based scripted policy for the SO100 arm."""

        def _policy(observation: Dict[str, Any], step: int) -> np.ndarray:
            # Persistent state for the policy
            if not hasattr(_policy, "_state"):
                _policy._state = "APPROACH" # APPROACH -> GRASP -> LIFT
                _policy._trajectory = []
                _policy._traj_idx = 0

            cube_pos = observation.get("cube_pos")
            if cube_pos is None:
                return self.random_policy(observation, step)

            # Check if MoveIt interface is available
            moveit = getattr(self.env, "moveit", None)
            
            # Helper to plan to a target
            def plan_to(target_pos, quat=None):
                if not moveit: return None
                # Use the client's internal Pose helper to avoid ROS2 dependencies in 3.11
                from moveit_interface import Pose
                pose = Pose()
                pose.position.x = float(target_pos[0])
                pose.position.y = float(target_pos[1])
                pose.position.z = float(target_pos[2])
                # Downward orientation
                pose.orientation.x = 0.0
                pose.orientation.y = 0.707
                pose.orientation.z = 0.0
                pose.orientation.w = 0.707
                
                print(f"[MoveIt] Planning to {target_pos}...")
                return moveit.plan_to_pose(pose)

            # State Transitions and Planning
            reward_engine = getattr(self.env, "reward_engine", None)
            stage_flags = getattr(reward_engine, "stage_flags", {}) if reward_engine is not None else {}
            grasped_flag = bool(stage_flags.get("grasped"))

            # Plan new phase if trajectory is empty or state changed
            new_state = None
            if step == 0 or not _policy._trajectory or _policy._traj_idx >= len(_policy._trajectory):
                if _policy._state == "APPROACH" and (step > 10 or (not _policy._trajectory and step > 0)):
                    # Fallback to next state if approach is done (simple heuristic)
                    if step > 250:
                        new_state = "GRASP"
                elif _policy._state == "GRASP" and (grasped_flag or step > 600):
                    new_state = "LIFT"

                if new_state:
                    _policy._state = new_state
                    _policy._trajectory = []
                    _policy._traj_idx = 0
                    print(f"[MoveIt] Transitioned to {new_state}")

                # If no trajectory, plan it!
                if not _policy._trajectory or _policy._traj_idx >= len(_policy._trajectory):
                    target = None
                    if _policy._state == "APPROACH":
                        target = np.array(cube_pos) + np.array([0, 0, 0.20])
                    elif _policy._state == "GRASP":
                        target = np.array(cube_pos) + np.array([0, 0, 0.02])
                    elif _policy._state == "LIFT":
                        target = np.array(cube_pos) + np.array([0, 0, 0.30])
                    
                    if target is not None:
                        traj = plan_to(target)
                        if traj:
                            _policy._trajectory = traj
                            _policy._traj_idx = 0
                        else:
                            print(f"[MoveIt] Planning failed for {_policy._state}, using fallback IK")
                            # Fallback to single-point IK if planning fails
                            fallback_q = self.env.compute_ik(target)
                            _policy._trajectory = [fallback_q[:5].tolist()]
                            _policy._traj_idx = 0

            # Execute trajectory
            if _policy._trajectory and _policy._traj_idx < len(_policy._trajectory):
                arm_q = _policy._trajectory[_policy._traj_idx]
                _policy._traj_idx += 1
                
                # Construct full 6D action
                next_q = np.zeros(6, dtype=np.float32)
                next_q[:5] = arm_q
                
                # Handle gripper
                if _policy._state == "APPROACH":
                    next_q[5] = 1.2
                elif _policy._state == "GRASP":
                    next_q[5] = 0.05 if (step > 450 or grasped_flag) else 1.2
                else:
                    next_q[5] = 0.05
                    
                return next_q
            
            # Absolute fallback
            return self.env.robot_articulation.get_joint_positions()

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

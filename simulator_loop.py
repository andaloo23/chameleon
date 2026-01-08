import functools
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from load_scene import IsaacPickPlaceEnv

PolicyFn = Callable[[Dict[str, Any], int], np.ndarray]


class PolicyState:
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    LIFT = "LIFT"


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
        """RMPFlow-based scripted policy for the SO100 arm with scaled assets."""

        def _policy(observation: Dict[str, Any], step: int) -> np.ndarray:
            # Persistent state for the policy
            if not hasattr(_policy, "state"):
                _policy.state = PolicyState.APPROACH  # APPROACH -> GRASP -> LIFT
                _policy.state_timer = 0

            cube_pos = observation.get("cube_pos")
            gripper_pos = observation.get("gripper_pos")
            if cube_pos is None or gripper_pos is None:
                return self.random_policy(observation, step)

            # RMPFlow controller from env
            rmpflow = getattr(self.env, "rmpflow", None)
            motion_policy = getattr(self.env, "motion_policy", None)

            # State Transitions
            reward_engine = getattr(self.env, "reward_engine", None)
            stage_flags = getattr(reward_engine, "stage_flags", {}) if reward_engine is not None else {}
            grasped_flag = bool(stage_flags.get("grasped"))

            target_pos = None
            gripper_action = None # Default to None, will be set later

            if _policy.state == PolicyState.APPROACH:
                target_pos = np.array(cube_pos) + np.array([0, 0, 0.25])
                dist = np.linalg.norm(np.array(gripper_pos) - target_pos)
                if dist < 0.05:
                    print(f"[RMPFlow] Transitioned to GRASP")
                    _policy.state = PolicyState.GRASP
                    _policy.state_timer = 0
            elif _policy.state == PolicyState.GRASP:
                # Gradually lower to the cube instead of jumping
                alpha = min(1.0, _policy.state_timer / 40.0)
                approach_pos = np.array(cube_pos) + np.array([0, 0, 0.25])
                target_pos = (1 - alpha) * approach_pos + alpha * np.array(cube_pos)
                
                # Keep gripper OPEN while approaching/handling
                gripper_action = 0.05 
                _policy.state_timer += 1
                if _policy.state_timer > 60:
                    print(f"[RMPFlow] Transitioned to LIFT")
                    _policy.state = PolicyState.LIFT
                    _policy.state_timer = 0
            elif _policy.state == PolicyState.LIFT:
                target_pos = np.array(cube_pos) + np.array([0, 0, 0.25])
                # Keep gripper CLOSED while lifting
                gripper_action = 1.5

            if rmpflow and motion_policy and target_pos is not None:
                # Simplify for debugging: no orientation target for now
                rmpflow.set_end_effector_target(
                    target_position=target_pos,
                    target_orientation=None
                )
                
                # Update RMPFlow state with current joint positions/velocities
                # This is crucial for reactive control
                rmpflow.update_robot_state()
                
                # Get joint command
                articulation_action = motion_policy.get_next_articulation_action(step_size=1.0/60.0)
                
                if articulation_action and articulation_action.joint_positions is not None:
                    # RMPFlow handles the first 5 degrees of freedom (cspace)
                    arm_q = articulation_action.joint_positions[:5]
                else:
                    raise RuntimeError(f"RMPFlow failed to generate an action at step {step}")
            else:
                raise RuntimeError("RMPFlow is not initialized or target position is missing")

            # Construct full 6D action
            next_q = np.zeros(6, dtype=np.float32)
            next_q[:5] = arm_q

            # Handle gripper
            if gripper_action is not None:
                next_q[5] = gripper_action
            else:
                # Fallback logic if gripper_action wasn't set (shouldn't happen with current states)
                if _policy.state == PolicyState.APPROACH:
                    next_q[5] = 1.2  # Open gripper
                elif _policy.state == PolicyState.GRASP:
                    next_q[5] = 0.05 if (step > 300 or grasped_flag) else 1.2  # Close gripper
                else:
                    next_q[5] = 0.05  # Keep gripper closed during lift

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

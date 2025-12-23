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
            # Persistent storage for IK targets
            if not hasattr(_policy, "_ik_targets"):
                _policy._ik_targets = None
                _policy._lift_start_step = None
                _policy._debug_printed = False

            cube_pos = observation.get("cube_pos")
            if cube_pos is None:
                return self.random_policy(observation, step)

            # Initialize IK targets at step 0
            if _policy._ik_targets is None or step == 0:
                print(f"\n[POLICY] ========== Computing IK for cube at {cube_pos} ==========")
                
                # Get robot base position
                robot_base_pos, _ = self.env.robot_articulation.get_world_pose()
                print(f"[POLICY] Robot base at: {robot_base_pos}")
                
                # Pan angle to the cube
                dx = cube_pos[0] - robot_base_pos[0]
                dy = cube_pos[1] - robot_base_pos[1]
                pan_angle = np.arctan2(dx, -dy)
                
                # Small offset to center gripper on cube
                side_offset = -0.015  # 1.5cm offset
                off_x = side_offset * np.cos(pan_angle)
                off_y = side_offset * np.sin(pan_angle)
                
                target_grasp = np.array(cube_pos) + np.array([off_x, off_y, 0.02])  # Slightly above cube center
                
                # Pre-grasp: 15cm above grasp point
                target_pre = target_grasp.copy()
                target_pre[2] += 0.15
                
                # Lift: 25cm above table
                target_lift = target_grasp.copy()
                target_lift[2] = 0.25
                
                print(f"[POLICY] Target positions - Pre: z={target_pre[2]:.3f}, Grasp: z={target_grasp[2]:.3f}, Lift: z={target_lift[2]:.3f}")
                
                # Home position: arm raised (LOWER lift = arm UP)
                q_home = np.array([0.0, 1.2, -1.0, 0.0, 0.0, 1.0])
                
                # Solve IK for each waypoint
                print("[POLICY] Computing IK for pre-grasp...")
                q_pre = self.env.compute_ik(target_pre, initial_q=q_home)
                print("[POLICY] Computing IK for grasp...")
                q_grasp = self.env.compute_ik(target_grasp, initial_q=q_pre)
                print("[POLICY] Computing IK for lift...")
                q_lift = self.env.compute_ik(target_lift, initial_q=q_grasp)
                
                # Set gripper states
                q_home[5] = 1.0   # Open
                q_pre[5] = 1.2    # Wide open for approach
                q_grasp[5] = 0.05 # Closed to grasp
                q_lift[5] = 0.05  # Keep closed
                
                _policy._ik_targets = {
                    "home": q_home,
                    "pre": q_pre,
                    "grasp": q_grasp,
                    "lift": q_lift,
                }
                
                print(f"[POLICY] Final joint targets:")
                print(f"  Home:  lift={q_home[1]:.2f}, elbow={q_home[2]:.2f}")
                print(f"  Pre:   lift={q_pre[1]:.2f}, elbow={q_pre[2]:.2f}")
                print(f"  Grasp: lift={q_grasp[1]:.2f}, elbow={q_grasp[2]:.2f}")
                print(f"  Lift:  lift={q_lift[1]:.2f}, elbow={q_lift[2]:.2f}")
                print("[POLICY] ======================================================\n")

            # Get current state
            reward_engine = getattr(self.env, "reward_engine", None)
            stage_flags = getattr(reward_engine, "stage_flags", {}) if reward_engine is not None else {}
            grasped_flag = bool(stage_flags.get("grasped"))
            
            targets = _policy._ik_targets
            current_q = np.array(observation.get("joint_positions"), dtype=np.float32)

            # Debug: print joint positions periodically
            if step % 50 == 0:
                print(f"[POLICY] Step {step}: current lift={current_q[1]:.2f}, elbow={current_q[2]:.2f}, "
                      f"target_pre_lift={targets['pre'][1]:.2f}")

            # Stage 1: Move to pre-grasp position (0-150 steps)
            if step < 150:
                target_q = targets["pre"].copy()
                # Use higher alpha for faster convergence, but smooth start
                alpha = min(0.15, 0.02 + step * 0.001)
                result = current_q * (1 - alpha) + target_q * alpha
                return result

            # Stage 2: Descend to grasp (150-350 steps)
            elif step < 350 and not grasped_flag:
                target_q = targets["grasp"].copy()
                progress = (step - 150) / 200.0
                
                # Gradually close gripper during descent
                if progress > 0.7:
                    target_q[5] = 0.05
                
                alpha = 0.12
                result = current_q * (1 - alpha) + target_q * alpha
                return result

            # Stage 3: Lift (350+ steps or after grasp detected)
            else:
                if _policy._lift_start_step is None:
                    _policy._lift_start_step = step
                    print(f"[POLICY] === LIFT STAGE at step {step} ===")
                
                target_q = targets["lift"].copy()
                alpha = 0.10
                result = current_q * (1 - alpha) + target_q * alpha
                return result

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

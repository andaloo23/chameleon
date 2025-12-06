from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from load_scene import IsaacPickPlaceEnv


class PolicyFn(Protocol):
    """Callable protocol for policies used by the simulation loop."""

    def __call__(self, observation: Dict[str, Any], step: int) -> Sequence[float]: ...


@dataclass
class Transition:
    """Single environment transition collected during a rollout."""

    observation: Dict[str, Any]
    action: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    next_observation: Optional[Dict[str, Any]] = None


@dataclass
class EpisodeResult:
    """Container holding rollout data and summary statistics."""

    transitions: List[Transition]
    cumulative_reward: float
    success: bool
    termination_reason: Optional[str] = None
    final_info: Dict[str, Any] = field(default_factory=dict)

    def to_trajectory(self) -> Dict[str, Any]:
        """Return a light-weight dict representation useful for logging/serialization."""
        obs = [t.observation for t in self.transitions]
        next_obs = [t.next_observation for t in self.transitions]
        actions = [t.action for t in self.transitions]
        rewards = [t.reward for t in self.transitions]
        dones = [t.done for t in self.transitions]
        infos = [t.info for t in self.transitions]
        return {
            "observations": obs,
            "next_observations": next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
            "cumulative_reward": self.cumulative_reward,
            "success": self.success,
            "termination_reason": self.termination_reason,
        }


def _default_random_policy(env: IsaacPickPlaceEnv, observation: Dict[str, Any], step: int) -> np.ndarray:
    """Fallback random policy that samples joint targets within the robot limits."""
    joint_names = getattr(env.robot, "joint_names", [])
    joint_limits = getattr(env.robot, "joint_limits", {})
    joint_positions = observation.get("joint_positions")

    noisy_target: List[float] = []
    for idx, name in enumerate(joint_names):
        lower, upper = joint_limits.get(name, (-np.pi, np.pi))
        span = max(upper - lower, 1e-6)
        margin = min(max(0.05 * span, 0.01), span / 2.0 - 1e-4)
        margin = max(margin, 1e-3)
        low = lower + margin
        high = upper - margin
        center = 0.5 * (lower + upper)
        amplitude = min(0.2 * span, (high - low) * 0.5)
        if amplitude <= 0.0:
            proposed = float(np.clip(center, low, high))
        else:
            delta = np.random.uniform(-amplitude, amplitude)
            proposed = float(np.clip(center + delta, low, high))
        noisy_target.append(proposed)
    return np.asarray(noisy_target, dtype=np.float32)


class SimulationLoop:
    """High-level rollout harness around `IsaacPickPlaceEnv`.

    This helper wires together episode resets, policy execution, domain randomization and
    reward bookkeeping so downstream training code can focus on the policy logic itself.
    """

    def __init__(
        self,
        env: Optional[IsaacPickPlaceEnv] = None,
        *,
        max_steps: int = 360,
        headless: bool = True,
        capture_images: bool = False,
        image_interval: int = 3,
        random_seed: Optional[int] = None,
    ) -> None:
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("`max_steps` must be greater than zero.")

        if env is None:
            self.env = IsaacPickPlaceEnv(
                headless=headless,
                capture_images=capture_images,
                image_interval=image_interval,
                random_seed=random_seed,
            )
            self._owns_env = True
        else:
            self.env = env
            self._owns_env = False

        self.random_policy = functools.partial(_default_random_policy, self.env)

    def scripted_policy(self) -> PolicyFn:
        """Return a simple reach-grasp-lift-place heuristic controller.
        
        Uses a known working grasp configuration as reference:
        - shoulder_pan: 0.0, shoulder_lift: -0.43, elbow_flex: 1.15, 
        - wrist_flex: 0.35, wrist_roll: 0.0, gripper: 0.0 (closed)
        """

        def _policy(observation: Dict[str, Any], step: int) -> np.ndarray:
            joint_positions = observation.get("joint_positions")
            cube_pos = observation.get("cube_pos")
            gripper_pos = observation.get("gripper_pos")
            
            if joint_positions is None:
                return self.random_policy(observation, step)

            joint_positions = np.asarray(joint_positions, dtype=np.float32)
            target = joint_positions.copy()

            robot = getattr(self.env, "robot", None)
            joint_names = getattr(robot, "joint_names", [])
            joint_limits = getattr(robot, "joint_limits", {})
            name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

            gripper_idx = name_to_idx.get("gripper")
            reward_engine = getattr(self.env, "reward_engine", None)
            stage_flags = getattr(reward_engine, "stage_flags", {}) if reward_engine is not None else {}
            grasped_flag = bool(stage_flags.get("grasped"))

            # Calculate cube-relative adjustments if cube position is available
            cube_offset_pan = 0.0
            cube_adapted_lift = 1.8  # Default grasp height (HIGHER - pinch from above, not wrap around)
            cube_adapted_elbow = -1.0  # Default elbow bend (less bent, stay higher)
            
            if cube_pos is not None and gripper_pos is not None:
                cube_pos = np.asarray(cube_pos, dtype=np.float32)
                gripper_pos = np.asarray(gripper_pos, dtype=np.float32)
                
                # Adjust shoulder_pan based on cube X position (left/right)
                cube_offset_pan = cube_pos[0] * 1.2
                
                # Adjust shoulder_lift and elbow based on cube Y position (forward/back)
                # Cube Y is typically around -0.5 (now positioned further forward)
                # More negative Y = cube is further forward
                # shoulder_lift controls reach: HIGHER value = ARM GOES DOWN/FORWARD
                cube_y = cube_pos[1]
                
                # Mapping: cube at -0.5 should use lift~1.8 (STAY HIGH - pinch from above)
                # If cube is MORE negative (further): INCREASE lift slightly
                # If cube is LESS negative (closer): DECREASE lift  
                # Strategy: Approach from ABOVE and pinch, don't try to wrap around
                cube_adapted_lift = np.clip(1.8 + (cube_y + 0.5) * 1.5, 1.5, 2.2)
                cube_adapted_elbow = np.clip(-1.0 + (cube_y + 0.5) * 1.0, -1.4, -0.8)
                
                if step == 0:
                    print(f"[DEBUG] Cube at Y={cube_y:.3f} Z={cube_pos[2]:.3f}, adapted: lift={cube_adapted_lift:.3f}, elbow={cube_adapted_elbow:.3f}")

            def safe_clip(value, joint_name):
                """Clip joint value to safe limits with margin."""
                lower, upper = joint_limits.get(joint_name, (-np.pi, np.pi))
                margin = 0.05 if joint_name != "gripper" else 0.002
                return np.clip(value, lower + margin, upper - margin)

            # IMPORTANT: Joint angles must respect URDF limits!
            # shoulder_lift: 0 (highest) to 3.5 (lowest) - POSITIVE ONLY!
            # elbow_flex: -3.14 (bent) to 0 (straight) - NEGATIVE ONLY!
            
            # Grasp configuration - adapted to cube position
            # Note: shoulder_lift values are POSITIVE (0=high, 3.5=low)
            # Strategy: PINCH from above with minimal gripper closure
            grasp_config = np.array([
                cube_offset_pan,      # shoulder_pan (track cube X)
                cube_adapted_lift,    # shoulder_lift (adapted to cube Y - STAYS HIGH)
                cube_adapted_elbow,   # elbow_flex (adapted to cube Y)
                0.5,                  # wrist_flex
                0.0,                  # wrist_roll
                0.12,                 # gripper (PINCH - just enough to hold, cube is 0.075m)
            ], dtype=np.float32)
            
            # High starting position (arm up and back)
            home_config = np.array([
                0.0,    # shoulder_pan
                0.8,    # shoulder_lift (high - LOW VALUE = higher position)
                -1.0,   # elbow_flex (MUST BE NEGATIVE!)
                0.3,    # wrist_flex
                0.0,    # wrist_roll
                0.3,    # gripper (WIDE OPEN for 2.5x cube - was 0.039)
            ], dtype=np.float32)
            
            # Lift configuration (arm high with cube)
            lift_config = np.array([
                0.0,    # shoulder_pan
                0.5,    # shoulder_lift (very high - LOW VALUE)
                -0.8,   # elbow_flex (MUST BE NEGATIVE! less bent)
                0.4,    # wrist_flex
                0.0,    # wrist_roll
                0.12,   # gripper (PINCHING - minimal closure)
            ], dtype=np.float32)

            # Policy stages - FASTER pace to complete within 360 steps
            # Steps 0-100: Move toward cube XY while staying very high (gripper open)
            # Steps 100-130: Hover and stabilize
            # Steps 130-230: Descend carefully to grasp position
            # Steps 230-250: Close gripper
            # Steps 250+: Lift if grasped (110 steps remaining!)
            
            # Use current joint position to compute smooth incremental changes
            current_pos = joint_positions.copy()
            
            if step < 100:
                # Stage 1: Move toward cube XY while staying very high (FASTER)
                progress = step / 100.0
                target_config = home_config.copy()
                target_config[0] = safe_clip(cube_offset_pan * progress, "shoulder_pan")
                # home_config already has shoulder_lift=0.8 (high position)
                
                # Smooth blend: slightly faster
                alpha = 0.07
                target = current_pos * (1 - alpha) + target_config[:len(current_pos)] * alpha
                
            elif step < 130:
                # Stage 2: Hover and stabilize (ensure proper XY alignment) - SHORTER
                target_config = home_config.copy()
                target_config[0] = safe_clip(cube_offset_pan, "shoulder_pan")
                # home_config already has shoulder_lift=0.8 (high position)
                
                # Smooth blend
                alpha = 0.05
                target = current_pos * (1 - alpha) + target_config[:len(current_pos)] * alpha
                
            elif step < 230:
                # Stage 3: Descend carefully from high position to grasp position
                progress = (step - 130) / 100.0  # 100 steps for descent (was 120)
                target_config = home_config.copy()
                target_config[0] = safe_clip(cube_offset_pan, "shoulder_pan")
                # Interpolate from high (0.8) to cube-adapted grasp position
                # Note: shoulder_lift increases as arm descends (0=highest, 3.5=lowest)
                target_config[1] = safe_clip(
                    0.8 * (1 - progress) + cube_adapted_lift * progress, 
                    "shoulder_lift"
                )
                target_config[2] = safe_clip(
                    -1.0 * (1 - progress) + cube_adapted_elbow * progress,
                    "elbow_flex"
                )
                target_config[5] = safe_clip(0.3, "gripper")  # Keep WIDE OPEN
                
                # Smooth motion but slightly faster
                alpha = 0.03  # Move 3% toward target each step (was 0.02)
                target = current_pos * (1 - alpha) + target_config[:len(current_pos)] * alpha
                
            elif step < 250:
                # Stage 4: Close gripper at grasp position (FASTER)
                progress = (step - 230) / 20.0
                target_config = grasp_config.copy()
                target_config[0] = safe_clip(cube_offset_pan, "shoulder_pan")
                # Gradually close gripper from wide open (0.3) to PINCH (0.12)
                # PINCH strategy: barely close around cube (0.075m), don't fully wrap
                target_config[5] = safe_clip(
                    0.3 * (1 - progress) + 0.12 * progress,
                    "gripper"
                )
                
                # Smooth blend (slightly faster)
                alpha = 0.05
                target = current_pos * (1 - alpha) + target_config[:len(current_pos)] * alpha
                
            else:
                # Stage 5: Lift if grasped (starts at step 250, 110 steps to complete!)
                if grasped_flag:
                    # Lift up with cube
                    progress = min(1.0, (step - 250) / 80.0)
                    target_config = grasp_config.copy()
                    target_config[0] = safe_clip(cube_offset_pan, "shoulder_pan")
                    # Interpolate from cube-adapted grasp position to lift position (0.5)
                    # Note: shoulder_lift decreases as arm lifts up
                    target_config[1] = safe_clip(
                        cube_adapted_lift * (1 - progress) + 0.5 * progress,
                        "shoulder_lift"
                    )
                    target_config[2] = safe_clip(
                        cube_adapted_elbow * (1 - progress) + -0.8 * progress,
                        "elbow_flex"
                    )
                    target_config[3] = safe_clip(
                        0.5 * (1 - progress) + 0.4 * progress,
                        "wrist_flex"
                    )
                    target_config[5] = safe_clip(0.12, "gripper")  # Keep PINCHING
                    
                    # Smooth blend (slightly faster for lifting)
                    alpha = 0.04
                    target = current_pos * (1 - alpha) + target_config[:len(current_pos)] * alpha
                else:
                    # Stay at grasp position, wait for grasp detection
                    target_config = grasp_config.copy()
                    target_config[0] = safe_clip(cube_offset_pan, "shoulder_pan")
                    target_config[5] = safe_clip(0.12, "gripper")  # Keep PINCHING
                    
                    # Smooth blend
                    alpha = 0.04
                    target = current_pos * (1 - alpha) + target_config[:len(current_pos)] * alpha

            if step % 60 == 0 and gripper_idx is not None:
                cube_str = f"Cube=[{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]" if cube_pos is not None else "Cube=N/A"
                grip_str = f"Grip=[{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]" if gripper_pos is not None else "Grip=N/A"
                dist = np.linalg.norm(cube_pos - gripper_pos) if (cube_pos is not None and gripper_pos is not None) else 0.0
                print(f"[DEBUG] Step {step} | {cube_str} | {grip_str} | Dist={dist:.3f}")
                print(f"[DEBUG] Step {step} Target: lift={target[1]:.4f} pan={target[0]:.4f} elbow={target[2]:.4f} grip={target[gripper_idx]:.4f}")
                print(f"[DEBUG] Step {step} Adapted: cube_lift={cube_adapted_lift:.3f} cube_elbow={cube_adapted_elbow:.3f} cube_pan={cube_offset_pan:.3f} | Grasped={grasped_flag}")

            return target

        return _policy

    def run_episode(
        self,
        policy: Optional[PolicyFn] = None,
        *,
        reset: bool = True,
        render: Optional[bool] = None,
    ) -> EpisodeResult:
        """Execute a single rollout using the provided policy callable.

        Args:
            policy: Callable mapping (observation, step) -> action. Defaults to a random policy.
            reset: Whether to reset the environment before starting the rollout.
            render: Optional override to force rendering on/off inside the env.step loop.

        Returns:
            EpisodeResult describing the collected trajectory.
        """
        policy_fn: PolicyFn
        if policy is None:
            policy_fn = self.random_policy
        else:
            policy_fn = policy

        if reset:
            try:
                observation = self.env.reset(render=render)
            except TypeError:
                observation = self.env.reset()
        else:
            observation = self.env._get_observation()

        transitions: List[Transition] = []
        cumulative_reward = 0.0
        final_info: Dict[str, Any] = {}
        termination_reason: Optional[str] = None

        for step_idx in range(self.max_steps):
            action = np.asarray(policy_fn(observation, step_idx), dtype=np.float32)
            next_observation, reward, done, info = self.env.step(action, render=render)

            transitions.append(
                Transition(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    done=bool(done),
                    info=dict(info),
                    next_observation=next_observation,
                )
            )

            cumulative_reward += float(reward)
            final_info = dict(info)
            if done:
                termination_reason = info.get("termination_reason")
                observation = next_observation
                break

            observation = next_observation

        success = bool(final_info.get("stage_flags", {}).get("success", False))
        return EpisodeResult(
            transitions=transitions,
            cumulative_reward=cumulative_reward,
            success=success,
            termination_reason=termination_reason,
            final_info=final_info,
        )

    def collect(
        self,
        num_episodes: int,
        policy: Optional[PolicyFn] = None,
        *,
        progress_callback: Optional[Callable[[int, EpisodeResult], None]] = None,
        render: Optional[bool] = None,
    ) -> List[EpisodeResult]:
        """Collect multiple rollouts in sequence."""
        results: List[EpisodeResult] = []
        for episode_idx in range(int(num_episodes)):
            result = self.run_episode(policy=policy, reset=True, render=render)
            results.append(result)
            if progress_callback is not None:
                progress_callback(episode_idx, result)
        return results

    def close(self) -> None:
        """Dispose of the underlying environment."""
        if self._owns_env and hasattr(self.env, "shutdown"):
            self.env.shutdown()

    def __enter__(self) -> "SimulationLoop":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

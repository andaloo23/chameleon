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
        """Return a simple reach-grasp-lift-place heuristic controller."""

        def _policy(observation: Dict[str, Any], step: int) -> np.ndarray:
            joint_positions = observation.get("joint_positions")
            if joint_positions is None:
                return self.random_policy(observation, step)

            joint_positions = np.asarray(joint_positions, dtype=np.float32)
            target = joint_positions.copy()

            robot = getattr(self.env, "robot", None)
            joint_names = getattr(robot, "joint_names", [])
            joint_limits = getattr(robot, "joint_limits", {})
            name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

            gripper_idx = name_to_idx.get("gripper")

            # Base config for the robot
            base_config = np.array([
                0.0,   # shoulder_pan
                -0.35,  # shoulder_lift (start high)
                1.15,   # elbow_flex (default)
                0.35,   # wrist_flex
                0.0,    # wrist_roll
                0.035 if gripper_idx is not None else 0.0,
            ], dtype=np.float32)
            base_config = base_config[: len(target)]

            def apply_offsets(base: np.ndarray, offsets: Dict[str, float]) -> np.ndarray:
                updated = base.copy()
                for name, delta in offsets.items():
                    idx = name_to_idx.get(name)
                    if idx is None or idx >= len(updated):
                        continue
                    lower, upper = joint_limits.get(name, (-np.pi, np.pi))
                    margin = 0.05 if name != "gripper" else 0.002
                    updated[idx] = np.clip(base[idx] + delta, lower + margin, upper - margin)
                return updated

            # Define key poses (offsets from base_config)
            # Stage 2 Grasp target:
            # pan: -0.05 (Correct X offset)
            # lift: 0.0 (Correct Z height to ~0.015)
            # elbow: 0.0 offset (Keep 1.15 to reach Y ~ -0.336)
            # gripper: 0.025 (2.5cm for 3cm cube)

            # Stage 3 Lift target:
            # lift: -0.4 (Lift up)
            # elbow: -0.1 (Pull back to 1.05)

            # Smooth interpolation for Lift stage (Stage 3, steps 180+)
            if step >= 180:
                # Interpolate shoulder_lift and elbow
                progress = min(1.0, (step - 180) / 50.0)
                
                # Start from Grasp pose
                start_lift = 0.0
                end_lift = -0.4
                
                start_elbow = 0.0 # relative to 1.15 base
                end_elbow = -0.1  # relative to 1.15 base -> 1.05
                
                current_lift = start_lift + (end_lift - start_lift) * progress
                current_elbow = start_elbow + (end_elbow - start_elbow) * progress
                
                target = apply_offsets(base_config, {
                    "shoulder_pan": -0.05,
                    "shoulder_lift": current_lift, 
                    "elbow_flex": current_elbow,
                    "wrist_flex": 0.0, # default 0.35
                    "gripper": 0.025   # keep closed
                })
                
                # Explicit gripper override
                if gripper_idx is not None:
                    target[gripper_idx] = 0.025
                return target

            # Discrete stages
            stage_configs = (
                # Stage 0: Hover HIGH above
                (apply_offsets(base_config, {"shoulder_lift": -0.6}), 0.04),
                # Stage 1: Pre-grasp (lower)
                (apply_offsets(base_config, {"shoulder_lift": -0.2}), 0.04),
                # Stage 2: Grasp (Action!)
                # Set absolute targets via offsets
                (apply_offsets(base_config, {
                    "shoulder_pan": -0.05, # Turn Right
                    "shoulder_lift": 0.0,  # Go Low (Z~0.015)
                    "elbow_flex": 0.0,     # Extend Y (1.15)
                    "wrist_flex": 0.0      # default
                }), 0.025), # Gripper 2.5cm
            )

            # Map steps to stages: 0-60 (Stage 0), 60-100 (Stage 1), 100-180 (Stage 2)
            if step < 60:
                stage = 0
            elif step < 100:
                stage = 1
            else:
                stage = 2
            
            stage_target, gripper_value = stage_configs[stage]
            target[: len(stage_target)] = stage_target
            if gripper_idx is not None and gripper_idx < len(target):
                lower, upper = joint_limits.get("gripper", (0.0, 0.04))
                margin = 0.001
                safe_value = np.clip(gripper_value, lower + margin, upper - margin)
                target[gripper_idx] = safe_value

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
                # Fallback for environments that don't accept render in reset
                observation = self.env.reset()
        else:
            observation = self.env._get_observation()  # type: ignore[attr-defined]

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

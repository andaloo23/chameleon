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
            reward_engine = getattr(self.env, "reward_engine", None)
            stage_flags = getattr(reward_engine, "stage_flags", {}) if reward_engine is not None else {}
            grasped_flag = bool(stage_flags.get("grasped"))

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
            stage_configs = (
                # Stage 0: Hover HIGH above
                (apply_offsets(base_config, {"shoulder_lift": -0.6}), 0.04),
                # Stage 1: Pre-grasp (bring arm closer but stay open)
                (apply_offsets(base_config, {"shoulder_lift": -0.2}), 0.04),
                # Stage 2a: Aggressive approach (open gripper)
                (apply_offsets(base_config, {
                    "shoulder_pan": -0.005,
                    "shoulder_lift": 0.70,
                    "elbow_flex": -1.15,
                    "wrist_flex": -0.55,
                }), 0.04),
                # Stage 2b: Close while staying low
                (apply_offsets(base_config, {
                    "shoulder_pan": -0.005,
                    "shoulder_lift": 0.85,
                    "elbow_flex": -1.25,
                    "wrist_flex": -0.70,
                }), 0.004),
                # Stage 2c: Press down slightly with firm grip
                (apply_offsets(base_config, {
                    "shoulder_pan": -0.005,
                    "shoulder_lift": 0.95,
                    "elbow_flex": -1.32,
                    "wrist_flex": -0.78,
                }), 0.002),
            )

            press_pose = stage_configs[-1][0]
            lift_force_step = 300
            lift_ready_step = 180
            ready_to_lift = grasped_flag and step >= lift_ready_step
            forced_lift = step >= lift_force_step

            if ready_to_lift or forced_lift:
                lift_anchor_step = lift_ready_step if ready_to_lift else lift_force_step
                progress = min(1.0, (step - lift_anchor_step) / 80.0)
                lift_target_pose = apply_offsets(base_config, {
                    "shoulder_pan": -0.015,
                    "shoulder_lift": -0.45,
                    "elbow_flex": -0.15,
                    "wrist_flex": 0.10,
                })
                target = press_pose + (lift_target_pose - press_pose) * progress
                if gripper_idx is not None:
                    target[gripper_idx] = 0.002
            else:
                # Map steps to stages:
                # 0-60: Stage 0
                # 60-100: Stage 1
                # 100-140: Stage 2a (Approach)
                # 140-170: Stage 2b (Close)
                # >=170: Stage 2c (Press) while waiting for grasp
                if step < 60:
                    stage = 0
                elif step < 100:
                    stage = 1
                elif step < 140:
                    stage = 2
                elif step < 170:
                    stage = 3
                else:
                    stage = 4

                stage_target, gripper_value = stage_configs[stage]
                target = stage_target.copy()
                if gripper_idx is not None and gripper_idx < len(target):
                    lower, upper = joint_limits.get("gripper", (0.0, 0.04))
                    margin = 0.001
                    safe_value = np.clip(gripper_value, lower + margin, upper - margin)
                    target[gripper_idx] = safe_value

            if step % 60 == 0:
                 # Print the actual target for shoulder_lift (idx 1) and shoulder_pan (idx 0)
                 print(f"[DEBUG] Step {step} Target: lift={target[1]:.4f} pan={target[0]:.4f} grip={target[gripper_idx] if gripper_idx else 0:.4f}")

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

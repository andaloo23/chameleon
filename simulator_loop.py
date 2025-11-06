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
    try:
        return np.array(env.robot.get_random_joint_positions(), dtype=np.float32)
    except Exception:
        joint_positions = observation.get("joint_positions")
        if joint_positions is not None:
            noise = np.random.uniform(-0.05, 0.05, size=len(joint_positions))
            return np.asarray(joint_positions + noise, dtype=np.float32)
        return np.zeros(len(env.robot.joint_names), dtype=np.float32)


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

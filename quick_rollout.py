"""Utility script to sanity-check the simulator loop with a random policy."""

from __future__ import annotations

import argparse
import json

import numpy as np

from simulator_loop import EpisodeResult, SimulationLoop


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def summarize_episode(idx: int, result: EpisodeResult, verbose: bool = False) -> dict:
    summary = {
        "episode": idx,
        "steps": len(result.transitions),
        "cumulative_reward": result.cumulative_reward,
        "success": result.success,
        "termination_reason": result.termination_reason,
        "validation_ok": bool(result.final_info.get("validation", {}).get("ok", True)),
    }
    if verbose:
        components = result.final_info.get("reward_components")
        if components:
            summary["reward_components"] = components
        stage_flags = result.final_info.get("stage_flags")
        if stage_flags:
            summary["stage_flags"] = stage_flags
    return summary


def run(num_episodes: int, max_steps: int, headless: bool, verbose: bool, policy: str, grasp_mode: str) -> None:
    with SimulationLoop(max_steps=max_steps, headless=headless, grasp_mode=grasp_mode) as loop:
        if policy == "heuristic":
            policy_fn = loop.scripted_policy()
        else:
            policy_fn = None

        for ep_idx in range(num_episodes):
            result = loop.run_episode(policy=policy_fn)
            summary = summarize_episode(ep_idx, result, verbose=verbose)
            print(json.dumps(summary, indent=2, default=_json_default), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--max-steps", type=int, default=360, help="Maximum steps per episode.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--verbose", action="store_true", help="Include reward/stage details in the output.")
    parser.add_argument("--policy", choices=("random", "heuristic"), default="random", help="Which policy to run.")
    parser.add_argument(
        "--grasp-mode",
        choices=("sticky", "physics"),
        default="sticky",
        help="Choose sticky attachment or physics-based grasping.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        verbose=args.verbose,
        policy=args.policy,
        grasp_mode=args.grasp_mode,
    )

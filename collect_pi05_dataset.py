#!/usr/bin/env python3
"""
Collect successful PPO rollouts and export them as a LeRobot dataset for PI0.5.

The exported dataset intentionally records only inference-time observations:
two RGB cameras plus robot proprioception. It does not store privileged cube/cup
 state from the PPO observation vector.

Example:
    ~/IsaacLab/isaaclab.sh -p collect_pi05_dataset.py \
        --checkpoint ppo_checkpoint_final.pt \
        --repo-id yourname/so100_pick_place_pi05 \
        --root ./datasets \
        --episodes 200 \
        --n-envs 4

Training example after collection:
    lerobot-train \
        --dataset.repo_id=yourname/so100_pick_place_pi05 \
        --policy.type=pi05 \
        --policy.pretrained_path=lerobot/pi05_base \
        --policy.empty_cameras=1 \
        --rename_map='{"observation.images.third_person":"observation.images.base_0_rgb","observation.images.wrist":"observation.images.left_wrist_0_rgb"}'
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Normal


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_ppo import TinyMLP


class FrozenObsNormalizer:
    """Checkpoint-backed observation normalizer used during PPO inference."""

    def __init__(self, mean: torch.Tensor, var: torch.Tensor, clip: float = 10.0):
        self.mean = mean
        self.var = var
        self.clip = clip

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip)


def _invoke_with_supported_kwargs(func, **kwargs):
    """Call a function while filtering kwargs unsupported by the current signature."""
    sig = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return func(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
    return func(**filtered)


class LeRobotEpisodeWriter:
    """Thin compatibility wrapper around LeRobotDataset creation and episode writing."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None,
        fps: int,
        features: dict[str, Any],
        robot_type: str,
        use_videos: bool,
        image_writer_threads: int,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is required for dataset export. Install the repo requirements "
                "in your Isaac Lab Python environment with `python -m pip install -r requirements.txt`."
            ) from exc

        create_kwargs = {
            "repo_id": repo_id,
            "fps": int(fps),
            "root": Path(root) if root is not None else None,
            "robot_type": robot_type,
            "features": features,
            "use_videos": use_videos,
            "image_writer_threads": image_writer_threads,
        }
        self.dataset = _invoke_with_supported_kwargs(LeRobotDataset.create, **create_kwargs)
        dataset_root = getattr(self.dataset, "root", None)
        if dataset_root is None and hasattr(self.dataset, "meta"):
            dataset_root = getattr(self.dataset.meta, "root", None)
        if dataset_root is None:
            raise RuntimeError("Could not determine the dataset root returned by LeRobotDataset.create().")
        self.root = Path(dataset_root)

        self._add_frame_sig = inspect.signature(self.dataset.add_frame)
        self._save_episode_sig = inspect.signature(self.dataset.save_episode)
        self._supports_add_task = "task" in self._add_frame_sig.parameters
        self._supports_save_task = "task" in self._save_episode_sig.parameters
        self._supports_save_tasks = "tasks" in self._save_episode_sig.parameters

    def write_episode(self, frames: list[dict[str, Any]], task: str) -> None:
        for frame in frames:
            if self._supports_add_task:
                try:
                    self.dataset.add_frame(frame, task=task)
                    continue
                except Exception:
                    # Some revisions expect a task payload rather than a raw string.
                    self.dataset.add_frame(frame, task={"task": task})
                    continue
            self.dataset.add_frame({**frame, "task": task})

        save_kwargs: dict[str, Any] = {}
        if self._supports_save_task:
            save_kwargs["task"] = task
        if self._supports_save_tasks:
            save_kwargs["tasks"] = [task]
        self.dataset.save_episode(**save_kwargs)

    def finalize(self) -> None:
        if hasattr(self.dataset, "finalize"):
            self.dataset.finalize()

    def push_to_hub(self) -> None:
        if hasattr(self.dataset, "push_to_hub"):
            self.dataset.push_to_hub()


@dataclass
class PendingEpisode:
    prompt: str
    cube_variant_idx: int
    cup_variant_idx: int
    frames: list[dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    control_steps: int = 0
    recorded_steps: int = 0
    ever_grasped: bool = False
    ever_lifted: bool = False
    ever_droppable: bool = False
    ever_success: bool = False

    def to_summary(self, cfg) -> dict[str, Any]:
        cube_dims = cfg.cube_size_variants[self.cube_variant_idx]
        cup_height = cfg.cup_height_variants[self.cup_variant_idx]
        return {
            "prompt": self.prompt,
            "reward": self.reward,
            "control_steps": self.control_steps,
            "recorded_steps": self.recorded_steps,
            "grasped": self.ever_grasped,
            "lifted": self.ever_lifted,
            "droppable": self.ever_droppable,
            "success": self.ever_success,
            "cube_variant_idx": self.cube_variant_idx,
            "cube_dims_cm": [round(200.0 * v, 2) for v in cube_dims],
            "cup_variant_idx": self.cup_variant_idx,
            "cup_height_cm": round(100.0 * cup_height, 2),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a PI0.5-compatible dataset from the PPO policy")
    parser.add_argument("--checkpoint", type=str, default="ppo_checkpoint_final.pt", help="PPO checkpoint path")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo id, e.g. user/so100_pick_place_pi05")
    parser.add_argument("--root", type=str, default="datasets", help="Local root folder for the LeRobot dataset")
    parser.add_argument("--episodes", type=int, default=200, help="Number of successful episodes to save")
    parser.add_argument("--max-attempts", type=int, default=0, help="Optional cap on total attempted episodes (0 = no cap)")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel Isaac Lab environments for collection")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum control steps per episode")
    parser.add_argument("--record-every", type=int, default=2, help="Save one frame every N control steps")
    parser.add_argument("--prompt", type=str, default="Pick up the cube and place it in the cup.", help="Default task prompt")
    parser.add_argument("--prompts-file", type=str, default="", help="Optional text file with one prompt per line")
    parser.add_argument("--stochastic", action="store_true", help="Sample PPO actions instead of using mean actions")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Lab headless")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--joint-noise", type=float, default=0.0, help="Optional reset joint noise during collection")
    parser.add_argument("--use-images", action="store_true", help="Store raw images instead of encoding videos")
    parser.add_argument("--image-writer-threads", type=int, default=4, help="LeRobot image writer threads when using videos")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the dataset after local finalization")
    parser.add_argument("--no-domain-randomization", action="store_true", help="Disable collection-time domain randomization")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = [args.prompt.strip()]
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        file_prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
        if not file_prompts:
            raise ValueError(f"No prompts found in {prompts_path}")
        prompts = file_prompts
    return prompts


def build_features(cfg, use_videos: bool) -> dict[str, Any]:
    image_dtype = "image" if not use_videos else "video"
    state_names = [f"{name}_pos" for name in cfg.joint_names] + [f"{name}_vel" for name in cfg.joint_names]
    return {
        "observation.images.third_person": {
            "dtype": image_dtype,
            "shape": (3, cfg.camera_height, cfg.camera_width),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": image_dtype,
            "shape": (3, cfg.camera_height, cfg.camera_width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": {"axes": state_names},
        },
        "action": {
            "dtype": "float32",
            "shape": (cfg.action_space,),
            "names": {"axes": list(cfg.joint_names)},
        },
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_prompt(prompts: list[str]) -> str:
    return random.choice(prompts)


def make_pending_episode(env, cfg, env_idx: int, prompts: list[str]) -> PendingEpisode:
    return PendingEpisode(
        prompt=choose_prompt(prompts),
        cube_variant_idx=int(env._active_cube_variant[env_idx].item()),
        cup_variant_idx=int(env._active_cup_variant[env_idx].item()),
    )


def compute_expert_joint_targets(env, actions: torch.Tensor) -> torch.Tensor:
    """Mirror the environment's action processing before stepping."""
    actions = torch.clamp(actions, -1.0, 1.0)
    delta = actions * env.cfg.action_scale
    current_pos = env.robot.data.joint_pos.clone()
    raw_targets = current_pos + delta

    for joint_idx, name in enumerate(env.cfg.joint_names):
        lower, upper = env.cfg.joint_limits[name]
        raw_targets[:, joint_idx] = torch.clamp(raw_targets[:, joint_idx], lower, upper)

    alpha = env.cfg.action_smooth_alpha
    smoothed = alpha * env._smoothed_joint_targets.clone() + (1.0 - alpha) * raw_targets

    gripper_open_pos = env.cfg.joint_limits["gripper"][1]
    force_open = env.grasp_detector.should_force_open
    smoothed[force_open, env._gripper_joint_idx] = gripper_open_pos
    return smoothed


def build_frame(images: dict[str, torch.Tensor], joint_pos: torch.Tensor, joint_vel: torch.Tensor, action_targets: torch.Tensor, env_idx: int) -> dict[str, Any]:
    third_person = images["third_person"][env_idx, ..., :3].detach().cpu().numpy().copy()
    wrist = images["wrist"][env_idx, ..., :3].detach().cpu().numpy().copy()
    state = torch.cat([joint_pos[env_idx], joint_vel[env_idx]], dim=0).detach().cpu().numpy().astype(np.float32)
    action = action_targets[env_idx].detach().cpu().numpy().astype(np.float32)
    return {
        "observation.images.third_person": third_person,
        "observation.images.wrist": wrist,
        "observation.state": state,
        "action": action,
    }


def summarize_and_write(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    if args.record_every < 1:
        raise ValueError("--record-every must be >= 1")

    prompts = load_prompts(args)
    seed_everything(args.seed)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    import carb
    carb.settings.get_settings().set("/rtx/dlss/enabled", False)
    carb.settings.get_settings().set("/rtx/post/dlss/execMode", 0)

    from lab.pick_place_env import PickPlaceEnv
    from lab.pick_place_env_cfg import PickPlaceEnvCfg

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = PickPlaceEnvCfg()
    cfg.scene.num_envs = args.n_envs
    cfg.enable_cameras = True
    cfg.initial_joint_noise = args.joint_noise
    cfg.domain_randomization_enable = not args.no_domain_randomization
    cfg.max_episode_steps = args.max_steps
    cfg.episode_length_s = args.max_steps * cfg.sim.dt * cfg.decimation

    env = PickPlaceEnv(cfg)
    policy_device = torch.device(env.device)

    policy = TinyMLP(obs_dim=cfg.observation_space, act_dim=cfg.action_space, hidden_dim=256)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(policy_device)
    policy.eval()

    obs_normalizer = FrozenObsNormalizer(
        mean=ckpt["obs_normalizer_mean"].to(policy_device),
        var=ckpt["obs_normalizer_var"].to(policy_device),
    )

    control_dt = cfg.sim.dt * cfg.decimation
    effective_hz = 1.0 / (control_dt * args.record_every)
    dataset_fps = max(1, int(round(effective_hz)))

    writer = LeRobotEpisodeWriter(
        repo_id=args.repo_id,
        root=args.root,
        fps=dataset_fps,
        features=build_features(cfg, use_videos=not args.use_images),
        robot_type="so100_isaaclab",
        use_videos=not args.use_images,
        image_writer_threads=args.image_writer_threads,
    )

    obs_dict, _ = env.reset()
    pending = [make_pending_episode(env, cfg, env_idx=i, prompts=prompts) for i in range(args.n_envs)]

    total_attempts = 0
    saved_episodes = 0
    failed_episodes = 0
    saved_frames = 0
    episode_manifest: list[dict[str, Any]] = []
    max_attempts = args.max_attempts if args.max_attempts > 0 else None

    print(
        f"[INFO] Collecting {args.episodes} successful episodes | "
        f"n_envs={args.n_envs} | record_every={args.record_every} | "
        f"dataset_fps≈{effective_hz:.2f}Hz (stored as {dataset_fps}Hz) | "
        f"domain_randomization={cfg.domain_randomization_enable}"
    )

    try:
        while saved_episodes < args.episodes and (max_attempts is None or total_attempts < max_attempts):
            joint_pos = env.robot.data.joint_pos.clone()
            joint_vel = env.robot.data.joint_vel.clone()

            policy_obs = obs_normalizer.normalize(obs_dict["policy"].to(policy_device))
            with torch.inference_mode():
                if args.stochastic:
                    action_mean, _ = policy.forward(policy_obs)
                    action_std = torch.exp(policy.policy_log_std)
                    actions = Normal(action_mean, action_std).sample()
                else:
                    actions, _ = policy.get_action(policy_obs, deterministic=True)

            expert_targets = compute_expert_joint_targets(env, actions)

            next_obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated | truncated

            images = env.update_cameras()

            for env_idx in range(args.n_envs):
                episode = pending[env_idx]
                if episode.control_steps % args.record_every == 0:
                    episode.frames.append(build_frame(images, joint_pos, joint_vel, expert_targets, env_idx))
                    episode.recorded_steps += 1
            milestones = info.get("milestone_flags", {})
            grasped = milestones.get("grasped", torch.zeros(args.n_envs, dtype=torch.bool, device=env.device))
            lifted = milestones.get("lifted", torch.zeros(args.n_envs, dtype=torch.bool, device=env.device))
            droppable = milestones.get("droppable", torch.zeros(args.n_envs, dtype=torch.bool, device=env.device))
            success = milestones.get("success", torch.zeros(args.n_envs, dtype=torch.bool, device=env.device))

            for env_idx in range(args.n_envs):
                episode = pending[env_idx]
                episode.reward += float(reward[env_idx].item())
                episode.control_steps += 1
                episode.ever_grasped |= bool(grasped[env_idx].item())
                episode.ever_lifted |= bool(lifted[env_idx].item())
                episode.ever_droppable |= bool(droppable[env_idx].item())
                episode.ever_success |= bool(success[env_idx].item())

                if not bool(done[env_idx].item()):
                    continue

                total_attempts += 1
                summary = episode.to_summary(cfg)
                summary["attempt_index"] = total_attempts

                if episode.ever_success and episode.frames and saved_episodes < args.episodes:
                    writer.write_episode(episode.frames, task=episode.prompt)
                    saved_episodes += 1
                    saved_frames += len(episode.frames)
                    summary["saved_episode_index"] = saved_episodes - 1
                    episode_manifest.append(summary)
                    print(
                        f"[SAVE {saved_episodes:04d}/{args.episodes}] "
                        f"attempt={total_attempts:04d} reward={episode.reward:8.1f} "
                        f"frames={len(episode.frames):03d} cube={episode.cube_variant_idx} cup={episode.cup_variant_idx}"
                    )
                else:
                    failed_episodes += 1
                    print(
                        f"[DROP {failed_episodes:04d}] "
                        f"attempt={total_attempts:04d} reward={episode.reward:8.1f} "
                        f"success={episode.ever_success} frames={len(episode.frames):03d}"
                    )

                pending[env_idx] = make_pending_episode(env, cfg, env_idx=env_idx, prompts=prompts)

            obs_dict = next_obs_dict

            if not args.headless:
                simulation_app.update()

        success_rate = (saved_episodes / total_attempts) if total_attempts else 0.0
        summary = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "repo_id": args.repo_id,
            "dataset_root": str(writer.root),
            "requested_successes": args.episodes,
            "saved_successes": saved_episodes,
            "failed_attempts": failed_episodes,
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "saved_frames": saved_frames,
            "n_envs": args.n_envs,
            "record_every": args.record_every,
            "control_dt_s": control_dt,
            "effective_record_hz": effective_hz,
            "stored_dataset_fps": dataset_fps,
            "deterministic_policy": not args.stochastic,
            "domain_randomization": cfg.domain_randomization_enable,
            "domain_randomization_config": {
                "light_intensity_range": list(cfg.light_intensity_range),
                "light_color_range": [list(cfg.light_color_range[0]), list(cfg.light_color_range[1])],
                "cube_color_range": [list(cfg.cube_color_range[0]), list(cfg.cube_color_range[1])],
                "cup_color_range": [list(cfg.cup_color_range[0]), list(cfg.cup_color_range[1])],
                "third_person_eye_jitter": list(cfg.camera_third_person_eye_jitter),
                "third_person_target_jitter": list(cfg.camera_third_person_target_jitter),
                "wrist_pos_jitter": list(cfg.camera_wrist_pos_jitter),
                "wrist_rot_jitter_deg": list(cfg.camera_wrist_rot_jitter_deg),
            },
            "joint_noise": args.joint_noise,
            "prompts": prompts,
            "episodes": episode_manifest,
        }
        writer.finalize()
        summary_path = writer.root / "collection_summary.json"
        summarize_and_write(summary_path, summary)
        if args.push_to_hub:
            writer.push_to_hub()

        print("=" * 72)
        print(
            f"[DONE] saved={saved_episodes} failed={failed_episodes} attempts={total_attempts} "
            f"success_rate={100.0 * success_rate:.1f}% frames={saved_frames}"
        )
        print(f"[DONE] dataset: {writer.root}")
        print(f"[DONE] summary: {summary_path}")
        if max_attempts is not None and total_attempts >= max_attempts and saved_episodes < args.episodes:
            print("[WARN] Stopped because --max-attempts was reached before hitting the requested success count.")
            return 1
        return 0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

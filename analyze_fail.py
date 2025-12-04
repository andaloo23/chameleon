
import argparse
import json
import os

import numpy as np


def analyze(path: str):
    if not os.path.exists(path):
        print(f"No data found at {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data['infos'])} steps from {path}.")

    for i, info in enumerate(data["infos"]):
        state = info.get("task_state", {})

        cube_pos = state.get("cube_pos")
        gripper_pos = state.get("gripper_pos")

        if i % 20 == 0 or i > 180:
            c_str = f"[{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]" if cube_pos else "N/A"
            g_str = f"[{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]" if gripper_pos else "N/A"

            dist = 0.0
            if cube_pos and gripper_pos:
                dist = np.linalg.norm(np.array(cube_pos) - np.array(gripper_pos))

            print(f"Step {i}: Dist={dist:.3f} | Cube={c_str} | Grip={g_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a saved trajectory JSON.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/rollouts/episode_0/trajectory.json",
        help="Path to trajectory.json produced by collect_data.py",
    )
    analyze(parser.parse_args().path)

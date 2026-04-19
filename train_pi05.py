import sys
sys.argv = [
    "lerobot_train.py",
    "--policy.path=lerobot/pi05_base",
    "--policy.repo_id=andaloo23/pi05_so100_pick_place",
    "--dataset.repo_id=andaloo23/so100_pick_place_pi05",
    "--policy.dtype=bfloat16",
    "--policy.gradient_checkpointing=true",
    "--batch_size=8",
    "--steps=5000",
    "--output_dir=outputs/pi05_so100_finetune",
    "--policy.empty_cameras=1",
    '--rename_map={"observation.images.third_person": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}',
]

from lerobot.scripts.lerobot_train import train
train()

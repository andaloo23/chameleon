import sys
import re

# ── Patch 1: handle missing 'names' field on image features in dataset metadata ──
import lerobot.datasets.utils as _dataset_utils
import types

_orig_dataset_to_policy_features = _dataset_utils.dataset_to_policy_features

def _patched_dataset_to_policy_features(features):
    for ft in features.values():
        if ft.get("dtype") in ("image", "video") and not ft.get("names"):
            ft["names"] = ["channels", "height", "width"]
    return _orig_dataset_to_policy_features(features)

_dataset_utils.dataset_to_policy_features = _patched_dataset_to_policy_features

# ── Patch 2: bypass transformers fork version check (standard transformers lacks it) ──
import lerobot.policies.pi05.modeling_pi05 as _pi05
import lerobot.policies.pi05.modeling_pi05 as _pi05_mod

_orig_PI05Pytorch_init = _pi05.PI05Pytorch.__init__

def _patched_PI05Pytorch_init(self, config, **kwargs):
    try:
        _orig_PI05Pytorch_init(self, config, **kwargs)
    except ValueError as e:
        if "incorrect transformer version" in str(e):
            pass  # transformers fork check unavailable, continuing
        else:
            raise

_pi05.PI05Pytorch.__init__ = _patched_PI05Pytorch_init

# ── Patch 3: prevent Windows path mangling of HuggingFace repo IDs ──
import lerobot.configs.train as _train_cfg
from pathlib import Path as _Path

_orig_train_post_init = _train_cfg.TrainPipelineConfig.__post_init__

def _patched_train_post_init(self):
    _orig_train_post_init(self)
    if self.policy and self.policy.pretrained_path:
        p = str(self.policy.pretrained_path).replace("\\", "/")
        self.policy.pretrained_path = p if re.match(r'^[\w.-]+/[\w.-]+$', p) else _Path(p)

_train_cfg.TrainPipelineConfig.__post_init__ = _patched_train_post_init

# ─────────────────────────────────────────────────────────────────────────────

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

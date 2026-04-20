import os
import site

def _patch_file(path, old, new):
    if not os.path.exists(path):
        return
    with open(path) as f:
        content = f.read()
    if old not in content:
        return  # already patched or changed
    with open(path, "w") as f:
        f.write(content.replace(old, new))

for _sp in site.getsitepackages():
    # Patch 1: handle missing 'names' field on image features in dataset metadata
    _patch_file(
        os.path.join(_sp, "lerobot", "datasets", "utils.py"),
        'names = ft["names"]',
        'names = ft.get("names") or ["channels", "height", "width"]',
    )
    # Patch 2: bypass transformers fork version check (standard transformers lacks it)
    _patch_file(
        os.path.join(_sp, "lerobot", "policies", "pi05", "modeling_pi05.py"),
        "except ImportError:\n            raise ValueError(msg) from None",
        "except ImportError:\n            pass  # transformers fork check unavailable",
    )
    # Patch 3: prevent Windows path mangling of HuggingFace repo IDs
    _patch_file(
        os.path.join(_sp, "lerobot", "configs", "train.py"),
        "self.policy.pretrained_path = Path(policy_path)",
        'import re as _re; self.policy.pretrained_path = policy_path if _re.match(r"^[\\w.-]+/[\\w.-]+$", policy_path) else Path(policy_path)',
    )
    # Patch 4: keep vision tower + multi_modal_projector in float32 (standard transformers
    # has mixed-dtype issues in SigLIP encoder; projector must match vision tower dtype)
    # cast output features to language model dtype in embed_image
    _patch_file(
        os.path.join(_sp, "lerobot", "policies", "pi05", "modeling_pi05.py"),
        '            "vision_tower.vision_model.embeddings.patch_embedding.weight",\n            "vision_tower.vision_model.embeddings.patch_embedding.bias",\n            "vision_tower.vision_model.embeddings.position_embedding.weight",',
        '            "vision_tower",  # keep entire vision tower in float32 — encoder has mixed dtype issues with standard transformers\n            "multi_modal_projector",  # receives float32 from vision tower, must match',
    )
    _patch_file(
        os.path.join(_sp, "lerobot", "policies", "pi05", "modeling_pi05.py"),
        "return self.paligemma.model.get_image_features(image)",
        "features = self.paligemma.model.get_image_features(image.float())\n        return features.to(dtype=next(self.paligemma.language_model.parameters()).dtype)",
    )
    # Patch 5: use strict=False so checkpoint loads even with key mismatches from
    # transformers version differences (input_layernorm.dense.weight vs input_layernorm.weight)
    _patch_file(
        os.path.join(_sp, "lerobot", "policies", "pi05", "modeling_pi05.py"),
        "missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)",
        "missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)",
    )

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

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import train
    train()

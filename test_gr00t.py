import sys
from importlib import import_module

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel

def register_gr00t(repo_id: str = "nvidia/GR00T-N1-2B", local_dir: str = "/home/ubuntu/models/gr00t-dyn"):
    # Download the dynamic modules that define the architecture
    repo_path = snapshot_download(
        repo_id="nvidia/GR00T-N1-2B",
        local_dir="/home/ubuntu/models/gr00t-dyn",
        repo_type="model",
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.py",            # pick up configuration_gr00t_n1.py, modeling_gr00t_n1.py, etc.
            "config.json",
            "*.safetensors",
        ],
    )
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    cfg_module = import_module("configuration_gr00t_n1")
    mdl_module = import_module("modeling_gr00t_n1")

    # Register the new architecture with Transformers
    AutoConfig.register("gr00t_n1", cfg_module.Gr00TN1Config)
    AutoModel.register(cfg_module.Gr00TN1Config, mdl_module.Gr00TN1Model)

register_gr00t()
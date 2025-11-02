import sys
from importlib import import_module
from pathlib import Path

from huggingface_hub import hf_hub_download, model_info
from transformers import AutoConfig, AutoModel


def register_gr00t(repo_id: str = "nvidia/GR00T-N1-2B") -> Path:
    info = model_info(repo_id)
    py_files = [f.rfilename for f in info.siblings if f.rfilename.endswith(".py")]
    if not py_files:
        raise RuntimeError("No Python files found in the repo; cannot register gr00t_n1.")

    local_dir = Path.home() / "models" / "gr00t-dyn"
    local_dir.mkdir(parents=True, exist_ok=True)

    local_paths = {}
    for filename in py_files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            repo_type="model",
        )
        local_paths[filename] = Path(local_path)

    for path in local_paths.values():
        directory = str(path.parent)
        if directory not in sys.path:
            sys.path.insert(0, directory)

    cfg_module_name = next((f.stem for f in local_paths.values() if "configuration" in f.stem), None)
    mdl_module_name = next((f.stem for f in local_paths.values() if "modeling" in f.stem), None)
    if not cfg_module_name or not mdl_module_name:
        raise RuntimeError("Could not identify configuration/modeling modules for GR00T-N1-2B.")

    cfg_module = import_module(cfg_module_name)
    mdl_module = import_module(mdl_module_name)

    AutoConfig.register("gr00t_n1", cfg_module.Gr00TN1Config)
    AutoModel.register(cfg_module.Gr00TN1Config, mdl_module.Gr00TN1Model)
    return local_paths[cfg_module_name + ".py"].parent


if __name__ == "__main__":
    repo_path = register_gr00t()
    model = AutoModel.from_pretrained(
        "nvidia/GR00T-N1-2B",
        torch_dtype="float16",
        trust_remote_code=True,
    )
    print("Model device:", next(model.parameters()).device)
import sys
from importlib import import_module
from pathlib import Path

from huggingface_hub import model_info, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from transformers import AutoConfig, AutoModel
from transformers import PreTrainedModel, PretrainedConfig


def _download_remote_code(repo_id: str, destination: Path) -> Path:
    """Download remote code from Hugging Face once and return the snapshot path."""
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=destination,
            local_dir_use_symlinks=False,
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            "Access to the specified Hugging Face repository is gated. "
            "Please accept the license and run `huggingface-cli login` with a token that has access."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Failed to download remote code for {repo_id}: {exc}") from exc

    return Path(snapshot_path)


def register_openvla(repo_id: str = "openvla/openvla-7b") -> Path:
    local_dir = Path.home() / "models" / "openvla"
    local_dir.mkdir(parents=True, exist_ok=True)

    info = model_info(repo_id)
    py_members = [s.rfilename for s in info.siblings if s.rfilename.endswith(".py")]
    if not py_members:
        raise RuntimeError(
            "Hugging Face did not list any Python files for the OpenVLA repo. "
            "If you just requested access, wait a few minutes and retry. "
            "Otherwise, verify that the repository still exposes remote code."
        )

    py_paths = {}
    snapshot_path = _download_remote_code(repo_id, local_dir)

    for rel_path in py_members:
        candidate = snapshot_path / rel_path
        if candidate.is_file():
            py_paths[candidate.stem.lower()] = candidate

    py_files = list(py_paths.values())
    if not py_files:
        raise RuntimeError(
            "No Python files were downloaded for OpenVLA. "
            "Verify your Hugging Face credentials or the repository permissions."
        )

    for path in py_files:
        directory = str(path.parent.resolve())
        if directory not in sys.path:
            sys.path.insert(0, directory)

    cfg_file = next((p for p in py_files if "configuration" in p.stem.lower()), None)
    mdl_file = next((p for p in py_files if "modeling" in p.stem.lower()), None)
    if cfg_file is None or mdl_file is None:
        raise RuntimeError("Could not identify configuration/modeling modules for OpenVLA.")

    cfg_module = import_module(cfg_file.stem)
    mdl_module = import_module(mdl_file.stem)

    cfg_candidates = []
    for name in dir(cfg_module):
        attr = getattr(cfg_module, name)
        if isinstance(attr, type) and issubclass(attr, PretrainedConfig) and attr is not PretrainedConfig:
            cfg_candidates.append(attr)

    mdl_candidates = []
    for name in dir(mdl_module):
        attr = getattr(mdl_module, name)
        if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr is not PreTrainedModel:
            mdl_candidates.append(attr)

    if not cfg_candidates or not mdl_candidates:
        raise RuntimeError("Could not locate config/model classes in the downloaded OpenVLA code.")

    cfg_cls = cfg_candidates[0]
    mdl_cls = mdl_candidates[0]

    AutoConfig.register("openvla", cfg_cls)
    AutoModel.register(cfg_cls, mdl_cls)

    return cfg_file.parent


if __name__ == "__main__":
    repo_path = register_openvla()
    model = AutoModel.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype="float16",
        trust_remote_code=True,
    )
    print("Model device:", next(model.parameters()).device)

import sys
from types import ModuleType
from importlib import import_module
from pathlib import Path
import torch

from huggingface_hub import model_info, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from typing import Tuple, Type

from transformers import AutoConfig, AutoModel
from transformers import PreTrainedModel


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


def register_openvla(repo_id: str = "openvla/openvla-7b") -> Tuple[Path, Type[PreTrainedModel]]:
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

    snapshot_path = _download_remote_code(repo_id, local_dir)

    if str(snapshot_path) not in sys.path:
        sys.path.insert(0, str(snapshot_path))

    has_package_root = any(Path(p).parts and Path(p).parts[0] == "openvla" for p in py_members)
    if not has_package_root and "openvla" not in sys.modules:
        pkg = ModuleType("openvla")
        pkg.__path__ = [str(snapshot_path)]
        pkg.__file__ = str(snapshot_path / "__init__.py")
        sys.modules["openvla"] = pkg

    module_names = []
    for rel_path in py_members:
        candidate = snapshot_path / rel_path
        if not candidate.is_file():
            continue
        module_parts = Path(rel_path).with_suffix("").parts
        if any(not part.isidentifier() for part in module_parts):
            continue
        if has_package_root:
            if module_parts[0] != "openvla":
                continue
            module_name = ".".join(module_parts)
        else:
            module_name = ".".join(("openvla",) + module_parts)
        module_names.append(module_name)

    if not module_names:
        raise RuntimeError(
            "No Python files were downloaded for OpenVLA. "
            "Verify your Hugging Face credentials or the repository permissions."
        )

    cfg_module_name = next(
        (name for name in module_names if "configuration" in name.split(".")[-1].lower()),
        None,
    )
    mdl_module_name = next(
        (name for name in module_names if "modeling" in name.split(".")[-1].lower()),
        None,
    )
    if cfg_module_name is None or mdl_module_name is None:
        raise RuntimeError("Could not identify configuration/modeling modules for OpenVLA.")

    cfg = AutoConfig.from_pretrained(
        repo_id,
        revision=info.sha,
        trust_remote_code=True,
    )

    cfg_cls = cfg.__class__
    remote_prefix = cfg_cls.__module__.rsplit(".", 1)[0]
    mdl_suffix = mdl_module_name.split(".", 1)[1] if "." in mdl_module_name else mdl_module_name

    try:
        remote_mdl_module = import_module(f"{remote_prefix}.{mdl_suffix}")
    except ModuleNotFoundError:
        remote_mdl_module = import_module(mdl_module_name)

    mdl_module = remote_mdl_module
    mdl_candidates = []
    for name in dir(mdl_module):
        attr = getattr(mdl_module, name)
        if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr is not PreTrainedModel:
            mdl_candidates.append(attr)

    if not mdl_candidates:
        raise RuntimeError("Could not locate model classes in the downloaded OpenVLA code.")

    mdl_cls = mdl_candidates[0]

    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False  # Ensure base class exposes the attribute expected by recent Transformers.
    if not hasattr(mdl_cls, "_supports_sdpa"):
        mdl_cls._supports_sdpa = False  # Some downstream classes also get checked directly.

    try:
        AutoConfig.register("openvla", cfg_cls)
    except ValueError:
        pass
    try:
        AutoConfig.register(cfg.model_type, cfg_cls)
    except ValueError:
        pass
    try:
        AutoModel.register(cfg_cls, mdl_cls)
    except ValueError:
        pass

    cfg_module_path = (snapshot_path / cfg_module_name.replace(".", "/")).with_suffix(".py")
    return cfg_module_path.parent, mdl_cls


if __name__ == "__main__":
    repo_path, mdl_cls = register_openvla()
    device_kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    try:
        model = AutoModel.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **device_kwargs,
        )
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        model = mdl_cls.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **device_kwargs,
        )
    print("Model device:", next(model.parameters()).device)

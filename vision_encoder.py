import os
from typing import Optional, Tuple, List, Union

import torch
from torch import nn
from transformers import AutoModel
from peft import PeftModel
from PIL import Image
import torchvision.transforms as T
import numpy as np

DEFAULT_IMAGE_SIZE = (224, 224)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class VisionEncoder(nn.Module):
    """Wrapper that loads GR00T base weights and an optional LoRA adapter on GPU."""

    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.image_size = image_size
        self.dtype = dtype
        self.device = device or torch.device("cuda")

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

        base_model = AutoModel.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        base_model.to(self.device)
        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False

        if self.lora_adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_adapter_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
        else:
            self.model = base_model
        self.model.to(self.device)

        if freeze_base:
            for name, param in self.model.named_parameters():
                if getattr(param, "requires_grad", False) and self.lora_adapter_path is not None:
                    if "lora" not in name:
                        param.requires_grad = False
                elif getattr(param, "requires_grad", False) and self.lora_adapter_path is None:
                    param.requires_grad = False

    def _ensure_pil(self, image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(image)
        raise TypeError("Unsupported image format; expected PIL.Image or numpy array.")

    def preprocess(self, images: List[Union[Image.Image, np.ndarray]]):
        tensors = []
        for img in images:
            pil_image = self._ensure_pil(img)
            tensors.append(self.transform(pil_image))
        pixel_values = torch.stack(tensors, dim=0)
        pixel_values = pixel_values.to(self.device, dtype=self.dtype, non_blocking=True)
        return pixel_values

    @torch.no_grad()
    def encode(self, images):
        self.model.eval()
        pixel_values = images if torch.is_tensor(images) else self.preprocess(images)
        outputs = self.model(pixel_values=pixel_values)
        embedding = outputs.last_hidden_state[:, 0]
        return embedding

    def forward(self, images):
        return self.encode(images)


def load_vision_encoder(
    base_model_path: str,
    lora_adapter_path: Optional[str] = None,
    device: Optional[str] = "cuda",
    dtype: torch.dtype = torch.float16,
) -> VisionEncoder:
    target_device = torch.device(device)
    encoder = VisionEncoder(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        device=target_device,
        dtype=dtype,
    )
    return encoder
"""Construction du réseau, prétraitement des images et prédiction.

Utilisé partout : entraînement (Modal, GPU), évaluation, démo Gradio et API (CPU).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .labels import BY_ID, CLASS_IDS, crop_indices

IMG_SIZE = 224
RESIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# En dessous de cette probabilité, la démo prévient que le modèle hésite.
LOW_CONFIDENCE = 0.5


def build_model(arch: str, num_classes: int = len(CLASS_IDS), pretrained: bool = False) -> torch.nn.Module:
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def eval_transform(img_size: int = IMG_SIZE, resize: int = RESIZE):
    from torchvision.transforms import v2 as T

    return T.Compose([
        T.ToImage(),
        T.Resize(resize, antialias=True),
        T.CenterCrop(img_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


def load_image(path_or_img) -> Image.Image:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = ImageOps.exif_transpose(img)  # photos de téléphone : respecter l'orientation
    return img.convert("RGB")


@dataclass
class Prediction:
    class_id: str
    prob: float

    @property
    def info(self):
        return BY_ID[self.class_id]


class Predictor:
    """Charge le modèle exporté (poids safetensors + meta.json) et prédit sur CPU."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        if self.meta["classes"] != CLASS_IDS:
            raise ValueError("Les classes du modèle ne correspondent pas à leaf_core.labels")
        self.model = build_model(self.meta["arch"])
        self.model.load_state_dict(load_file(model_dir / "model.safetensors"))
        self.model.eval().to(device)
        self.device = device
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.transform = eval_transform(self.meta.get("img_size", IMG_SIZE), self.meta.get("resize", RESIZE))

    @torch.inference_mode()
    def probabilities(self, image, crop: str | None = None) -> np.ndarray:
        """Probabilités (calibrées) sur les 22 classes.

        Si `crop` est donné, on ne garde que les classes de cette culture
        et on renormalise : c'est le mode « je sais quelle culture j'ai photographiée ».
        """
        x = self.transform(load_image(image)).unsqueeze(0).to(self.device)
        logits = self.model(x)[0].float() / self.temperature
        if crop:
            mask = torch.full_like(logits, float("-inf"))
            idx = crop_indices(crop)
            mask[idx] = 0
            logits = logits + mask
        return torch.softmax(logits, dim=0).cpu().numpy()

    def predict(self, image, crop: str | None = None, k: int = 3) -> list[Prediction]:
        p = self.probabilities(image, crop)
        top = np.argsort(-p)[:k]
        return [Prediction(CLASS_IDS[i], float(p[i])) for i in top]

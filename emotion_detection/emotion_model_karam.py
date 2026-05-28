"""Emotion detection CNN model definition (Karam).

Architecture: EfficientNet-B0 pretrained backbone + 7-class emotion head.
Trained on FER-2013 (Kaggle) with 7 emotion classes:
    0: Angry   1: Disgust  2: Fear  3: Happy
    4: Neutral  5: Sad     6: Surprise

Input: 48x48 grayscale image from FER-2013,
       converted to 3-channel 224x224 for EfficientNet compatibility.
"""
# Author: Karam

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
NUM_EMOTIONS   = len(EMOTION_LABELS)
IMAGE_SIZE     = (224, 224)
TORCH_CACHE_DIR = Path("models/torch_cache")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class EmotionModel(nn.Module):
    """EfficientNet-B0 backbone with a 7-class emotion classification head.

    EfficientNet-B0 is chosen (vs the ResNet34 used in face recognition)
    to produce independently trainable weights and enable a fair comparison
    of model architectures in the report.
    """

    def __init__(self, num_classes: int = NUM_EMOTIONS, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(TORCH_CACHE_DIR))

        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.classifier(features)


def load_emotion_model(
    model_path: str | Path | None = None,
    num_classes: int = NUM_EMOTIONS,
    device: str | torch.device | None = None,
) -> EmotionModel:
    """Load an EmotionModel, optionally restoring saved checkpoint weights.

    Args:
        model_path: Path to .pth checkpoint saved by train_emotion_karam.py.
        num_classes: Number of emotion classes (default 7 for FER-2013).
        device:     Target device. Defaults to CUDA if available.

    Returns:
        EmotionModel in eval mode on the target device.
    """
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if model_path:
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Emotion model not found: {checkpoint_path}")

        checkpoint  = torch.load(checkpoint_path, map_location=selected_device)
        num_classes = checkpoint.get("num_classes", num_classes)
        model       = EmotionModel(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = EmotionModel(num_classes=num_classes, pretrained=True)

    model.to(selected_device)
    model.eval()
    return model


def preprocess_emotion_image(face_image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Prepare a 224×224 RGB face crop for the emotion model.

    Args:
        face_image: RGB numpy array with shape (224, 224, 3).
        device:     Target torch device.

    Returns:
        Normalised float32 tensor with shape (1, 3, 224, 224).
    """
    if face_image is None or not isinstance(face_image, np.ndarray):
        raise ValueError("face_image must be a numpy.ndarray.")
    if face_image.shape != (224, 224, 3):
        raise ValueError(f"Expected shape (224, 224, 3), got {face_image.shape}.")

    image  = face_image.astype("float32") / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    mean   = IMAGENET_MEAN.to(device)
    std    = IMAGENET_STD.to(device)
    return ((tensor - mean) / std).to(device)

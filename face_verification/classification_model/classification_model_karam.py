"""Classification-based face embedding model for Karam's supervised learning module.

Architecture: ResNet34 pretrained backbone + custom classification head.
  - During training : full model (backbone → classifier) with CrossEntropyLoss.
  - During inference: backbone only, L2-normalised → 512-dim face embedding.

The embedding is drop-in compatible with the metric-learning gallery format,
allowing direct performance comparison between the two approaches.
"""
# Author: Karam

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models


IMAGE_SIZE = (224, 224)
EMBEDDING_DIM = 512
TORCH_CACHE_DIR = Path("models/torch_cache")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FaceClassificationModel(nn.Module):
    """ResNet34 backbone with a two-layer softmax classification head.

    Call ``forward()`` during training to get class logits.
    Call ``get_embedding()`` at inference time to extract L2-normalised embeddings.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(TORCH_CACHE_DIR))
        backbone = models.resnet34(weights=weights)

        # Strip original FC; backbone now outputs raw 512-dim feature vectors
        self.embedding_dim: int = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Trainable classification head (dropped after training)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits — used only during training."""
        features = self.backbone(images)
        return self.classifier(features)

    def get_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised face embeddings — used at inference / gallery build."""
        features = self.backbone(images)
        return nn.functional.normalize(features, p=2, dim=1)


def load_classification_model(
    model_path: str | Path | None = None,
    num_classes: int = 1000,
    device: str | torch.device | None = None,
    pretrained: bool = True,
) -> FaceClassificationModel:
    """Load a FaceClassificationModel, optionally restoring saved checkpoint weights.

    Args:
        model_path: Path to a .pth checkpoint saved by train_classification_karam.py.
                    If None, returns a freshly initialised (ImageNet-pretrained) model.
        num_classes: Number of identity classes (only used when model_path is None).
        device:      Target device string or torch.device. Defaults to CUDA if available.
        pretrained:  Load ImageNet weights when model_path is None.

    Returns:
        FaceClassificationModel in eval mode on the target device.
    """
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if model_path:
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Classification model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=selected_device)
        num_classes = checkpoint.get("num_classes", num_classes)
        model = FaceClassificationModel(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = FaceClassificationModel(num_classes=num_classes, pretrained=pretrained)

    model.to(selected_device)
    model.eval()
    return model


def preprocess_face_image(face_image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a 224×224 RGB face array into a normalised torch batch tensor.

    Identical preprocessing to the metric-learning module so embeddings are
    comparable across both approaches.

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


@torch.no_grad()
def generate_embedding(
    model: FaceClassificationModel,
    face_image: np.ndarray,
) -> np.ndarray:
    """Generate one L2-normalised embedding vector for a standardised face image.

    Args:
        model:      Loaded FaceClassificationModel in eval mode.
        face_image: RGB numpy array with shape (224, 224, 3).

    Returns:
        Float32 numpy array with shape (512,).
    """
    device = next(model.parameters()).device
    batch  = preprocess_face_image(face_image, device=device)
    embedding = model.get_embedding(batch).squeeze(0).detach().cpu().numpy()
    return embedding.astype("float32")

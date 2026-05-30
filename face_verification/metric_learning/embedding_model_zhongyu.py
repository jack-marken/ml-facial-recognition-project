"""Embedding model utilities for Zhongyu's metric-learning recognition module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models


IMAGE_SIZE = (224, 224)
DEFAULT_ARCHITECTURE = "resnet34"
SUPPORTED_ARCHITECTURES = ("efficientnet_b0", "resnet34")
TORCH_CACHE_DIR = Path("models/torch_cache")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FaceEmbeddingModel(nn.Module):
    """Backbone wrapper that returns L2-normalized face embeddings."""

    def __init__(self, architecture: str = DEFAULT_ARCHITECTURE, pretrained: bool = True):
        super().__init__()
        self.architecture = architecture
        self.feature_extractor, self.embedding_dim = _build_feature_extractor(
            architecture=architecture,
            pretrained=pretrained,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.feature_extractor(images)
        embeddings = torch.flatten(embeddings, start_dim=1)
        return nn.functional.normalize(embeddings, p=2, dim=1)


def load_embedding_model(
    architecture: str = DEFAULT_ARCHITECTURE,
    model_path: str | Path | None = None,
    device: str | torch.device | None = None,
    pretrained: bool = True,
) -> FaceEmbeddingModel:
    """Load an embedding model, optionally with trained checkpoint weights."""
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FaceEmbeddingModel(architecture=architecture, pretrained=pretrained)

    if model_path:
        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=selected_device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model.to(selected_device)
    model.eval()
    return model


def preprocess_face_image(face_image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a 224x224 RGB face image into a normalized torch batch."""
    if face_image is None or not isinstance(face_image, np.ndarray):
        raise ValueError("face_image must be a numpy.ndarray.")
    if face_image.shape != (224, 224, 3):
        raise ValueError(f"face_image must have shape (224, 224, 3), got {face_image.shape}.")

    image = face_image.astype("float32") / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    tensor = (tensor - mean) / std
    return tensor.to(device)


@torch.no_grad()
def generate_embedding(model: FaceEmbeddingModel, face_image: np.ndarray) -> np.ndarray:
    """Generate one normalized embedding vector for a standardized face image."""
    device = next(model.parameters()).device
    batch = preprocess_face_image(face_image, device=device)
    embedding = model(batch).squeeze(0).detach().cpu().numpy()
    return embedding.astype("float32")


def _build_feature_extractor(
    architecture: str,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    normalized_architecture = architecture.lower()
    if pretrained:
        TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(TORCH_CACHE_DIR))

    if normalized_architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        embedding_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, embedding_dim

    if normalized_architecture == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        embedding_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, embedding_dim

    raise ValueError(
        f"Unsupported architecture '{architecture}'. "
        f"Use one of: {', '.join(SUPPORTED_ARCHITECTURES)}."
    )

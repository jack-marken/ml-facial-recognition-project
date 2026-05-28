import torch
from torch import nn


SUPPORTED_MODELS = ("resnet18", "mobilenetv2")


class SiameseEmbeddingModel(nn.Module):
    """Backbone + projection head returning L2-normalized embeddings."""

    def __init__(self, feature_extractor: nn.Module, projection_head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projection_head = projection_head

    def forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        features = torch.flatten(features, start_dim=1)
        embeddings = self.projection_head(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)

    def forward(self, first_images: torch.Tensor, second_images: torch.Tensor):
        return self.forward_once(first_images), self.forward_once(second_images)


def pairwise_distance(first_embeddings: torch.Tensor, second_embeddings: torch.Tensor):
    return nn.functional.pairwise_distance(first_embeddings, second_embeddings, p=2)


def build_siamese_model(model_name: str, pretrained_backbone: bool = False):
    model_name = model_name.lower()

    if model_name == "resnet18":
        from face_verification.metric_learning.train_siamese_resnet18_kaixiang import (
            build_resnet18_siamese_model,
        )

        return build_resnet18_siamese_model(pretrained_backbone)

    if model_name == "mobilenetv2":
        from face_verification.metric_learning.train_siamese_mobilenetv2_kaixiang import (
            build_mobilenetv2_siamese_model,
        )

        return build_mobilenetv2_siamese_model(pretrained_backbone)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}")


def load_siamese_checkpoint(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_siamese_model(model_name, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device

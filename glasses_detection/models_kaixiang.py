from pathlib import Path

import torch
from torch import nn
from torchvision import models


SUPPORTED_MODELS = ("mobilenetv2", "efficientnetb0")


class GlassesClassifier(nn.Module):
    def __init__(self, backbone, classifier_getter):
        super().__init__()
        self.backbone = backbone
        self.classifier_getter = classifier_getter

    def forward(self, images):
        return self.backbone(images).squeeze(1)


def build_glasses_model(model_name, pretrained_backbone=True, dropout=0.3):
    normalized_name = model_name.lower()
    if normalized_name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = models.mobilenet_v2(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        return GlassesClassifier(backbone, "classifier")

    if normalized_name == "efficientnetb0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        return GlassesClassifier(backbone, "classifier")

    raise ValueError(f"Unsupported glasses model '{model_name}'. Choose one of: {SUPPORTED_MODELS}")


def freeze_backbone(model):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.backbone.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_feature_blocks(model, model_name, unfreeze_blocks):
    freeze_backbone(model)
    feature_blocks = getattr(model.backbone, "features", None)
    if feature_blocks is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    safe_count = max(1, int(unfreeze_blocks))
    for block in feature_blocks[-safe_count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def save_glasses_checkpoint(path, model, model_name, epoch, metrics, args_dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "args": args_dict,
        },
        path,
    )


def load_glasses_checkpoint(checkpoint_path, device=None):
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_glasses_model(model_name, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()
    return model, checkpoint, selected_device


import argparse

from torch import nn
from torchvision import models

from face_verification.metric_learning.siamese_models_kaixiang import (
    SiameseEmbeddingModel,
)
from face_verification.metric_learning.siamese_training_kaixiang import (
    add_common_training_args,
    run_siamese_training,
)


RESNET18_DEFAULTS = {
    "batch_size": 16,
    "workers": 0,
    "head_epochs": 5,
    "finetune_epochs": 15,
    "unfreeze_blocks": 2,
    "head_lr": 1e-3,
    "finetune_lr": 1e-4,
    "weight_decay": 1e-4,
    "margin": 1.0,
    "pairs_per_epoch": 3000,
    "max_positive_pairs_per_identity": 20,
    "max_negative_pairs": 1000,
    "early_stopping_patience": 5,
}


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def build_resnet18_siamese_model(pretrained_backbone: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    backbone = models.resnet18(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()

    projection_head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 128),
    )
    return SiameseEmbeddingModel(backbone, projection_head)


def freeze_resnet18_backbone(model: SiameseEmbeddingModel) -> None:
    _set_trainable(model.feature_extractor, False)
    _set_trainable(model.projection_head, True)


def unfreeze_resnet18_last_blocks(model: SiameseEmbeddingModel, num_blocks: int) -> None:
    if num_blocks <= 0:
        return

    candidate_blocks = [
        model.feature_extractor.layer4,
        model.feature_extractor.layer3,
        model.feature_extractor.layer2,
        model.feature_extractor.layer1,
    ]
    for block in candidate_blocks[:num_blocks]:
        _set_trainable(block, True)
    _set_trainable(model.projection_head, True)


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese + ResNet18 for metric-learning recognition."
    )
    add_common_training_args(parser, RESNET18_DEFAULTS)
    args = parser.parse_args()
    run_siamese_training(
        "resnet18",
        args,
        build_model=build_resnet18_siamese_model,
        freeze_backbone=freeze_resnet18_backbone,
        unfreeze_backbone=unfreeze_resnet18_last_blocks,
    )


if __name__ == "__main__":
    main()

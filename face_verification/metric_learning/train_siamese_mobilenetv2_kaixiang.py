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


MOBILENETV2_DEFAULTS = {
    "batch_size": 16,
    "workers": 0,
    "head_epochs": 5,
    "finetune_epochs": 15,
    "unfreeze_blocks": 6,
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


def build_mobilenetv2_siamese_model(pretrained_backbone: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
    backbone = models.mobilenet_v2(weights=weights)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    projection_head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU6(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(256, 128),
    )
    return SiameseEmbeddingModel(backbone, projection_head)


def freeze_mobilenetv2_backbone(model: SiameseEmbeddingModel) -> None:
    _set_trainable(model.feature_extractor.features, False)
    _set_trainable(model.projection_head, True)


def unfreeze_mobilenetv2_last_blocks(model: SiameseEmbeddingModel, num_blocks: int) -> None:
    if num_blocks <= 0:
        return

    feature_blocks = list(model.feature_extractor.features.children())
    for block in feature_blocks[-num_blocks:]:
        _set_trainable(block, True)
    _set_trainable(model.projection_head, True)


def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese + MobileNetV2 for metric-learning recognition."
    )
    add_common_training_args(parser, MOBILENETV2_DEFAULTS)
    args = parser.parse_args()
    run_siamese_training(
        "mobilenetv2",
        args,
        build_model=build_mobilenetv2_siamese_model,
        freeze_backbone=freeze_mobilenetv2_backbone,
        unfreeze_backbone=unfreeze_mobilenetv2_last_blocks,
    )


if __name__ == "__main__":
    main()

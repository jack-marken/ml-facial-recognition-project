import argparse

from torch import nn
from torchvision import models

from anti_spoofing.liveness_training_kaixiang import (
    add_common_training_args,
    run_liveness_training,
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
    "early_stopping_patience": 5,
}


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def build_mobilenetv2_liveness_model(pretrained_backbone: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, 1),
    )
    return model


def freeze_mobilenetv2_backbone(model: nn.Module) -> None:
    _set_trainable(model.features, False)
    _set_trainable(model.classifier, True)


def unfreeze_mobilenetv2_last_blocks(model: nn.Module, num_blocks: int) -> None:
    if num_blocks <= 0:
        return

    blocks = list(model.features.children())
    for block in blocks[-num_blocks:]:
        _set_trainable(block, True)
    _set_trainable(model.classifier, True)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 + binary head for liveness detection."
    )
    add_common_training_args(parser, MOBILENETV2_DEFAULTS)
    args = parser.parse_args()
    run_liveness_training(
        "mobilenetv2",
        args,
        build_model=build_mobilenetv2_liveness_model,
        freeze_backbone=freeze_mobilenetv2_backbone,
        unfreeze_last_blocks=unfreeze_mobilenetv2_last_blocks,
    )


if __name__ == "__main__":
    main()

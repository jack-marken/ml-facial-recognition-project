import argparse

from torch import nn
from torchvision import models

from anti_spoofing.liveness_training_kaixiang import (
    add_common_training_args,
    run_liveness_training,
)


EFFICIENTNETB0_DEFAULTS = {
    "batch_size": 8,
    "workers": 0,
    "head_epochs": 5,
    "finetune_epochs": 15,
    "unfreeze_blocks": 4,
    "head_lr": 1e-3,
    "finetune_lr": 5e-5,
    "weight_decay": 1e-4,
    "early_stopping_patience": 5,
}


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def build_efficientnetb0_liveness_model(pretrained_backbone: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(256, 1),
    )
    return model


def freeze_efficientnetb0_backbone(model: nn.Module) -> None:
    _set_trainable(model.features, False)
    _set_trainable(model.classifier, True)


def unfreeze_efficientnetb0_last_blocks(model: nn.Module, num_blocks: int) -> None:
    if num_blocks <= 0:
        return

    blocks = list(model.features.children())
    for block in blocks[-num_blocks:]:
        _set_trainable(block, True)
    _set_trainable(model.classifier, True)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNetB0 + binary head for liveness detection."
    )
    add_common_training_args(parser, EFFICIENTNETB0_DEFAULTS)
    args = parser.parse_args()
    run_liveness_training(
        "efficientnetb0",
        args,
        build_model=build_efficientnetb0_liveness_model,
        freeze_backbone=freeze_efficientnetb0_backbone,
        unfreeze_last_blocks=unfreeze_efficientnetb0_last_blocks,
    )


if __name__ == "__main__":
    main()

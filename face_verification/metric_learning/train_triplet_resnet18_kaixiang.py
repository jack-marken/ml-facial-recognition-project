import argparse

from face_verification.metric_learning.train_siamese_resnet18_kaixiang import (
    build_resnet18_siamese_model,
    freeze_resnet18_backbone,
    unfreeze_resnet18_last_blocks,
)
from face_verification.metric_learning.triplet_training_kaixiang import (
    add_triplet_args,
    run_triplet_training,
)


TRIPLET_RESNET18_DEFAULTS = {
    "head_epochs": 3,
    "finetune_epochs": 12,
    "unfreeze_blocks": 2,
    "identities_per_batch": 8,
    "samples_per_identity": 4,
    "batches_per_epoch": 120,
    "eval_batch_size": 32,
    "head_lr": 1e-3,
    "finetune_lr": 5e-5,
    "weight_decay": 1e-4,
    "margin": 0.3,
    "max_positive_pairs_per_identity": 20,
    "max_negative_pairs": 1000,
    "early_stopping_patience": 5,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train Batch-Hard Triplet + ResNet18 for metric-learning recognition."
    )
    add_triplet_args(parser, TRIPLET_RESNET18_DEFAULTS)
    args = parser.parse_args()
    run_triplet_training(
        "resnet18",
        args,
        build_model=build_resnet18_siamese_model,
        freeze_backbone=freeze_resnet18_backbone,
        unfreeze_backbone=unfreeze_resnet18_last_blocks,
    )


if __name__ == "__main__":
    main()

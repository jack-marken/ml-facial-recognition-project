import argparse

from glasses_detection.training_kaixiang import add_training_args, run_glasses_training


MOBILENETV2_DEFAULTS = {
    "batch_size": 32,
    "head_epochs": 3,
    "finetune_epochs": 8,
    "unfreeze_blocks": 6,
    "head_lr": 1e-3,
    "finetune_lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "early_stopping_patience": 4,
    "max_train_per_class": 6000,
    "max_val_per_class": 1500,
    "progress_every": 100,
}


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 + binary head for Kaixiang glasses detection."
    )
    add_training_args(parser, MOBILENETV2_DEFAULTS)
    args = parser.parse_args()
    run_glasses_training("mobilenetv2", args)


if __name__ == "__main__":
    main()

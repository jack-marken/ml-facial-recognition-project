"""Convenience wrapper for training the ResNet50V2 liveness model."""

try:
    from .train_liveness_zhongyu import main
except ImportError:
    from train_liveness_zhongyu import main


if __name__ == "__main__":
    main(default_architecture="resnet50v2")

"""Convenience wrapper for training the DenseNet121 liveness model."""

try:
    from .train_liveness_zhongyu import main
except ImportError:
    from train_liveness_zhongyu import main


if __name__ == "__main__":
    main(default_architecture="densenet121")

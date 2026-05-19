"""Convenience wrapper for training the ResNet50V2 liveness model."""

from __future__ import annotations

import sys

try:
    from .train_liveness_zhongyu import main
except ImportError:
    from train_liveness_zhongyu import main


if __name__ == "__main__":
    sys.argv.extend(["--architecture", "resnet50v2"])
    main()

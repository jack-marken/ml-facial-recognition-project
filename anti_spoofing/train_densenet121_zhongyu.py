"""Convenience wrapper for training the DenseNet121 liveness model."""

from __future__ import annotations

import sys

try:
    from .train_liveness_zhongyu import main
except ImportError:
    from train_liveness_zhongyu import main


if __name__ == "__main__":
    sys.argv.extend(["--architecture", "densenet121"])
    main()

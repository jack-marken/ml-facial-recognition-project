"""Anti-spoofing / liveness detection module."""

from .liveness_zhongyu import predict_liveness

__all__ = ["predict_liveness"]

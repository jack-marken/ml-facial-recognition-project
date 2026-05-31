"""Temporal liveness decision helper for Zhongyu's innovation feature.

The base liveness model predicts one cropped face at a time. This module adds a
small temporal decision layer on top of those frame-level predictions so the
system can make more stable webcam decisions across consecutive frames.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from anti_spoofing.liveness_zhongyu import (
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    predict_liveness_probability,
)

KAIXIANG_EFFICIENTNETB0_MODEL_PATH = Path(
    "models/liveness_efficientnetb0_kaixiang_final1_best.pth"
)
DEFAULT_TEMPORAL_MODEL_PATH = (
    KAIXIANG_EFFICIENTNETB0_MODEL_PATH
    if KAIXIANG_EFFICIENTNETB0_MODEL_PATH.exists()
    else DEFAULT_MODEL_PATH
)
DEFAULT_TEMPORAL_THRESHOLD = (
    0.5 if DEFAULT_TEMPORAL_MODEL_PATH.suffix.lower() == ".pth" else DEFAULT_THRESHOLD
)

_cached_torch_model = None
_cached_torch_model_path: Path | None = None
_cached_torch_transform = None
_cached_torch_device = None


class TemporalLivenessDetector:
    """Convert frame-level liveness probabilities into a stable temporal result."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TEMPORAL_MODEL_PATH,
        real_threshold: float = DEFAULT_TEMPORAL_THRESHOLD,
        spoof_threshold: float | None = None,
        window_size: int = 15,
        min_frames: int = 5,
        required_confirmations: int = 3,
        max_stable_std: float = 0.18,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if min_frames < 1:
            raise ValueError("min_frames must be at least 1.")
        if min_frames > window_size:
            raise ValueError("min_frames cannot be larger than window_size.")
        if required_confirmations < 1:
            raise ValueError("required_confirmations must be at least 1.")

        self.model_path = Path(model_path)
        self.real_threshold = float(real_threshold)
        self.spoof_threshold = float(
            spoof_threshold if spoof_threshold is not None else real_threshold - 0.08
        )
        self.window_size = int(window_size)
        self.min_frames = int(min_frames)
        self.required_confirmations = int(required_confirmations)
        self.max_stable_std = float(max_stable_std)

        self._probabilities: deque[float] = deque(maxlen=self.window_size)
        self._current_label = "UNCERTAIN"
        self._candidate_label: str | None = None
        self._candidate_count = 0

    def reset(self) -> None:
        """Clear temporal history, usually when no face is detected."""
        self._probabilities.clear()
        self._current_label = "UNCERTAIN"
        self._candidate_label = None
        self._candidate_count = 0

    def update(self, face_image: np.ndarray) -> dict[str, float | int | str]:
        """Add one face frame and return the current temporal liveness result."""
        raw_probability = _predict_real_probability(face_image, self.model_path)
        return self.update_probability(raw_probability)

    def update_probability(
        self,
        real_probability: float,
    ) -> dict[str, float | int | str]:
        """Add one raw REAL probability and return a temporal decision."""
        real_probability = float(np.clip(real_probability, 0.0, 1.0))
        self._probabilities.append(real_probability)

        temporal_probability = mean(self._probabilities)
        temporal_std = (
            pstdev(self._probabilities) if len(self._probabilities) > 1 else 0.0
        )
        stable_enough = temporal_std <= self.max_stable_std
        proposed_label = self._propose_label(
            temporal_probability,
            stable_enough,
        )
        self._apply_confirmation(proposed_label)

        return self._format_result(
            raw_probability=real_probability,
            temporal_probability=temporal_probability,
            temporal_std=temporal_std,
            stable_enough=stable_enough,
        )

    def _propose_label(
        self,
        temporal_probability: float,
        stable_enough: bool,
    ) -> str:
        if len(self._probabilities) < self.min_frames:
            return "UNCERTAIN"
        if not stable_enough:
            return "UNCERTAIN"
        if temporal_probability >= self.real_threshold:
            return "REAL"
        if temporal_probability <= self.spoof_threshold:
            return "SPOOF"
        return "UNCERTAIN"

    def _apply_confirmation(self, proposed_label: str) -> None:
        if proposed_label == "UNCERTAIN":
            self._candidate_label = None
            self._candidate_count = 0
            return

        if proposed_label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = proposed_label
            self._candidate_count = 1

        if self._candidate_count >= self.required_confirmations:
            self._current_label = proposed_label

    def _format_result(
        self,
        raw_probability: float,
        temporal_probability: float,
        temporal_std: float,
        stable_enough: bool,
    ) -> dict[str, float | int | str]:
        if self._current_label == "REAL":
            confidence = temporal_probability
        elif self._current_label == "SPOOF":
            confidence = 1.0 - temporal_probability
        else:
            confidence = 1.0 - abs(temporal_probability - 0.5) * 2.0

        return {
            "liveness": self._current_label,
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "raw_real_probability": round(raw_probability, 4),
            "temporal_probability": round(float(temporal_probability), 4),
            "temporal_std": round(float(temporal_std), 4),
            "stable_frames": len(self._probabilities),
            "stable_enough": "YES" if stable_enough else "NO",
            "method": "temporal_liveness",
        }


def _predict_real_probability(face_image: np.ndarray, model_path: Path) -> float:
    if model_path.suffix.lower() == ".pth":
        return _predict_torch_real_probability(face_image, model_path)

    return predict_liveness_probability(face_image, model_path=model_path)


def _predict_torch_real_probability(face_image: np.ndarray, model_path: Path) -> float:
    global _cached_torch_device
    global _cached_torch_model
    global _cached_torch_model_path
    global _cached_torch_transform

    if face_image is None:
        raise ValueError("face_image must not be None")
    if not isinstance(face_image, np.ndarray):
        raise TypeError("face_image must be a numpy.ndarray")
    if face_image.ndim != 3 or face_image.shape[2] != 3:
        raise ValueError("face_image must have shape (H, W, 3) in RGB format")

    resolved_model_path = model_path.resolve()
    if _cached_torch_model is None or _cached_torch_model_path != resolved_model_path:
        import torch
        from torchvision import transforms

        from anti_spoofing.liveness_models_kaixiang import load_checkpoint
        from anti_spoofing.liveness_training_kaixiang import IMAGENET_MEAN, IMAGENET_STD

        _cached_torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _cached_torch_model, _, _cached_torch_device = load_checkpoint(
            resolved_model_path,
            device=_cached_torch_device,
        )
        _cached_torch_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        _cached_torch_model_path = resolved_model_path

    import torch

    image = face_image
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    tensor = _cached_torch_transform(image).unsqueeze(0).to(_cached_torch_device)
    with torch.no_grad():
        logit = _cached_torch_model(tensor).squeeze(1)
        return float(torch.sigmoid(logit).item())

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


class TemporalLivenessDetector:
    """Convert frame-level liveness probabilities into a stable temporal result."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        real_threshold: float = DEFAULT_THRESHOLD,
        spoof_threshold: float | None = None,
        window_size: int = 15,
        min_frames: int = 5,
        required_confirmations: int = 3,
        max_stable_std: float = 0.18,
        min_motion_for_real: float = 0.004,
        enable_motion_check: bool = True,
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
        self.min_motion_for_real = float(min_motion_for_real)
        self.enable_motion_check = bool(enable_motion_check)

        self._probabilities: deque[float] = deque(maxlen=self.window_size)
        self._motion_scores: deque[float] = deque(maxlen=self.window_size)
        self._previous_gray_face: np.ndarray | None = None
        self._current_label = "UNCERTAIN"
        self._candidate_label: str | None = None
        self._candidate_count = 0

    def reset(self) -> None:
        """Clear temporal history, usually when no face is detected."""
        self._probabilities.clear()
        self._motion_scores.clear()
        self._previous_gray_face = None
        self._current_label = "UNCERTAIN"
        self._candidate_label = None
        self._candidate_count = 0

    def update(self, face_image: np.ndarray) -> dict[str, float | int | str]:
        """Add one face frame and return the current temporal liveness result."""
        raw_probability = predict_liveness_probability(
            face_image,
            model_path=self.model_path,
        )
        motion_score = self._calculate_motion_score(face_image)
        return self.update_probability(raw_probability, motion_score=motion_score)

    def update_probability(
        self,
        real_probability: float,
        motion_score: float | None = None,
    ) -> dict[str, float | int | str]:
        """Add one raw REAL probability and return a temporal decision."""
        real_probability = float(np.clip(real_probability, 0.0, 1.0))
        self._probabilities.append(real_probability)
        if motion_score is not None:
            self._motion_scores.append(float(max(0.0, motion_score)))

        temporal_probability = mean(self._probabilities)
        temporal_std = (
            pstdev(self._probabilities) if len(self._probabilities) > 1 else 0.0
        )
        stable_enough = temporal_std <= self.max_stable_std
        motion_score_average = mean(self._motion_scores) if self._motion_scores else 0.0
        motion_enough = self._has_enough_motion_for_real(motion_score_average)
        proposed_label = self._propose_label(
            temporal_probability,
            stable_enough,
            motion_enough,
        )
        self._apply_confirmation(proposed_label)

        return self._format_result(
            raw_probability=real_probability,
            temporal_probability=temporal_probability,
            temporal_std=temporal_std,
            stable_enough=stable_enough,
            motion_score=motion_score_average,
            motion_enough=motion_enough,
        )

    def _propose_label(
        self,
        temporal_probability: float,
        stable_enough: bool,
        motion_enough: bool,
    ) -> str:
        if len(self._probabilities) < self.min_frames:
            return "UNCERTAIN"
        if not stable_enough:
            return "UNCERTAIN"
        if temporal_probability >= self.real_threshold:
            if not motion_enough:
                return "UNCERTAIN"
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
        motion_score: float,
        motion_enough: bool,
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
            "motion_score": round(float(motion_score), 6),
            "stable_frames": len(self._probabilities),
            "stable_enough": "YES" if stable_enough else "NO",
            "motion_enough": "YES" if motion_enough else "NO",
            "method": "temporal_liveness",
        }

    def _calculate_motion_score(self, face_image: np.ndarray) -> float:
        gray_face = _to_gray_float(face_image)
        if self._previous_gray_face is None:
            self._previous_gray_face = gray_face
            return 0.0

        motion_score = float(np.mean(np.abs(gray_face - self._previous_gray_face)))
        self._previous_gray_face = gray_face
        return motion_score

    def _has_enough_motion_for_real(self, motion_score: float) -> bool:
        if not self.enable_motion_check:
            return True
        if len(self._probabilities) < self.min_frames:
            return False
        return motion_score >= self.min_motion_for_real


def _to_gray_float(face_image: np.ndarray) -> np.ndarray:
    image = face_image.astype("float32") / 255.0
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    return 0.299 * red + 0.587 * green + 0.114 * blue

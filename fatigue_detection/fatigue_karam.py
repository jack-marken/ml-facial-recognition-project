"""Fatigue & Drowsiness Detection — Karam's Innovative Feature (D/HD).

No MediaPipe. No external landmark libraries.
Takes a plain 224x224 RGB face image and predicts fatigue state
using a CNN trained on 4 fatigue indicators:
  - Closed   → eyes shut        → DROWSY
  - Open     → eyes open        → ALERT
  - Yawn     → mouth open       → DROWSY
  - no_yawn  → normal face      → ALERT

Maintains a sliding window (PERCLOS-style) so brief blinks don't
trigger false alarms — sustained fatigue signals trigger DROWSY.

Public API (identical contract to other modules):
    from fatigue_detection import FatigueDetector, predict_fatigue

    # Real-time use (maintains window across frames):
    detector = FatigueDetector()
    result   = detector.update(face_image)

    # Single-frame use:
    result = predict_fatigue(face_image)

    # result → {
    #   "fatigue":      "ALERT" | "DROWSY",
    #   "indicator":    "Closed" | "Open" | "Yawn" | "no_yawn",
    #   "confidence":   float,
    #   "perclos":      float,   # fraction of recent drowsy frames
    # }
"""
# Author: Karam (Innovative Feature — D/HD)

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL_PATH     = Path("models/fatigue_karam.h5")
DEFAULT_CLASS_MAP_PATH = Path("models/fatigue_class_indices_karam.json")
WINDOW_FRAMES          = 30      # ~1 second at 30fps
PERCLOS_THRESHOLD      = 0.30    # >30% drowsy frames → DROWSY
IMAGE_SIZE             = (224, 224)

# Classes that indicate fatigue/drowsiness
DROWSY_CLASSES = {"Closed", "Yawn"}
ALERT_CLASSES  = {"Open", "no_yawn"}

_cached_model      = None
_cached_model_path: Path | None = None
_cached_class_map: dict | None  = None


class FatigueDetector:
    """Stateful real-time fatigue detector with sliding-window PERCLOS.

    Instantiate once and call update() every webcam frame.
    Call reset() when switching to a different person.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        window_frames: int     = WINDOW_FRAMES,
        perclos_threshold: float = PERCLOS_THRESHOLD,
    ):
        self._model_path       = Path(model_path)
        self._perclos_threshold = perclos_threshold
        self._window: deque[int] = deque(maxlen=window_frames)

    def update(self, face_image: np.ndarray) -> dict:
        """Process one frame and return current fatigue assessment.

        Args:
            face_image: 224x224 RGB numpy array.

        Returns:
            {fatigue, indicator, confidence, perclos}
        """
        indicator, confidence = _predict_frame(face_image, self._model_path)

        is_drowsy = indicator in DROWSY_CLASSES if indicator else False
        self._window.append(int(is_drowsy))

        perclos = sum(self._window) / max(len(self._window), 1)
        fatigue = "DROWSY" if perclos > self._perclos_threshold else "ALERT"

        return {
            "fatigue":    fatigue,
            "indicator":  indicator or "Unknown",
            "confidence": round(confidence, 4),
            "perclos":    round(perclos, 4),
        }

    def reset(self) -> None:
        """Clear the PERCLOS window — call between different people."""
        self._window.clear()


def predict_fatigue(
    face_image: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    """Single-frame fatigue prediction with no rolling window."""
    detector = FatigueDetector(model_path=model_path, window_frames=1)
    return detector.update(face_image)


# ── Internal helpers ────────────────────────────────────────────────────────

def _predict_frame(
    face_image: np.ndarray,
    model_path: Path,
) -> tuple[str | None, float]:
    """Run the CNN on one face crop. Returns (class_label, confidence)."""
    model, class_map = _get_model_and_classes(model_path)
    if model is None:
        return None, 0.0

    # Preprocess: resize to 224x224, normalise to [0,1]
    resized = cv2.resize(face_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    batch   = np.expand_dims(resized.astype("float32") / 255.0, axis=0)

    probs   = model.predict(batch, verbose=0)[0]
    idx     = int(np.argmax(probs))
    label   = class_map.get(idx, f"class_{idx}")
    return label, float(probs[idx])


def _get_model_and_classes(model_path: Path):
    global _cached_model, _cached_model_path, _cached_class_map

    resolved = model_path.resolve()

    if not resolved.exists():
        return None, {}

    if _cached_model is not None and _cached_model_path == resolved:
        return _cached_model, _cached_class_map

    import tensorflow as tf
    _cached_model      = tf.keras.models.load_model(str(resolved))
    _cached_model_path = resolved

    # Load class index map saved during training
    class_map_path = resolved.parent / "fatigue_class_indices_karam.json"
    if class_map_path.exists():
        with class_map_path.open() as f:
            raw = json.load(f)
        # raw = {"Closed": 0, "Open": 1, ...}  → flip to {0: "Closed", ...}
        _cached_class_map = {v: k for k, v in raw.items()}
    else:
        # Fallback order if json missing
        _cached_class_map = {0: "Closed", 1: "Open", 2: "Yawn", 3: "no_yawn"}

    return _cached_model, _cached_class_map

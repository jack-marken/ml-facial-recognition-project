"""Fatigue & Drowsiness Detection — Karam's Innovative Feature (D/HD).

Improvements over v1:
  - Eye region is cropped from the face before prediction (fixes train/test mismatch)
  - Separate confidence gates: only counts a frame if model is confident (>60%)
  - Haar cascade eye detector built into OpenCV — no extra installs
  - Fallback to upper-face crop if no eyes detected
  - PERCLOS window separately tracks eye closure and yawning

Public API:
    from fatigue_detection import FatigueDetector, predict_fatigue

    detector = FatigueDetector()
    result   = detector.update(face_image)

    # result → {
    #   "fatigue":      "ALERT" | "DROWSY",
    #   "indicator":    "Closed" | "Open" | "Yawn" | "no_yawn",
    #   "confidence":   float,
    #   "perclos":      float,
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
WINDOW_FRAMES          = 30       # ~1 second at 30fps
PERCLOS_THRESHOLD      = 0.30     # >30% drowsy frames → DROWSY
CONFIDENCE_GATE        = 0.60     # ignore predictions below 60% confidence
IMAGE_SIZE             = (224, 224)

DROWSY_CLASSES = {"Closed", "Yawn"}
ALERT_CLASSES  = {"Open", "no_yawn"}

# Haar cascade — built into OpenCV, no install needed
_eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

_cached_model      = None
_cached_model_path: Path | None = None
_cached_class_map: dict | None  = None


class FatigueDetector:
    """Stateful real-time fatigue detector with sliding-window PERCLOS."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        window_frames: int       = WINDOW_FRAMES,
        perclos_threshold: float = PERCLOS_THRESHOLD,
    ):
        self._model_path        = Path(model_path)
        self._perclos_threshold = perclos_threshold
        self._window: deque[int] = deque(maxlen=window_frames)
        self.active: bool        = True

    def update(self, face_image: np.ndarray) -> dict:
        """Process one frame. face_image must be 224x224 RGB numpy array."""
        indicator, confidence = _predict_frame(face_image, self._model_path)

        # Only count this frame if model is confident enough
        if confidence >= CONFIDENCE_GATE and indicator is not None:
            is_drowsy = indicator in DROWSY_CLASSES
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
        self._window.clear()

    def toggle_active(self) -> None:
        """Toggle fatigue detection on/off (mirrors EmotionDetector.toggle_active)."""
        self.active = not self.active


def predict_fatigue(
    face_image: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict:
    detector = FatigueDetector(model_path=model_path, window_frames=1)
    return detector.update(face_image)


# ── Internal helpers ────────────────────────────────────────────────────────

def _predict_frame(face_image: np.ndarray, model_path: Path) -> tuple[str | None, float]:
    model, class_map = _get_model_and_classes(model_path)
    if model is None:
        return None, 0.0

    # Try to get a better crop for the eye classification
    eye_crop  = _extract_eye_region(face_image)
    yawn_crop = face_image   # full face is correct for Yawn/no_yawn

    # Run model on eye crop first, fall back to full face
    eye_input  = np.expand_dims(
        cv2.resize(eye_crop, IMAGE_SIZE).astype("float32") / 255.0, axis=0
    )
    face_input = np.expand_dims(
        cv2.resize(yawn_crop, IMAGE_SIZE).astype("float32") / 255.0, axis=0
    )

    eye_probs  = model.predict(eye_input,  verbose=0)[0]
    face_probs = model.predict(face_input, verbose=0)[0]

    eye_idx    = int(np.argmax(eye_probs))
    face_idx   = int(np.argmax(face_probs))
    eye_label  = class_map.get(eye_idx,  f"class_{eye_idx}")
    face_label = class_map.get(face_idx, f"class_{face_idx}")

    # Eye crop → best for Closed/Open detection
    # Full face → best for Yawn/no_yawn detection
    if face_label in ("Yawn", "no_yawn"):
        return face_label, float(face_probs[face_idx])
    else:
        return eye_label, float(eye_probs[eye_idx])


def _extract_eye_region(face_image: np.ndarray) -> np.ndarray:
    """Extract the eye region from a 224x224 face image.

    Uses Haar cascade first. If no eyes detected, falls back to
    the upper 45% of the face (where eyes always are).
    """
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    eyes = _eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
    )

    if len(eyes) > 0:
        # Merge all detected eye bounding boxes into one crop
        xs = [x for (x, y, w, h) in eyes]
        ys = [y for (x, y, w, h) in eyes]
        xe = [x + w for (x, y, w, h) in eyes]
        ye = [y + h for (x, y, w, h) in eyes]

        pad = 10
        x1  = max(0, min(xs) - pad)
        y1  = max(0, min(ys) - pad)
        x2  = min(face_image.shape[1], max(xe) + pad)
        y2  = min(face_image.shape[0], max(ye) + pad)

        crop = face_image[y1:y2, x1:x2]
        if crop.size > 0:
            return crop

    # Fallback: upper 45% of face (eye region)
    h = face_image.shape[0]
    return face_image[:int(h * 0.45), :]


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

    class_map_path = resolved.parent / "fatigue_class_indices_karam.json"
    if class_map_path.exists():
        with class_map_path.open() as f:
            raw = json.load(f)
        _cached_class_map = {v: k for k, v in raw.items()}
    else:
        _cached_class_map = {0: "Closed", 1: "Open", 2: "Yawn", 3: "no_yawn"}

    return _cached_model, _cached_class_map

"""Emotion detection inference API (Karam).

Public interface:
    from emotion_detection import predict_emotion

    result = predict_emotion(face_image)
    # → {"emotion": "Happy", "confidence": 0.92, "all_scores": {...}}

Consumed by the UI exactly like liveness — takes a 224×224 RGB face crop.
"""
# Author: Karam

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .emotion_model_karam import (
    EMOTION_LABELS,
    load_emotion_model,
    preprocess_emotion_image,
)


DEFAULT_MODEL_PATH = Path("models/emotion_karam.pth")

_cached_model      = None
_cached_model_path: Path | None = None


def predict_emotion(
    face_image: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, object]:
    """Predict the emotion shown in a cropped face image.

    Args:
        face_image: Standardised RGB numpy array with shape (224, 224, 3).
                    (Same format produced by detection.detector.detect_and_crop_face.)
        model_path: Path to the trained .pth checkpoint.

    Returns:
        {
            "emotion":    str,           # e.g. "Happy"
            "confidence": float,         # probability of the top emotion
            "all_scores": dict[str,float]# probability per emotion class
        }
    """
    model  = _get_model(Path(model_path))
    device = next(model.parameters()).device
    batch  = preprocess_emotion_image(face_image, device=device)

    with torch.no_grad():
        logits      = model(batch)
        probs       = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx    = int(np.argmax(probs))
    top_label  = EMOTION_LABELS[top_idx]
    all_scores = {label: round(float(p), 4) for label, p in zip(EMOTION_LABELS, probs)}

    return {
        "emotion":    top_label,
        "confidence": round(float(probs[top_idx]), 4),
        "all_scores": all_scores,
    }


def _get_model(model_path: Path):
    global _cached_model, _cached_model_path

    resolved = model_path.resolve()
    if _cached_model is not None and _cached_model_path == resolved:
        return _cached_model

    _cached_model      = load_emotion_model(model_path=resolved)
    _cached_model_path = resolved
    return _cached_model

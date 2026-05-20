"""Reusable liveness prediction API.

This module intentionally consumes an already-cropped face image. The shared
project contract is a 224x224 RGB numpy array, so detection and face cropping
should stay outside predict_liveness().
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


EXPECTED_FACE_SHAPE = (224, 224, 3)
DEFAULT_MODEL_PATH = Path("models/liveness_densenet121_zhongyu.keras")
DEFAULT_THRESHOLD = 0.43

_cached_model = None
_cached_model_path: Path | None = None


def preprocess_face_from_bbox(
    frame: np.ndarray,
    bbox: Iterable[int],
    output_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Temporary helper until detection exposes shared face preprocessing.

    Args:
        frame: Original OpenCV webcam frame in BGR format.
        bbox: Face box as [x1, y1, x2, y2].
        output_size: Output size as (width, height).

    Returns:
        Cropped face image as RGB numpy array with shape (224, 224, 3).
    """
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("frame must be a numpy.ndarray.")

    bbox_values = list(bbox)
    if len(bbox_values) != 4:
        raise ValueError("bbox must contain four values: [x1, y1, x2, y2].")

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox_values]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox does not describe a valid crop area.")

    face_bgr = frame[y1:y2, x1:x2]
    face_bgr = cv2.resize(face_bgr, output_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)


def predict_liveness(
    face_image: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | str]:
    """Predict whether a cropped face is real or spoofed.

    Args:
        face_image: Standardized RGB face image with shape (224, 224, 3).
        model_path: Saved Keras model path.
        threshold: Probability threshold for classifying a face as REAL.

    Returns:
        Dictionary matching the unified integration guideline:
        {"liveness": "REAL" | "SPOOF", "confidence": float}
    """
    validated_face = _validate_face_image(face_image)
    model = _load_model(Path(model_path))

    batch = np.expand_dims(validated_face.astype("float32"), axis=0)
    raw_prediction = model.predict(batch, verbose=0)
    real_probability = _extract_real_probability(raw_prediction)
    return format_liveness_result(real_probability, threshold=threshold)


def predict_liveness_probability(
    face_image: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> float:
    """Return the raw REAL probability for temporal smoothing."""
    validated_face = _validate_face_image(face_image)
    model = _load_model(Path(model_path))

    batch = np.expand_dims(validated_face.astype("float32"), axis=0)
    raw_prediction = model.predict(batch, verbose=0)
    return _extract_real_probability(raw_prediction)


def format_liveness_result(
    real_probability: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | str]:
    """Convert a REAL probability into the shared liveness output format."""
    real_probability = float(np.clip(real_probability, 0.0, 1.0))

    if real_probability >= threshold:
        return {
            "liveness": "REAL",
            "confidence": round(real_probability, 4),
        }

    return {
        "liveness": "SPOOF",
        "confidence": round(1.0 - real_probability, 4),
    }


def _validate_face_image(face_image: np.ndarray) -> np.ndarray:
    if face_image is None or not isinstance(face_image, np.ndarray):
        raise ValueError("face_image must be a numpy.ndarray.")

    if face_image.shape != EXPECTED_FACE_SHAPE:
        raise ValueError(
            f"face_image must have shape {EXPECTED_FACE_SHAPE}, "
            f"received {face_image.shape}."
        )

    if face_image.ndim != 3:
        raise ValueError("face_image must be an RGB image with three channels.")

    return face_image


def _load_model(model_path: Path):
    global _cached_model, _cached_model_path

    resolved_model_path = model_path.resolve()
    if _cached_model is not None and _cached_model_path == resolved_model_path:
        return _cached_model

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Liveness model not found at {resolved_model_path}. "
            "Train a model first or pass model_path to predict_liveness()."
        )

    if resolved_model_path.name.endswith(".weights.h5"):
        architecture = _infer_architecture_from_path(resolved_model_path)
        try:
            from .model_factory_zhongyu import build_liveness_model
        except ImportError:
            from model_factory_zhongyu import build_liveness_model

        _cached_model = build_liveness_model(architecture=architecture)
        _cached_model.load_weights(resolved_model_path)
        _cached_model_path = resolved_model_path
        return _cached_model

    from tensorflow.keras.models import load_model

    try:
        _cached_model = load_model(resolved_model_path, compile=False)
        _cached_model_path = resolved_model_path
        return _cached_model
    except TypeError as error:
        if "resnet50v2" in resolved_model_path.name.lower():
            raise TypeError(
                "ResNet50V2 full .keras model could not be loaded in this Keras "
                "version. Retrain ResNet50V2 with the default wrapper so it saves "
                "to models/liveness_resnet50v2_zhongyu.weights.h5, then pass that "
                "weights file to --model."
            ) from error
        raise


def _infer_architecture_from_path(model_path: Path) -> str:
    file_name = model_path.name.lower()
    if "resnet50v2" in file_name:
        return "resnet50v2"
    if "densenet121" in file_name:
        return "densenet121"
    raise ValueError(
        "Cannot infer liveness architecture from weights filename. "
        "Use a filename containing 'resnet50v2' or 'densenet121'."
    )


def _extract_real_probability(raw_prediction) -> float:
    prediction = np.asarray(raw_prediction).reshape(-1)
    if prediction.size != 1:
        raise ValueError(
            "Liveness model must output a single sigmoid probability for REAL."
        )

    return float(np.clip(prediction[0], 0.0, 1.0))

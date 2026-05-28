"""Face recognition inference API for Karam's classification-based module.

Output dictionary format is identical to recognition_zhongyu.py so either
module can be swapped in the UI without any other changes:

    {
        "identity":             str,    # matched name, or "UNKNOWN"
        "similarity_score":     float,  # cosine similarity to best match
        "best_identity":        str,    # top candidate regardless of threshold
        "best_similarity_score":float,
        "distance_metric":      "cosine",
        "method":               "classification",
    }
"""
# Author: Karam

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .classification_model_karam import (
    generate_embedding,
    load_classification_model,
)


DEFAULT_MODEL_PATH   = Path("models/recognition_classification_karam.pth")
DEFAULT_GALLERY_PATH = Path("models/recognition_gallery_classification_karam.pkl")
DEFAULT_THRESHOLD    = 0.65
DISTANCE_METRIC      = "cosine"

_cached_model        = None
_cached_model_path: Path | None = None
_cached_gallery      = None
_cached_gallery_path: Path | None = None


def predict_identity_classification(
    face_image: np.ndarray,
    gallery_path: str | Path = DEFAULT_GALLERY_PATH,
    model_path:   str | Path = DEFAULT_MODEL_PATH,
    threshold:    float      = DEFAULT_THRESHOLD,
) -> dict[str, float | str]:
    """Predict the closest registered identity from a cropped RGB face image.

    Args:
        face_image:   Standardised RGB numpy array with shape (224, 224, 3).
        gallery_path: Pickle gallery built by build_gallery_karam.py.
        model_path:   Trained .pth checkpoint.
        threshold:    Minimum cosine similarity to accept a match.

    Returns:
        Identity result dictionary (see module docstring).
    """
    scores = _calculate_identity_scores(face_image, gallery_path, model_path)
    return _format_result(scores, threshold=threshold)


def _calculate_identity_scores(
    face_image:   np.ndarray,
    gallery_path: str | Path,
    model_path:   str | Path,
) -> dict[str, float]:
    model   = _get_model(Path(model_path))
    gallery = _get_gallery(Path(gallery_path))
    query   = generate_embedding(model, face_image)

    return {
        identity: _cosine_similarity(query, gallery_embedding)
        for identity, gallery_embedding in gallery["embeddings"].items()
    }


def _format_result(
    identity_scores: dict[str, float],
    threshold: float,
) -> dict[str, float | str]:
    if not identity_scores:
        return {
            "identity":              "UNKNOWN",
            "similarity_score":      0.0,
            "best_identity":         "UNKNOWN",
            "best_similarity_score": 0.0,
            "distance_metric":       DISTANCE_METRIC,
            "method":                "classification",
        }

    best_identity = max(identity_scores, key=identity_scores.get)
    best_score    = float(identity_scores[best_identity])
    identity      = best_identity if best_score >= threshold else "UNKNOWN"

    return {
        "identity":              identity,
        "similarity_score":      round(best_score, 4),
        "best_identity":         best_identity,
        "best_similarity_score": round(best_score, 4),
        "distance_metric":       DISTANCE_METRIC,
        "method":                "classification",
    }


def _get_model(model_path: Path):
    global _cached_model, _cached_model_path

    resolved = model_path.resolve()
    if _cached_model is not None and _cached_model_path == resolved:
        return _cached_model

    _cached_model      = load_classification_model(model_path=resolved)
    _cached_model_path = resolved
    return _cached_model


def _get_gallery(gallery_path: Path) -> dict[str, Any]:
    global _cached_gallery, _cached_gallery_path

    resolved = gallery_path.resolve()
    if _cached_gallery is not None and _cached_gallery_path == resolved:
        return _cached_gallery

    if not resolved.exists():
        raise FileNotFoundError(
            f"Classification gallery not found: {resolved}. "
            "Run build_gallery_karam.py first."
        )

    with resolved.open("rb") as fh:
        _cached_gallery = pickle.load(fh)

    _cached_gallery_path = resolved
    return _cached_gallery


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

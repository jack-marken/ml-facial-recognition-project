"""Face recognition / verification API for Zhongyu's metric-learning module."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .embedding_model_zhongyu import (
    DEFAULT_ARCHITECTURE,
    generate_embedding,
    load_embedding_model,
)


DEFAULT_GALLERY_PATH = Path("models/recognition_gallery_zhongyu.pkl")
DEFAULT_MODEL_PATH: Path | None = None
DEFAULT_THRESHOLD = 0.65
DISTANCE_METRIC = "cosine"

_cached_model = None
_cached_model_key: tuple[str, str | None] | None = None
_cached_gallery = None
_cached_gallery_path: Path | None = None


def predict_identity_metric(
    face_image: np.ndarray,
    gallery_path: str | Path = DEFAULT_GALLERY_PATH,
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    architecture: str = DEFAULT_ARCHITECTURE,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | str]:
    """Predict the closest registered identity from a cropped RGB face image."""
    identity_scores = calculate_identity_scores(
        face_image,
        gallery_path=gallery_path,
        model_path=model_path,
        architecture=architecture,
    )
    return format_identity_result(identity_scores, threshold=threshold)


def calculate_identity_scores(
    face_image: np.ndarray,
    gallery_path: str | Path = DEFAULT_GALLERY_PATH,
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    architecture: str = DEFAULT_ARCHITECTURE,
) -> dict[str, float]:
    """Return cosine similarity scores for all registered identities."""
    model = _get_model(architecture=architecture, model_path=model_path)
    gallery = _get_gallery(Path(gallery_path))
    query_embedding = generate_embedding(model, face_image)

    return {
        identity: _cosine_similarity(query_embedding, gallery_embedding)
        for identity, gallery_embedding in gallery["embeddings"].items()
    }


def format_identity_result(
    identity_scores: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | str]:
    """Convert identity scores into the shared metric-learning output format."""
    if not identity_scores:
        return {
            "identity": "UNKNOWN",
            "similarity_score": 0.0,
            "distance_metric": DISTANCE_METRIC,
            "method": "metric_learning",
        }

    best_identity = max(identity_scores, key=identity_scores.get)
    best_score = float(identity_scores[best_identity])

    if best_score < threshold:
        identity = "UNKNOWN"
    else:
        identity = best_identity

    return {
        "identity": identity,
        "similarity_score": round(float(best_score), 4),
        "distance_metric": DISTANCE_METRIC,
        "method": "metric_learning",
    }


def _get_model(architecture: str, model_path: str | Path | None):
    global _cached_model, _cached_model_key

    model_path_string = str(model_path) if model_path else None
    model_key = (architecture, model_path_string)
    if _cached_model is not None and _cached_model_key == model_key:
        return _cached_model

    _cached_model = load_embedding_model(
        architecture=architecture,
        model_path=model_path,
        pretrained=model_path is None,
    )
    _cached_model_key = model_key
    return _cached_model


def _get_gallery(gallery_path: Path) -> dict[str, Any]:
    global _cached_gallery, _cached_gallery_path

    resolved_path = gallery_path.resolve()
    if _cached_gallery is not None and _cached_gallery_path == resolved_path:
        return _cached_gallery

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Recognition gallery not found: {resolved_path}. "
            "Run build_gallery_zhongyu.py first."
        )

    with resolved_path.open("rb") as file:
        _cached_gallery = pickle.load(file)

    _cached_gallery_path = resolved_path
    return _cached_gallery


def _find_best_match(
    query_embedding: np.ndarray,
    gallery_embeddings: dict[str, np.ndarray],
) -> tuple[str, float]:
    best_identity = "UNKNOWN"
    best_score = -1.0

    for identity, gallery_embedding in gallery_embeddings.items():
        score = _cosine_similarity(query_embedding, gallery_embedding)
        if score > best_score:
            best_identity = identity
            best_score = score

    return best_identity, best_score


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)

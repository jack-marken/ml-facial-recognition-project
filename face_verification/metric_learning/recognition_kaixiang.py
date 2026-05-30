import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from face_verification.metric_learning.siamese_dataset_kaixiang import (
    face_image_to_tensor,
)
from face_verification.metric_learning.siamese_models_kaixiang import (
    load_siamese_checkpoint,
)


DEFAULT_GALLERY_PATH = Path("models/recognition_gallery_kaixiang.pkl")
DEFAULT_MODEL_PATH = Path("models/recognition_triplet_resnet18_kaixiang_final30b_best.pth")
DEFAULT_DISTANCE_THRESHOLD = 0.006
DEFAULT_DISTANCE_MARGIN = 0.0003
DEFAULT_MATCHING_MODE = "mean"
DEFAULT_TOP_K = 3

_cached_model = None
_cached_model_path: Path | None = None
_cached_gallery = None
_cached_gallery_path: Path | None = None


def predict_identity(
    face_image: np.ndarray,
    gallery_path: str | Path = DEFAULT_GALLERY_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    distance_margin: float = DEFAULT_DISTANCE_MARGIN,
    matching_mode: str = DEFAULT_MATCHING_MODE,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, float | str]:
    scores = calculate_identity_distances(
        face_image,
        gallery_path=gallery_path,
        model_path=model_path,
        matching_mode=matching_mode,
        top_k=top_k,
    )
    return format_identity_result(
        scores,
        distance_threshold=distance_threshold,
        distance_margin=distance_margin,
    )


def calculate_identity_distances(
    face_image: np.ndarray,
    gallery_path: str | Path = DEFAULT_GALLERY_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    matching_mode: str = DEFAULT_MATCHING_MODE,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, float]:
    model = _get_model(Path(model_path))
    gallery = _get_gallery(Path(gallery_path))
    query_embedding = generate_embedding(model, face_image)

    if matching_mode == "sample" and "sample_embeddings" in gallery:
        return {
            identity: float(
                min(np.linalg.norm(query_embedding - sample_embedding) for sample_embedding in sample_embeddings)
            )
            for identity, sample_embeddings in gallery["sample_embeddings"].items()
        }

    if matching_mode == "topk" and "sample_embeddings" in gallery:
        safe_top_k = max(1, int(top_k))
        return {
            identity: float(
                np.mean(
                    sorted(
                        np.linalg.norm(query_embedding - sample_embedding)
                        for sample_embedding in sample_embeddings
                    )[:safe_top_k]
                )
            )
            for identity, sample_embeddings in gallery["sample_embeddings"].items()
        }

    if matching_mode != "mean":
        raise ValueError("matching_mode must be 'mean', 'sample', or 'topk'.")

    return {
        identity: float(np.linalg.norm(query_embedding - gallery_embedding))
        for identity, gallery_embedding in gallery["embeddings"].items()
    }


def format_identity_result(
    identity_distances: dict[str, float],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    distance_margin: float = DEFAULT_DISTANCE_MARGIN,
) -> dict[str, float | str]:
    if not identity_distances:
        return {
            "identity": "UNKNOWN",
            "distance": 0.0,
            "similarity_score": 0.0,
            "method": "siamese_metric_learning",
        }

    best_identity = min(identity_distances, key=identity_distances.get)
    best_distance = float(identity_distances[best_identity])
    similarity_score = 1.0 / (1.0 + best_distance)
    top_matches = sorted(identity_distances.items(), key=lambda item: item[1])[:3]
    second_distance = float(top_matches[1][1]) if len(top_matches) > 1 else float("inf")
    distance_gap = second_distance - best_distance
    is_ambiguous = distance_gap < distance_margin
    identity = best_identity if best_distance <= distance_threshold and not is_ambiguous else "UNKNOWN"

    return {
        "identity": identity,
        "best_identity": best_identity,
        "distance": round(best_distance, 4),
        "distance_threshold": round(distance_threshold, 4),
        "distance_margin": round(distance_margin, 4),
        "distance_gap": round(distance_gap, 4),
        "ambiguous": is_ambiguous,
        "similarity_score": round(similarity_score, 4),
        "top_matches": [
            {"identity": match_identity, "distance": round(float(distance), 4)}
            for match_identity, distance in top_matches
        ],
        "method": "siamese_metric_learning",
    }


@torch.no_grad()
def generate_embedding(model, face_image: np.ndarray) -> np.ndarray:
    if face_image is None or not isinstance(face_image, np.ndarray):
        raise ValueError("face_image must be a numpy.ndarray.")
    if face_image.shape != (224, 224, 3):
        raise ValueError(f"face_image must have shape (224, 224, 3), got {face_image.shape}.")

    device = next(model.parameters()).device
    tensor = face_image_to_tensor(face_image).unsqueeze(0).to(device)
    embedding = model.forward_once(tensor).squeeze(0).detach().cpu().numpy()
    return embedding.astype("float32")


def _get_model(model_path: Path):
    global _cached_model, _cached_model_path

    resolved_path = model_path.resolve()
    if _cached_model is not None and _cached_model_path == resolved_path:
        return _cached_model

    if not resolved_path.exists():
        raise FileNotFoundError(f"Recognition model not found: {resolved_path}")

    _cached_model, _, _ = load_siamese_checkpoint(resolved_path)
    _cached_model_path = resolved_path
    return _cached_model


def _get_gallery(gallery_path: Path) -> dict[str, Any]:
    global _cached_gallery, _cached_gallery_path

    resolved_path = gallery_path.resolve()
    if _cached_gallery is not None and _cached_gallery_path == resolved_path:
        return _cached_gallery

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Recognition gallery not found: {resolved_path}. "
            "Run build_gallery_kaixiang.py first."
        )

    with resolved_path.open("rb") as file:
        _cached_gallery = pickle.load(file)

    _cached_gallery_path = resolved_path
    return _cached_gallery

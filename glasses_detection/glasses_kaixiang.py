from pathlib import Path

import numpy as np
import torch

from glasses_detection.dataset_kaixiang import face_image_to_tensor
from glasses_detection.models_kaixiang import load_glasses_checkpoint


DEFAULT_MODEL_PATH = Path("models/glasses_efficientnetb0_kaixiang_final_best.pth")
DEFAULT_THRESHOLD = 0.5

_cached_model = None
_cached_model_path = None


def predict_glasses(face_image, model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD):
    probability = calculate_glasses_probability(face_image, model_path=model_path)
    return format_glasses_result(probability, threshold=threshold)


@torch.no_grad()
def calculate_glasses_probability(face_image, model_path=DEFAULT_MODEL_PATH):
    if face_image is None or not isinstance(face_image, np.ndarray):
        raise ValueError("face_image must be a numpy.ndarray.")
    if face_image.shape != (224, 224, 3):
        raise ValueError(f"face_image must have shape (224, 224, 3), got {face_image.shape}.")

    model = _get_model(Path(model_path))
    device = next(model.parameters()).device
    tensor = face_image_to_tensor(face_image).unsqueeze(0).to(device)
    logit = model(tensor).squeeze(0)
    return float(torch.sigmoid(logit).detach().cpu().item())


def format_glasses_result(probability, threshold=DEFAULT_THRESHOLD):
    label = "with_glasses" if probability >= threshold else "without_glasses"
    confidence = probability if label == "with_glasses" else 1.0 - probability
    return {
        "label": label,
        "with_glasses_probability": round(float(probability), 4),
        "confidence": round(float(confidence), 4),
        "threshold": threshold,
        "method": "glasses_detection_binary_classifier",
    }


def _get_model(model_path):
    global _cached_model, _cached_model_path
    resolved_path = model_path.resolve()
    if _cached_model is not None and _cached_model_path == resolved_path:
        return _cached_model
    if not resolved_path.exists():
        raise FileNotFoundError(f"Glasses model not found: {resolved_path}")
    _cached_model, _, _ = load_glasses_checkpoint(resolved_path)
    _cached_model_path = resolved_path
    return _cached_model


from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms

from anti_spoofing.liveness_models_kaixiang import load_checkpoint
from anti_spoofing.liveness_training_kaixiang import IMAGENET_MEAN, IMAGENET_STD


DEFAULT_CHECKPOINT = Path("models/liveness_mobilenetv2_kaixiang_best.pth")


class LivenessTemporalSmootherKaixiang:
    """Stabilize liveness labels across recent webcam frames."""

    def __init__(self, window_size=5, spoof_votes=3):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if spoof_votes <= 0 or spoof_votes > window_size:
            raise ValueError("spoof_votes must be between 1 and window_size")

        self.window_size = window_size
        self.spoof_votes = spoof_votes
        self.labels = deque(maxlen=window_size)

    def reset(self):
        self.labels.clear()

    def update(self, result):
        if "liveness" not in result:
            self.reset()
            return result

        raw_liveness = result["liveness"]
        self.labels.append(raw_liveness)

        spoof_count = sum(label == "SPOOF" for label in self.labels)
        real_count = sum(label == "REAL" for label in self.labels)

        smoothed = dict(result)
        if len(self.labels) < self.spoof_votes:
            smoothed["liveness"] = "CHECKING"
        elif spoof_count >= self.spoof_votes:
            smoothed["liveness"] = "SPOOF"
        else:
            smoothed["liveness"] = "REAL"

        smoothed["raw_liveness"] = raw_liveness
        smoothed["smooth_window"] = len(self.labels)
        smoothed["smooth_real_votes"] = real_count
        smoothed["smooth_spoof_votes"] = spoof_count
        return smoothed


class LivenessPredictorKaixiang:
    """Inference wrapper for Kaixiang's fine-tuned liveness model."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None, threshold=0.5):
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Missing trained liveness checkpoint. Train the model first: "
                f"{self.checkpoint_path}"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.checkpoint, self.device = load_checkpoint(
            self.checkpoint_path, self.device
        )
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def predict(self, face_image):
        """Predict liveness from a cropped RGB face image.

        Args:
            face_image: numpy.ndarray, shape (224, 224, 3), RGB format.
        """

        if face_image is None:
            return {"status": "NO_FACE", "message": "No face image provided"}

        if not isinstance(face_image, np.ndarray):
            return {
                "status": "INVALID_INPUT",
                "message": "face_image must be a numpy.ndarray",
            }

        if face_image.ndim != 3 or face_image.shape[2] != 3:
            return {
                "status": "INVALID_INPUT",
                "message": "face_image must have shape (H, W, 3) in RGB format",
            }

        image = face_image
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logit = self.model(tensor).squeeze(1)
        real_probability = float(torch.sigmoid(logit).item())

        if real_probability >= self.threshold:
            return {
                "liveness": "REAL",
                "confidence": round(real_probability, 4),
                "model": self.checkpoint["model_name"],
            }

        return {
            "liveness": "SPOOF",
            "confidence": round(1.0 - real_probability, 4),
            "model": self.checkpoint["model_name"],
        }


_DEFAULT_PREDICTOR = None


def predict_liveness(face_image, checkpoint_path=DEFAULT_CHECKPOINT, threshold=0.7):
    """Standard integration function for downstream modules."""

    global _DEFAULT_PREDICTOR
    checkpoint_path = Path(checkpoint_path)

    if (
        _DEFAULT_PREDICTOR is None
        or _DEFAULT_PREDICTOR.checkpoint_path != checkpoint_path
        or _DEFAULT_PREDICTOR.threshold != threshold
    ):
        _DEFAULT_PREDICTOR = LivenessPredictorKaixiang(
            checkpoint_path,
            threshold=threshold,
        )

    return _DEFAULT_PREDICTOR.predict(face_image)


def predict_liveness_from_bgr_frame(
    frame,
    bbox,
    checkpoint_path=DEFAULT_CHECKPOINT,
    threshold=0.5,
):
    """Helper for manual tests when only a BGR frame and bbox are available."""

    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"status": "CROP_FAILED", "message": "Empty face crop"}
    rgb_face = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb_face = cv2.resize(rgb_face, (224, 224))
    return predict_liveness(rgb_face, checkpoint_path=checkpoint_path, threshold=threshold)

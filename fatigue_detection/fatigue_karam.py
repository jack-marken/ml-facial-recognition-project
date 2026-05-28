"""Fatigue and Drowsiness Detection — Karam's Innovative Feature (D/HD).

Method (two complementary signals):
  1. Eye Aspect Ratio (EAR): computed from MediaPipe FaceMesh landmarks.
     EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
     Falls sharply when the eye closes.

  2. CNN eye-state classifier: EfficientNet-B0 trained on eye crops to
     classify each eye as open (1) or closed (0) — more robust than raw EAR
     under poor lighting or glasses.

  3. PERCLOS (sliding window): percentage of frames in the last N frames
     where both eyes are classified as closed. PERCLOS > threshold → DROWSY.

Public interface:
    detector = FatigueDetector(model_path="models/fatigue_eye_karam.pth")
    result   = detector.update(face_image)

    # or stateless single-frame call:
    result = predict_fatigue(face_image, model_path="models/fatigue_eye_karam.pth")

    # result → {
    #   "fatigue":      "ALERT" | "DROWSY",
    #   "ear":          float,   # mean EAR this frame
    #   "perclos":      float,   # fraction of recent frames with closed eyes
    #   "confidence":   float,   # CNN closed-eye probability (mean both eyes)
    # }

Requires:
    pip install mediapipe
"""
# Author: Karam (Innovative Feature — D/HD)

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
from torch import nn

from torchvision import transforms


# ── MediaPipe landmark indices for left and right eye ──────────────────────
# Based on the 468-point FaceMesh topology
LEFT_EYE_LANDMARKS  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33,  160, 158,  133, 153, 144]

EAR_THRESHOLD     = 0.21   # raw EAR below this → eye likely closed
PERCLOS_THRESHOLD = 0.25   # >25 % closed frames in window → DROWSY
WINDOW_FRAMES     = 30     # rolling window size (≈1 second at 30 fps)

DEFAULT_MODEL_PATH = Path("models/fatigue_eye_karam.pth")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_cached_mp_face_mesh = None
_cached_cnn_model    = None
_cached_cnn_path: Path | None = None


class FatigueDetector:
    """Stateful per-session fatigue detector with PERCLOS tracking.

    Instantiate once and call ``update(face_image)`` on every frame.
    Maintains a rolling deque so PERCLOS is computed across real time.

    Args:
        model_path:        Path to trained eye-state CNN checkpoint.
        window_frames:     Number of frames in the PERCLOS window.
        perclos_threshold: Fraction of closed-eye frames to trigger DROWSY.
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

    def update(self, face_image: np.ndarray) -> dict[str, object]:
        """Process one frame and return the current fatigue assessment.

        Args:
            face_image: 224×224 RGB numpy array (same contract as other modules).

        Returns:
            Fatigue result dictionary (see module docstring).
        """
        ear, cnn_closed_prob = _analyse_frame(face_image, self._model_path)

        # Use CNN probability as primary signal; EAR as fallback
        if cnn_closed_prob is not None:
            eyes_closed = int(cnn_closed_prob > 0.5)
        else:
            eyes_closed = int(ear < EAR_THRESHOLD) if ear is not None else 0

        self._window.append(eyes_closed)
        perclos = sum(self._window) / max(len(self._window), 1)

        fatigue = "DROWSY" if perclos > self._perclos_threshold else "ALERT"

        return {
            "fatigue":    fatigue,
            "ear":        round(float(ear), 4) if ear is not None else None,
            "perclos":    round(perclos, 4),
            "confidence": round(float(cnn_closed_prob), 4) if cnn_closed_prob is not None else None,
        }

    def reset(self) -> None:
        """Clear the rolling window (call between different people)."""
        self._window.clear()


def predict_fatigue(
    face_image: np.ndarray,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, object]:
    """Single-frame fatigue prediction — no PERCLOS window.

    Useful for quick integration tests. For real-time use, prefer
    FatigueDetector so the PERCLOS window persists across frames.

    Args:
        face_image: 224×224 RGB numpy array.
        model_path: Path to trained eye-state CNN checkpoint.

    Returns:
        Fatigue result dictionary (perclos will be 0.0 or 1.0 for one frame).
    """
    detector = FatigueDetector(model_path=model_path, window_frames=1)
    return detector.update(face_image)


# ── Internal helpers ───────────────────────────────────────────────────────

def _analyse_frame(
    face_image: np.ndarray,
    model_path: Path,
) -> tuple[float | None, float | None]:
    """Return (mean_EAR, cnn_closed_probability) for one face image."""
    # 1. EAR via MediaPipe landmarks
    ear = _compute_ear(face_image)

    # 2. CNN eye-crop classification
    cnn_prob = _cnn_eye_closed_prob(face_image, model_path)

    return ear, cnn_prob


def _compute_ear(face_image: np.ndarray) -> float | None:
    """Compute mean Eye Aspect Ratio using MediaPipe FaceMesh."""
    global _cached_mp_face_mesh
    try:
        import mediapipe as mp
    except ImportError:
        return None   # mediapipe not installed — fall back to CNN only

    if _cached_mp_face_mesh is None:
        _cached_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    results = _cached_mp_face_mesh.process(face_image)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = face_image.shape[:2]

    def lm(idx):
        pt = landmarks[idx]
        return np.array([pt.x * w, pt.y * h])

    left_ear  = _ear_from_landmarks([lm(i) for i in LEFT_EYE_LANDMARKS])
    right_ear = _ear_from_landmarks([lm(i) for i in RIGHT_EYE_LANDMARKS])
    return (left_ear + right_ear) / 2.0


def _ear_from_landmarks(pts: list[np.ndarray]) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical   = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return float(vertical / max(horizontal, 1e-6))


def _cnn_eye_closed_prob(face_image: np.ndarray, model_path: Path) -> float | None:
    """Return the CNN's probability that the eyes are closed."""
    global _cached_cnn_model, _cached_cnn_path

    resolved = model_path.resolve()
    if not resolved.exists():
        return None   # model not trained yet

    if _cached_cnn_model is None or _cached_cnn_path != resolved:
        _cached_cnn_model = _load_eye_cnn(resolved)
        _cached_cnn_path  = resolved

    model  = _cached_cnn_model
    device = next(model.parameters()).device

    tensor = _PREPROCESS(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # class 0 = closed, class 1 = open  →  return P(closed)
    return float(probs[0])


def _load_eye_cnn(model_path: Path):
    """Load the trained EyeStateModel checkpoint."""
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # Rebuild the same architecture used in train_fatigue_karam.py
    backbone = models.efficientnet_b0(weights=None)
    in_feats = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    model = nn.Sequential(
        backbone,
        nn.Dropout(p=0.3),
        nn.Linear(in_feats, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 2),
    )

    # Wrap so state_dict keys match
    class _EyeStateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone   = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_feats, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
            )
        def forward(self, x):
            return self.classifier(self.backbone(x))

    m = _EyeStateModel()
    m.load_state_dict(checkpoint["model_state_dict"])
    m.to(device)
    m.eval()
    return m

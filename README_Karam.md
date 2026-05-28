# Karam's Modules — Classification Face Recognition, Emotion Detection & Fatigue Detection

> **Branch:** `feature/classification-emotion-karam`
> **Author:** Karam
> **Modules:** Supervised Learning Face Recognition · Emotion Detection · Fatigue Detection (D/HD Innovative Feature)

---

## Overview

This branch adds three modules to the attendance system:

| Module | Folder | Innovative? |
|--------|--------|-------------|
| Supervised Learning Face Recognition | `face_verification/classification_model/` | — |
| Emotion Detection | `emotion_detection/` | — |
| Fatigue & Drowsiness Detection | `fatigue_detection/` | ✅ D/HD Feature |

All three modules follow the same integration contract as the rest of the system — they accept a `224×224 RGB numpy array` from `detect_and_crop_face()` and return a plain dictionary.

---

## Files Added

```
face_verification/classification_model/
    __init__.py
    classification_model_karam.py       — ResNet34 model architecture
    train_classification_karam.py       — Training script (CrossEntropyLoss)
    build_gallery_karam.py              — Builds the face embedding gallery (.pkl)
    recognition_karam.py                — Inference API (drop-in for metric learning)
    evaluate_karam.py                   — ROC curve + AUC evaluation

emotion_detection/
    __init__.py
    emotion_model_karam.py              — EfficientNet-B0 model architecture
    train_emotion_karam.py              — Training script (FER-2013)
    emotion_karam.py                    — Inference API

fatigue_detection/
    __init__.py
    train_fatigue_karam.py              — Trains open/closed eye CNN
    fatigue_karam.py                    — Real-time fatigue inference API
```

---

## 1. Supervised Learning Face Recognition

### How it works

Treats face recognition as a classification problem. A ResNet34 backbone is fine-tuned with `CrossEntropyLoss` to classify faces into known identity classes. After training, the classification head is discarded and the backbone's 512-dim output is L2-normalised to produce a **face embedding** — directly comparable with Zhongyu's metric learning embeddings for the report.

### Dataset

Uses the same dataset as the metric learning module:

```
datasets/classification_data/
    train_data/
        id_001/   img1.jpg  img2.jpg  ...
        id_002/   ...
    val_data/
        id_001/   ...
```

> The `datasets/` folder is in `.gitignore` — download from the Kaggle link in the project spec and place it here.

### Step 1 — Train

```bash
python -m face_verification.classification_model.train_classification_karam \
    --data-dir datasets/classification_data/train_data \
    --val-dir  datasets/classification_data/val_data \
    --output   models/recognition_classification_karam.pth \
    --epochs   30
```

Saves the best checkpoint (by validation accuracy) to `models/recognition_classification_karam.pth`.

Optional flags:
```bash
--batch-size 32         # default 32
--learning-rate 1e-4    # default 1e-4
--train-backbone        # unfreeze full backbone (default: only last block)
```

### Step 2 — Build the gallery

After training, build the embedding gallery from the registered faces database:

```bash
python -m face_verification.classification_model.build_gallery_karam \
    --db-path datasets/faces_db \
    --model   models/recognition_classification_karam.pth \
    --output  models/recognition_gallery_classification_karam.pkl
```

Database layout:
```
datasets/faces_db/
    john_smith/    img1.jpg  img2.jpg  ...
    jane_doe/      ...
```

### Step 3 — Evaluate (ROC / AUC)

```bash
python -m face_verification.classification_model.evaluate_karam \
    --pairs   datasets/verification_pairs_val.txt \
    --model   models/recognition_classification_karam.pth \
    --gallery models/recognition_gallery_classification_karam.pkl
```

Prints the AUC score and saves the ROC curve to `models/roc_classification_karam.png`.

### Integration (UI)

```python
from face_verification.classification_model import predict_identity_classification

result = predict_identity_classification(face_image)

# Output:
# {
#     "identity":              "john_smith",   # or "UNKNOWN"
#     "similarity_score":      0.81,
#     "best_identity":         "john_smith",
#     "best_similarity_score": 0.81,
#     "distance_metric":       "cosine",
#     "method":                "classification"
# }
```

> Output format is **identical** to `predict_identity_metric()` from Zhongyu's module — either can be swapped into the UI without any other changes.

---

## 2. Emotion Detection

### How it works

An EfficientNet-B0 backbone fine-tuned on FER-2013 to classify 7 emotions:
`Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise`

EfficientNet-B0 is used here (vs ResNet34 in face recognition) so both architectures can be independently compared in the report.

### Dataset — FER-2013

1. Download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Extract and arrange into this layout:

```
datasets/fer2013/
    train/
        angry/     disgust/    fear/
        happy/     neutral/    sad/    surprise/
    val/
        angry/     ...
```

### Step 1 — Train

```bash
python -m emotion_detection.train_emotion_karam \
    --data-dir datasets/fer2013/train \
    --val-dir  datasets/fer2013/val \
    --output   models/emotion_karam.pth \
    --epochs   40
```

Optional flags:
```bash
--batch-size 64         # default 64
--learning-rate 1e-4    # default 1e-4
--train-backbone        # unfreeze full EfficientNet backbone
```

### Integration (UI)

```python
from emotion_detection import predict_emotion

result = predict_emotion(face_image)

# Output:
# {
#     "emotion":    "Happy",
#     "confidence": 0.92,
#     "all_scores": {
#         "Angry": 0.01, "Disgust": 0.00, "Fear": 0.01,
#         "Happy": 0.92, "Neutral": 0.04, "Sad": 0.01, "Surprise": 0.01
#     }
# }
```

---

## 3. Fatigue & Drowsiness Detection ✅ D/HD Innovative Feature

### How it works

Three complementary signals are combined:

1. **Eye Aspect Ratio (EAR)** — computed from MediaPipe FaceMesh landmarks. EAR drops sharply when eyes close.
2. **CNN eye-state classifier** — EfficientNet-B0 trained on eye crop images to classify each eye as open or closed. More robust than raw EAR under poor lighting or glasses.
3. **PERCLOS (sliding window)** — percentage of the last 30 frames (~1 second) where eyes are classified as closed. PERCLOS > 25% triggers `DROWSY`.

### Why it's relevant

Detects fatigued or drowsy employees at check-in — directly applicable to workplace safety monitoring. A spoofed printed photo has no blinking pattern, so PERCLOS also strengthens the liveness signal.

### Dataset — Eye State

Recommended: **MRL Eye Dataset** (http://mrl.cs.vsb.cz/eyedataset)

Organise as:
```
datasets/eye_state/
    train/
        open/      closed/
    val/
        open/      closed/
```

### Step 1 — Install extra dependency

```bash
pip install mediapipe
```

### Step 2 — Train eye-state CNN

```bash
python -m fatigue_detection.train_fatigue_karam \
    --data-dir datasets/eye_state/train \
    --val-dir  datasets/eye_state/val \
    --output   models/fatigue_eye_karam.pth \
    --epochs   25
```

### Integration (UI) — use `FatigueDetector` for real-time

Instantiate **once** per session so the PERCLOS window persists across frames:

```python
from fatigue_detection import FatigueDetector

detector = FatigueDetector(model_path="models/fatigue_eye_karam.pth")

# Inside the webcam loop — call every frame:
result = detector.update(face_image)

# Output:
# {
#     "fatigue":    "ALERT",    # or "DROWSY"
#     "ear":        0.29,       # mean Eye Aspect Ratio this frame
#     "perclos":    0.10,       # fraction of recent frames with closed eyes
#     "confidence": 0.08        # CNN probability that eyes are closed
# }

# Reset between different people:
detector.reset()
```

---

## Full System Integration Example

```python
from detection.detector import detect_and_crop_face
from face_verification.classification_model import predict_identity_classification
from anti_spoofing.liveness_zhongyu import predict_liveness
from emotion_detection import predict_emotion
from fatigue_detection import FatigueDetector

fatigue_detector = FatigueDetector()

# Inside webcam loop:
result = detect_and_crop_face(frame)

if "face_image" in result:
    face = result["face_image"]

    liveness  = predict_liveness(face)
    identity  = predict_identity_classification(face)
    emotion   = predict_emotion(face)
    fatigue   = fatigue_detector.update(face)

    print(identity["identity"])   # "john_smith" or "UNKNOWN"
    print(liveness["liveness"])   # "REAL" or "SPOOF"
    print(emotion["emotion"])     # "Happy"
    print(fatigue["fatigue"])     # "ALERT" or "DROWSY"
```

---

## Models Produced

| File | Description |
|------|-------------|
| `models/recognition_classification_karam.pth` | Trained face recognition checkpoint |
| `models/recognition_gallery_classification_karam.pkl` | Face embedding gallery |
| `models/roc_classification_karam.png` | ROC curve (evaluation output) |
| `models/emotion_karam.pth` | Trained emotion detection checkpoint |
| `models/fatigue_eye_karam.pth` | Trained eye-state CNN checkpoint |

> Model files are **not** committed to Git (too large). Train locally and keep them in your `models/` folder.

---

## Quick Reference — Run Order

```bash
# 1. Face recognition
python -m face_verification.classification_model.train_classification_karam
python -m face_verification.classification_model.build_gallery_karam
python -m face_verification.classification_model.evaluate_karam

# 2. Emotion detection
python -m emotion_detection.train_emotion_karam

# 3. Fatigue detection (innovative feature)
python -m fatigue_detection.train_fatigue_karam
```
# ml-facial-recognition-project

TODO - Clearly explain the following:
* how to install requirements
* where to place datasets
* how to run training scripts
* how to run the final system
* how registration works
* which models are used and whether they are fine-tuned

## 1. Face Detection Module (YOLOv11) - PATRICK (100599029)

### Setup and Installation
1. Ensure you have created and activated a Python 3.11 virtual environment (`.venv`).
2. Install the required dependencies by running: `pip install -r requirements.txt`

### Dataset Overview
The face detection model was trained using a custom dataset from Roboflow:
* **Link:** [Face Detection Dataset](https://universe.roboflow.com/haos-workspace-mnq0m/face-detection-0luto)
* **Total Images:** 16,094
* **Split:** 70% Train (11,266) | 20% Valid (3,219) | 10% Test (1,609)
* **Preprocessing:** Auto-Orient applied, Resized (Stretch to 640x640).
* **Augmentations:** None applied.

**Note:** To prevent clogging the repository, the main dataset has *not* been uploaded to GitHub. It has been added to the `.gitignore`. If you need to retrain the model, you can download it from the link above and place it in a root `datasets/` folder.

### Training the Model
The model is a fine-tuned YOLOv11 Nano architecture. It was trained locally for 30 epochs using the `detection/train_yolo.py` script. 

### Final Model & Live Testing
The finished, best-performing model from the 30 epochs is saved as `models/detection_yolo.pt`. It has been configured to strictly detect a single class (`face`). 

If you want to visually test the detection tracking on your own webcam, you can run: `python detection/test_webcam.py`

---

### Files Pushed & Integration Guide
Here is a breakdown of the files I have added/updated and how to use them for the group project:

* **`models/detection_yolo.pt`**
  The final trained AI weights. This is the only model file the system actually needs to run face detection.

* **`detection/detector.py` (USE THIS FOR INTEGRATION)**
  This is the silent API built for the rest of the team. It does not open a webcam or print to the terminal. You just import it, feed it an image frame, and it returns the exact dictionary format required by the Unified Guidelines. It contains two functions depending on what downstream data your module needs:  
  **1. Basic Face Detection (Bounding Box Only)**:

    ```python
    from detection.detector import detect_faces
    result = detect_faces(video_frame)

    # Success Output: 
    # {"bbox": [x1, y1, x2, y2], "confidence": 0.85}   

    # Error Output: 
    # {"status": "NO_FACE", "message": "No face detected"}
    ```
  **2. Full Preprocessing (For Liveliness/Recognition/Emotion):**
    ```python
    from detection.detector import detect_and_crop_face

    result = detect_and_crop_face(frame)
    # Success Output: 
    # {"bbox": [x1, y1, x2, y2], "confidence": 0.85, "face_image": <numpy.ndarray 224x224x3 RGB>}
    ```

* **`detection/test_webcam.py`**
  A live visual prototype that opens a webcam window and draws bounding boxes. This will serve as the foundation for our final GUI implementation. 

* **`detection/test_crop.py`**  
  A secondary testing script that shows the detect_and_crop_face API working. It opens your main webcam and a smaller window showing the isolated 224x224 RGB face feed exactly as the downstream modules will receive it. Run via `python -m detection.test_crop`.

* **`detection/train_yolo.py`**
  The script used to train the dataset. Kept purely as a historical record of the methodology.

* **`.gitignore` (Updated)**
  Updated to block the massive datasets folder, the runs history folder, Python cache files, and the base YOLO downloaded weights so we don't destroy our GitHub storage limit.

## Face Recognition / Verification Module - Zhongyu

### Metric Learning Pipeline
This branch prepares a metric-learning recognition pipeline. It consumes the same cropped face format produced by detection:

```python
face_image: numpy.ndarray
shape = (224, 224, 3)
format = RGB
```

The reusable API is:

```python
from face_verification.metric_learning.recognition_zhongyu import predict_identity_metric

result = predict_identity_metric(face_image)

# Output:
# {
#     "identity": "Andrew",
#     "similarity_score": 0.87,
#     "distance_metric": "cosine",
#     "method": "metric_learning"
# }
```

### Gallery Construction
Place registration images under:

```text
datasets/faces_db/
+-- Andrew/
+-- Daniel/
+-- Jamie/
```

Build the gallery:

```bash
python -m face_verification.metric_learning.build_gallery_zhongyu
```

The output is saved to:

```text
models/recognition_gallery_zhongyu.pkl
```

### Live Recognition Test
After building the gallery, run:

```bash
python -m face_verification.metric_learning.test_recognition_integration_zhongyu
```

The default recognition model is the trained Triplet ResNet34 checkpoint at
`models/recognition_triplet_resnet34_zhongyu.pth`, because it performed more
reliably in local webcam testing. EfficientNet-B0 remains available as the
lightweight comparison path:

```bash
python -m face_verification.metric_learning.build_gallery_zhongyu --architecture efficientnet_b0
python -m face_verification.metric_learning.test_recognition_integration_zhongyu --architecture efficientnet_b0
```

Train the Triplet EfficientNet-B0 model:

```bash
python -m face_verification.metric_learning.train_triplet_efficientnet_zhongyu --epochs 20 --batch-size 16
```

Train the Triplet ResNet34 model:

```bash
python -m face_verification.metric_learning.train_triplet_resnet34_zhongyu --epochs 20 --batch-size 16
```

Pretrained torchvision weights are cached locally under `models/torch_cache/`,
which is ignored by Git.

## Liveness Detection Module - Zhongyu

### Integration Contract
The liveness module follows the unified downstream interface. It does not run
face detection by itself. It expects a cropped face image from the shared
detection/preprocessing pipeline:

```python
face_image: numpy.ndarray
shape = (224, 224, 3)
format = RGB
```

Use the reusable API from `anti_spoofing/liveness_zhongyu.py`:

```python
from anti_spoofing.liveness_zhongyu import predict_liveness

result = predict_liveness(face_image)

# Output:
# {"liveness": "REAL", "confidence": 0.93}
# or
# {"liveness": "SPOOF", "confidence": 0.91}
```

The valid liveness labels are `REAL` and `SPOOF`. Recognition should only run
when this module returns `REAL`.

### Detection Integration
The detection module now exposes the shared crop API required by downstream
modules. Use `detect_and_crop_face(frame)` to obtain the standardized
`224x224 RGB` face image, then pass that directly to liveness:

```python
from detection.detector import detect_and_crop_face
from anti_spoofing.liveness_zhongyu import predict_liveness

result = detect_and_crop_face(frame)

if "face_image" in result:
    liveness_result = predict_liveness(result["face_image"])
```

`preprocess_face_from_bbox(...)` remains available inside the Zhongyu module as
a fallback helper, but the preferred integration path is the shared detection
API above.

### Model Plan
This branch prepares two alternative transfer-learning models for comparison:

* `ResNet50V2 + binary classification head`
* `DenseNet121 + binary classification head`

Expected dataset layout:

```text
datasets/liveness/
+-- train/
|   +-- spoof/
|   +-- real/
+-- val/
|   +-- spoof/
|   +-- real/
+-- test/
    +-- spoof/
    +-- real/
```

The images are expected to be already-cropped face images. They can be slightly
larger or smaller than `224x224`; the training script resizes them to `224x224`
before feeding them into the model. The entire `datasets/` directory is ignored
by Git through `.gitignore`.

Train ResNet50V2:

```bash
python -m anti_spoofing.train_liveness_zhongyu --architecture resnet50v2
```

Train DenseNet121:

```bash
python -m anti_spoofing.train_liveness_zhongyu --architecture densenet121
```

Saved model paths:

```text
models/liveness_resnet50v2_zhongyu.keras
models/liveness_densenet121_zhongyu.keras
```

For a quick local webcam smoke test after a model is trained:

```bash
python -m anti_spoofing.test_liveness_integration_zhongyu
```

Current webcam calibration uses DenseNet121 as the default liveness model with
a REAL threshold of `0.43`. In local testing, DenseNet121 separated real faces
more clearly than ResNet50V2; ResNet50V2 is kept for model comparison.

## Supervised Learning (Classification-Based) Face Verification - Patrick (100599029)

### Overview and Methodology
This branch implements a supervised learning approach to Face Verification. The model was trained as a multi-class classifier to identify 4,000 distinct individuals based on their unique folder identities. 

Once the network learned to accurately map facial features to these specific identities, the final classification layer was bypassed. The network now acts as a feature extractor, outputting a highly discriminative `1024-dimensional` numerical face embedding. Two faces can then be reliably compared by calculating the Cosine Similarity between their respective embeddings to determine if they match.

### Dataset Overview & Training
The dataset used for this module is sourced from the [11-785 Fall 20 Face Verification Kaggle Competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). 

* **Training Data (`classification_data/train_data/`):** Contains 4,000 unique individual ID folders. Used to train the classification model to recognise and distinguish between these specific people.
* **Verification Data (`verification_data/`):** Unlabelled face images used for testing the system's ultimate ability to verify identities.
* **Evaluation Pairs (`verification_pairs_val.txt`):** A list of verification trials containing two image paths and a ground-truth label (`1` for the same person, `0` for different people). Used exclusively to compute the final AUC score.
* **Architecture:** MobileNetV2 (Pre-trained on ImageNet). Base convolutional layers were frozen, and a custom classification head featuring Dropout (0.5) and an intermediate 1024-node linear layer was appended.
* **Process:** Trained locally over 10 epochs using Cross-Entropy Loss and Automatic Mixed Precision (AMP).
* **Note:** To preserve GitHub storage limits, the training datasets are intentionally ignored via `.gitignore`. 

### System Evaluation & ROC Comparison
To fulfil the assignment requirements of comparing different distance metrics and baseline model architectures, two evaluation scripts were created. These scripts process the Kaggle verification pairs to generate True Positive (TPR) and False Positive (FPR) rates across various threshold settings.

* **`face_verification/classification_model/evaluate_verification_patrick.py`**
  Evaluates the MobileNetV2 Supervised model. It calculates both Euclidean Distance and Cosine Similarity to determine the superior distance metric. Cosine Similarity performed noticeably better (AUC: 0.7522). This generated the evaluation graph saved at `reports/Patrick_MobileNetV2_ROC_Curve.png`.
  
  **To reproduce this evaluation:**
  ```bash
  python face_verification/classification_model/evaluate_verification_patrick.py
  ```

* **`zongyu_test.py`**
  A unified testing environment built to run the alternate Metric Learning (ResNet34) model through the exact same verification pipeline. This allowed for a strict 1:1 scientific comparison between the two approaches, generating the graph saved at `reports/Zhongyu_ResNet34_ROC_Curve.png`.
  
  **To reproduce this evaluation:**
  ```bash
  python zongyu_test.py
  ```
  
  **Both code examples above assume you are in the root directory of the folder, and each component are in their correct folders**

### Files Pushed & Integration Guide
Here is a breakdown of the operational files added and how to deploy them for the frontend live system:

* **`models/verification_classification_patrick.pt`**
  The final trained AI weights. This is the mathematical core required for the system to extract facial structures.

* **`face_verification/classification_model/train_classifier_patrick.py`**
  The original script used to train the network on the Kaggle `train_data`. Kept as a record of the methodology and class blueprints.

* **`face_verification/classification_model/face_comparator_patrick.py` (USE THIS FOR FRONTEND INTEGRATION)**
  This is the dedicated integration API built for the UI. It acts as a wrapper that automatically builds the network blueprint, loads the `.pt` weights, safely strips the classification layer, and handles all the complex vector math behind the scenes.

**Implementation Examples:**
The `face_comparator_patrick.py` module expects the standardised cropped face image directly from the detection pipeline (`224x224 RGB numpy.ndarray`).

1. **Initialising the System:**
```python
# Import the verification class from the module.
from face_verification.classification_model.face_comparator_patrick import PatrickFaceVerifier

# Initialise the verification system.
# This automatically loads the neural network architecture and the trained weights.
face_verification_system = PatrickFaceVerifier()
```

2. **Registration (Saving a new face):**
```python
# Assuming 'new_face_image' is the standardised 224x224 RGB image array.
# Generate the 1024-dimensional face embedding for the new identity.
face_embedding = face_verification_system.generate_face_embedding(new_face_image)

# Save this mathematical embedding to the system's database.
# Example: saved_faces_database["Subject_A"] = face_embedding
```

3. **Live Verification:**
```python
# Assuming 'live_camera_image' is the 224x224 RGB image array currently on screen.
live_embedding = face_verification_system.generate_face_embedding(live_camera_image)

# Compare the live face against a saved face in the database.
verification_result = face_verification_system.compare_embeddings(
    live_webcam_embedding=live_embedding, 
    saved_database_embedding=saved_faces_database["Subject_A"]
)

# Handle the result
if verification_result["match"]:
    print(f"Identity Verified. Score: {verification_result['similarity_score']}")
else:
    print("Match Failed.")
```
**Note for UI Integration**: The `compare_embeddings` function defaults to a match threshold of 0.65. If you find the system is letting imposters through durig live testing, you can tighten the security by passing a higher number (e.g., `approval_threshold=0.75`).
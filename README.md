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

## 2. Liveness Detection Module - Zhongyu

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

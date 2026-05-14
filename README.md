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
  This is the silent API built for the rest of the team. It does not open a webcam or print to the terminal. You just import it, feed it an image frame, and it returns the exact dictionary format required by the Unified Guidelines.  
  Example usage:

    ```python
    from detection.detector import detect_faces
    #Example usage:
    result = detect_faces(video_frame)

    # Success Output: 
    # {"bbox": [x1, y1, x2, y2], "confidence": 0.85}   

    # Error Output: 
    # {"status": "NO_FACE", "message": "No face detected"}
    ```

* **`detection/test_webcam.py`**
  A live visual prototype that opens a webcam window and draws bounding boxes. This will serve as the foundation for our final GUI implementation. 

* **`detection/train_yolo.py`**
  The script used to train the dataset. Kept purely as a historical record of the methodology.

* **`.gitignore` (Updated)**
  Updated to block the massive datasets folder, the runs history folder, Python cache files, and the base YOLO downloaded weights so we don't destroy our GitHub storage limit.
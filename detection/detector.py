import cv2
from ultralytics import YOLO
# Author: Patrick (100599029)

# Load the trained face detection model.
detection_model = YOLO("models/detection_yolo.pt")

def detect_faces(frame):
    # Execute the model on the input frame with a confidence threshold of 50 percent and disable terminal output.
    results = detection_model(frame, conf=0.5, verbose=False)
    
    # Check if the model detected zero bounding boxes.
    if len(results[0].boxes) == 0:
        # Return the error dictionary.
        return {
            "status": "NO_FACE",
            "message": "No face detected"
        }
    
    # Isolate the data for the first detected face.
    face_data = results[0].boxes[0]
    
    # Extract the raw bounding box coordinates into a list.
    coordinates = face_data.xyxy[0].tolist()
    
    # Convert the floating point coordinates into integers.
    x1 = int(coordinates[0])
    y1 = int(coordinates[1])
    x2 = int(coordinates[2])
    y2 = int(coordinates[3])
    
    # Extract the confidence score as a floating point number.
    raw_confidence = float(face_data.conf[0])
    
    # Round the confidence score to two decimal places.
    final_confidence = round(raw_confidence, 2)
    
    # Return the coordinates and confidence score in the dictionary format.
    return {
        "bbox": [x1, y1, x2, y2],
        "confidence": final_confidence
    }

def detect_and_crop_face(frame):
    # Call the base detection function to get the coordinates and confidence.
    detection_result = detect_faces(frame)
    
    # Check if the base function returned a NO_FACE error and return it immediately.
    if "status" in detection_result and detection_result["status"] == "NO_FACE":
        return detection_result
        
    # Extract the coordinates from the returned dictionary.
    bounding_box = detection_result["bbox"]
    x1 = bounding_box[0]
    y1 = bounding_box[1]
    x2 = bounding_box[2]
    y2 = bounding_box[3]
    
    # Crop the face from the original frame using numpy array slicing.
    cropped_face = frame[y1:y2, x1:x2]
    
    # Verify the crop was successful and the array is not empty.
    if cropped_face.size == 0:
         return {
            "status": "CROP_FAILED",
            "message": "Face detected, but cropping failed due to boundary issues."
        }
        
    # Convert the color format from BGR to RGB.
    rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    detection_result["raw_face_image"] = rgb_face
    
    # Resize the cropped image to the unified standard of 224x224 pixels.
    standardized_face = cv2.resize(rgb_face, (224, 224))
    
    # Add the preprocessed face image to the existing dictionary.
    detection_result["face_image"] = standardized_face
    
    # Return the updated dictionary containing coordinates, confidence, and the image.
    return detection_result
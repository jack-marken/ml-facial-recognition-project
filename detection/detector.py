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
import cv2
from ultralytics import YOLO
# Author: Patrick (100599029)

# Load the custom trained YOLO model from the models folder.
custom_face_detection_model = YOLO("models/detection_yolo.pt")

# Initialise the video capture object to use the primary default webcam.
live_webcam_feed = cv2.VideoCapture(0)

print("Webcam initialised. Press 'q' in the video window to quit.")

# Begin an infinite loop to process the webcam feed frame by frame.
while live_webcam_feed.isOpened():
    
    # Read the current frame from the webcam.
    successful_read, current_video_frame = live_webcam_feed.read()
    
    # Break the loop if the webcam stops sending frames.
    if not successful_read:
        break

    # Execute the model to detect faces silently in the current frame.
    model_prediction_results = custom_face_detection_model(current_video_frame, conf=0.5, verbose=False)

    # Generate a new image frame that includes the drawn bounding boxes and labels.
    frame_with_drawn_bounding_boxes = model_prediction_results[0].plot()

    # Display the annotated frame in a new desktop window.
    cv2.imshow("Face Detection Live Test", frame_with_drawn_bounding_boxes)

    # Check if the user presses the 'q' key to terminate the loop.
    keyboard_input = cv2.waitKey(1)
    if keyboard_input & 0xFF == ord('q'):
        break

# Release the webcam hardware and close all created graphical windows.
live_webcam_feed.release()
cv2.destroyAllWindows()
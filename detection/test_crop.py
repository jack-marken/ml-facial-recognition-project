import cv2
from detection.detector import detect_and_crop_face

# Open the default webcam
webcam = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while True:
    successful_read, current_frame = webcam.read()
    if not successful_read:
        break

    # Pass the frame to your new API function
    api_result = detect_and_crop_face(current_frame)

    # Check if a face was actually found
    if "status" not in api_result:
        # Extract the cropped image from the dictionary
        ai_ready_face = api_result["face_image"]
        
        # OpenCV displays images in BGR format, so we briefly convert it back just to view it correctly on screen
        display_face = cv2.cvtColor(ai_ready_face, cv2.COLOR_RGB2BGR)
        
        # Show the tiny 224x224 cropped face in its own window
        cv2.imshow("AI Cropped Face (224x224)", display_face)

    # Show the main webcam feed in another window
    cv2.imshow("Main Camera Feed", current_frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
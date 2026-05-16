import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
from detection.detector import detect_faces
# Authors: Jack (105417647), Patrick (100599029)

class UserInterface:
    """Main user interface for the face recognition attendance system

    Usage:
        app = UserInterface()
        app.start()
    """

    def register_employee(self, cropped_img):
        """Tkinter pop-up window for registering a new employee

        Displays the employee's identified face, and asks for their first name and last name.
        Upon submission, their face is saved to the dataset.

        Args:
            cropped_img: a NumPy array representing the part of the image inside a bounding box identified by the custom_face_detection_model.
        """

        def save_img():
            first_name = fn_entry.get().lower()
            last_name = ln_entry.get().lower()

            # Make directory to store new employee face data
            target_dir = Path(f"datasets/new_employees/{first_name}_{last_name}")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Title the image '0.jpg'. If it exists, increment to '1.jpg', and so on.
            counter = 0
            while target_dir.joinpath(f"{counter}.jpg").is_file():
                counter += 1

            # Write the file to the employee directory. Example: 'datasets/jack_marken/0.jpg'
            filename = target_dir.joinpath(f"{counter}.jpg")
            cv2.imwrite(filename, cropped_img)
            print(f"Saved: {filename}")
            root.destroy()

        # main application window
        root = tk.Tk()
        root.title("Register")

        # Convert NumPy array to PIL image format
        pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

        # Convert PIL image format to Tkinter PhotoImage format
        img_tk = ImageTk.PhotoImage(image=pil_img)
        label = ttk.Label(root, image=img_tk)
        label.pack()
        label.image = img_tk # For memory cleanup

        # First and last name entry boxes
        fn_entry = ttk.Entry(root, width=30)
        fn_entry.pack(pady=20, padx=20)

        ln_entry = ttk.Entry(root, width=30)
        ln_entry.pack(pady=20)

        # create button to display the content of entry widget
        button = ttk.Button(root, text="Print Text", 
                        command=save_img)
        button.pack(pady=10)

        root.mainloop()

    def video_capture(self):
        """Live video capture with face detection
        TODO: Implement face recognition 
        """
        # Load the custom trained YOLO model from the models folder.
        custom_face_detection_model = YOLO("models/detection_yolo.pt")

        # Initialise the video capture object to use the primary default webcam.
        live_webcam_feed = cv2.VideoCapture(0)

        print("Webcam initialised. Press 'q' in the video window to quit.")

        print("Webcam initialised. Press 'Enter' in the video window to register a new face.")

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
            # Check if the user presses the 'Enter' key to register a new employee
            elif keyboard_input & 0xFF == 13:
                if model_prediction_results[0] is not None:
                    # Initiate a pop-up window for each detected face on the screen
                    for i, box in enumerate(model_prediction_results[0].boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box[:4])
                        cropped_img = current_video_frame[y1:y2, x1:x2]
                        self.register_employee(cropped_img)

        # Release the webcam hardware and close all created graphical windows.
        live_webcam_feed.release()
        cv2.destroyAllWindows()

    def start(self):
        self.video_capture()
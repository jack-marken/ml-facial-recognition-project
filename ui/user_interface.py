import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageOps
from pathlib import Path

# APIs from the local project files
from detection.detector import detect_and_crop_face
from face_verification.metric_learning.recognition_kaixiang import predict_identity
from face_verification.metric_learning.build_gallery_kaixiang import main as build_identity_gallery
from anti_spoofing.liveness_kaixiang import predict_liveness

# ==============================================================================================
# --- SUPERVISED CLASSIFICATION MODEL INTEGRATION (PATRICK LUNNEY) ---
import os
from face_verification.classification_model.face_comparator_patrick import PatrickFaceVerifier
# ==============================================================================

# Authors: Jack (105417647), Patrick (100599029)

class UserInterface:
    """Main user interface for the face recognition attendance system

    Usage:
        app = UserInterface()
        app.start()
    """

    def register_employee(self, cropped_img, standardized_img):
        """Tkinter pop-up window for registering a new employee

        Displays the employee's identified face, and asks for their first name and last name.
        Upon submission, their face is saved to the dataset.

        Args:
            cropped_img: a NumPy array representing the part of the image inside a bounding box identified by the custom_face_detection_model.
            standardized_img: The array in cropped_img, resized to fit the unified standard of 224x224 pixels.
        """

        def save_img():
            first_name = fn_entry.get().lower()
            last_name = ln_entry.get().lower()

            # Make directory to store new employee face data
            target_dir = Path(f"datasets/faces_db/{first_name}_{last_name}")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Title the image '0.jpg'. If it exists, increment to '1.jpg', and so on.
            counter = 0
            while target_dir.joinpath(f"{counter}.jpg").is_file():
                counter += 1

            # Write the file to the employee directory. Example: 'datasets/jack_marken/0.jpg'
            filename = target_dir.joinpath(f"{counter}.jpg")

            rgb_face = cv2.cvtColor(standardized_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, rgb_face)
            print(f"Saved: {filename}")
            build_identity_gallery()
            root.destroy()

        # main application window
        root = tk.Tk()
        root.title("Register")

        # Convert NumPy array to PIL image format
        pil_img = Image.fromarray(cropped_img)
        max_size = (300, 300)
        resized_img = ImageOps.contain(pil_img, max_size)

        # Convert PIL image format to Tkinter PhotoImage format
        img_tk = ImageTk.PhotoImage(image=resized_img)
        label = ttk.Label(root, image=img_tk)
        label.pack()
        label.image = img_tk # For memory cleanup

        # First and last name entry boxes
        tk.Label(root, text="First name:", padx=20).pack()
        fn_entry = ttk.Entry(root, width=30)
        fn_entry.pack(pady=20, padx=20)

        tk.Label(root, text="Last: name:", padx=20).pack()
        ln_entry = ttk.Entry(root, width=30)
        ln_entry.pack(pady=20)

        # create button to display the content of entry widget
        button = ttk.Button(root, text="Submit", 
                        command=save_img)
        button.pack(pady=10)

        root.mainloop()

    def video_capture(self):
        """Live video capture with face detection"""
        # Load the custom trained YOLO model from the models folder.
        custom_face_detection_model = YOLO("models/detection_yolo.pt")

        # Initialise the video capture object to use the primary default webcam.
        live_webcam_feed = cv2.VideoCapture(0)

        # Request a resolution of 1280x720 pixels from the webcam hardware.
        live_webcam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        live_webcam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ==============================================================================
        # --- SUPERVISED CLASSIFICATION MODEL INTEGRATION (PATRICK LUNNEY) ---
        print("Initialising the custom MobileNetV2 classification verifier.")
        classification_verifier = PatrickFaceVerifier(trained_model_weights_path="models/verification_classification_patrick.pt")

        print("Building the custom embedding gallery in system memory.")
        custom_identity_gallery = {}
        dataset_directory_path = "datasets/faces_db"

        # Check if the dataset directory exists before attempting to load images.
        if os.path.exists(dataset_directory_path):
            # Iterate through each folder within the dataset directory.
            for identity_folder_name in os.listdir(dataset_directory_path):
                specific_folder_path = os.path.join(dataset_directory_path, identity_folder_name)
                
                # Verify the path is a directory.
                if os.path.isdir(specific_folder_path):
                    # Construct the path to the first image file.
                    first_image_file_path = os.path.join(specific_folder_path, "0.jpg")
                    
                    # Verify the image file exists.
                    if os.path.exists(first_image_file_path):
                        # Load the image using OpenCV to match the live webcam format.
                        database_image_array = cv2.imread(first_image_file_path)
                        
                        # Resize the image array to the required 224x224 dimensions.
                        resized_database_image = cv2.resize(database_image_array, (224, 224))
                        
                        # Generate the 1024-feature embedding and store it in the dictionary.
                        custom_identity_gallery[identity_folder_name] = classification_verifier.generate_face_embedding(resized_database_image)

        print(f"Custom gallery built successfully. Total loaded identities: {len(custom_identity_gallery)}")
        # ==============================================================================

        print("Webcam initialised. Press 'q' in the video window to quit.")

        print("Webcam initialised. Press 'Enter' in the video window to register a new face.")

        # Begin an infinite loop to process the webcam feed frame by frame.
        while live_webcam_feed.isOpened():
            
            # Read the current frame from the webcam.
            successful_read, current_video_frame = live_webcam_feed.read()
            
            # Break the loop if the webcam stops sending frames.
            if not successful_read:
                break

            # Use the detector API to extract a cropped image from a detected bounding box
            detection_results = detect_and_crop_face(current_video_frame)

            # Draw bounding box
            if "face_image" in detection_results:
                x1, y1, x2, y2 = detection_results["bbox"]
                
                # If the anti-spoofing module determines that the face is not real, the bounding box will show grey
                liveness_result = predict_liveness(detection_results["face_image"], checkpoint_path="models/liveness_efficientnetb0_kaixiang_final1_best.pth", threshold=0.81)
                label = ""
                color = (150, 150, 150)
                if liveness_result["liveness"] == "REAL":
                    # --- KAIXIANG'S ORIGINAL CODE (COMMENTED OUT) ---
                    # classification_result = predict_identity(
                    #     detection_results["face_image"],
                    #     gallery_path="models/recognition_gallery_kaixiang.pkl",
                    #     model_path="models/recognition_triplet_resnet18_kaixiang_final30b_best.pth",
                    #     distance_threshold=0.006,
                    #     distance_margin=0.0003,
                    #     matching_mode="mean",
                    #     )
                    
                    # ==============================================================================
                    # --- SUPERVISED CLASSIFICATION MODEL INTEGRATION (PATRICK LUNNEY) ---
                    # Extract the 1024-feature embedding from the live webcam face crop.
                    live_webcam_embedding = classification_verifier.generate_face_embedding(detection_results["face_image"])
                    
                    # Initialise tracking variables to find the best match in the gallery.
                    highest_similarity_score = 0.0
                    best_matching_identity_name = "[unknown]"
                    
                    # Iterate through every identity in the loaded custom gallery.
                    for stored_identity_name, stored_database_embedding in custom_identity_gallery.items():
                        # Compare the live embedding against the stored embedding.
                        comparison_output_dictionary = classification_verifier.compare_embeddings(
                            live_webcam_embedding=live_webcam_embedding, 
                            saved_database_embedding=stored_database_embedding
                        )
                        
                        # Extract the numerical similarity score.
                        current_comparison_score = comparison_output_dictionary["similarity_score"]
                        
                        # Update the tracking variables if the current score is the highest found.
                        if current_comparison_score > highest_similarity_score:
                            highest_similarity_score = current_comparison_score
                            best_matching_identity_name = stored_identity_name
                    
                    # Structure the output dictionary to match the expected format of the downstream user interface.
                    classification_result = {
                        "best_identity": best_matching_identity_name,
                        "similarity_score": highest_similarity_score
                    }
                    # ==============================================================================

                    color = (0, 255, 0)
                    if classification_result["similarity_score"] < 0.88:
                        label = "[unknown]"
                    else:
                        label = f"{classification_result['best_identity']} {classification_result['similarity_score']}" 

                cv2.rectangle(current_video_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    current_video_frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # Display a new image frame in a new desktop window that includes the drawn bounding boxes and labels.
            cv2.imshow("Face Detection Live Test", current_video_frame)

            # Check if the user presses the 'q' key to terminate the loop.
            keyboard_input = cv2.waitKey(1)
            if keyboard_input & 0xFF == ord('q'):
                break
            # Check if the user presses the 'Enter' key to register a new employee
            elif keyboard_input & 0xFF == 13:
                if "face_image" in detection_results:
                    if liveness_result["liveness"] == "REAL":
                        # Open a window to register the cropped face in the employee database
                        self.register_employee(detection_results["raw_face_image"], detection_results["face_image"])
                    else:
                        messagebox.showinfo("Anti-spoofing verification", "Face was not clear enough to be verified. Please ensure that your face is fully shown.")

        # Release the webcam hardware and close all created graphical windows.
        live_webcam_feed.release()
        cv2.destroyAllWindows()

    def start(self):
        self.video_capture()
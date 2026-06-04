import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageOps
from pathlib import Path
import os

# APIs from the local project files
from detection.detector import detect_and_crop_face
from emotion_detection.emotion_detector import EmotionDetector
from face_verification.metric_learning.recognition_kaixiang import predict_identity
from face_verification.metric_learning.build_gallery_kaixiang import main as build_identity_gallery
from anti_spoofing.liveness_kaixiang import predict_liveness

# ==============================================================================
# --- PATRICK LUNNEY: INDIVIDUAL FEATURES ---
from spatial_tracking_hd_patrick.spatial_tracker_hd_patrick import SpatialAttendanceTracker
from face_verification.classification_model.face_comparator_patrick import PatrickFaceVerifier
# ==============================================================================

# Authors: Jack (105417647), Patrick (100599029)

class UserInterface:
    """Main user interface for the face recognition attendance system"""
    def __init__(self):
        self.emotion_detector = EmotionDetector()

    def register_employee(self, cropped_img, standardized_img):
        """Tkinter pop-up window for registering a new employee"""
        def save_img():
            first_name = fn_entry.get().lower()
            last_name = ln_entry.get().lower()

            target_dir = Path(f"datasets/faces_db/{first_name}_{last_name}")
            target_dir.mkdir(parents=True, exist_ok=True)
            
            counter = 0
            while target_dir.joinpath(f"{counter}.jpg").is_file():
                counter += 1

            filename = target_dir.joinpath(f"{counter}.jpg")

            rgb_face = cv2.cvtColor(standardized_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, rgb_face)
            print(f"Saved: {filename}")
            build_identity_gallery()
            root.destroy()

        root = tk.Tk()
        root.title("Register")

        pil_img = Image.fromarray(cropped_img)
        max_size = (300, 300)
        resized_img = ImageOps.contain(pil_img, max_size)

        img_tk = ImageTk.PhotoImage(image=resized_img)
        label = ttk.Label(root, image=img_tk)
        label.pack()
        label.image = img_tk 

        tk.Label(root, text="First name:", padx=20).pack()
        fn_entry = ttk.Entry(root, width=30)
        fn_entry.pack(pady=20, padx=20)

        tk.Label(root, text="Last name:", padx=20).pack()
        ln_entry = ttk.Entry(root, width=30)
        ln_entry.pack(pady=20)

        button = ttk.Button(root, text="Submit", command=save_img)
        button.pack(pady=10)

        root.mainloop()

    def video_capture(self):
        """Live video capture with face detection"""
        custom_face_detection_model = YOLO("models/detection_yolo.pt")
        live_webcam_feed = cv2.VideoCapture(0)

        # ==============================================================================
        # --- HD SPATIAL TRACKING INIT ---
        camera_frame_width = int(live_webcam_feed.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera_frame_height = int(live_webcam_feed.get(cv2.CAP_PROP_FRAME_HEIGHT))
        attendance_tracker = SpatialAttendanceTracker(camera_frame_width, camera_frame_height)
        
        # ==============================================================================
        # [SWAP STEP 1 OF 2] : INITIALIZATION SWAP
        # ==============================================================================
        # ---> TO RUN MOBILENET: UNCOMMENT THIS ENTIRE BLOCK <---
        # print("Initialising the custom MobileNetV2 classification verifier.")
        # classification_verifier = PatrickFaceVerifier(trained_model_weights_path="models/verification_classification_patrick.pt")
        # print("Building the custom embedding gallery in system memory.")
        # custom_identity_gallery = {}
        # dataset_directory_path = "datasets/faces_db"
        # if os.path.exists(dataset_directory_path):
        #     for identity_folder_name in os.listdir(dataset_directory_path):
        #         specific_folder_path = os.path.join(dataset_directory_path, identity_folder_name)
        #         if os.path.isdir(specific_folder_path):
        #             first_image_file_path = os.path.join(specific_folder_path, "0.jpg")
        #             if os.path.exists(first_image_file_path):
        #                 database_image_array = cv2.imread(first_image_file_path)
        #                 resized_database_image = cv2.resize(database_image_array, (224, 224))
        #                 custom_identity_gallery[identity_folder_name] = classification_verifier.generate_face_embedding(resized_database_image)
        # print(f"Custom gallery built successfully. Total loaded identities: {len(custom_identity_gallery)}")
        # ==============================================================================

        print("Webcam initialised. Press 'q' in the video window to quit.")
        print("Webcam initialised. Press 'Enter' in the video window to register a new face.")

        while live_webcam_feed.isOpened():
            
            successful_read, current_video_frame = live_webcam_feed.read()
            if not successful_read:
                break

            detection_results = detect_and_crop_face(current_video_frame)

            if "face_image" in detection_results:
                x1, y1, x2, y2 = detection_results["bbox"]
                
                liveness_result = predict_liveness(detection_results["face_image"], checkpoint_path="models/liveness_efficientnetb0_kaixiang_final1_best.pth", threshold=0.81)
                label = ""
                color = (150, 150, 150)
                
                if liveness_result["liveness"] == "REAL":
                    
                    # ==============================================================================
                    # [SWAP STEP 2 OF 2] : LIVE DEMO MODEL SWAP
                    # ==============================================================================
                    
                    # ---> MODEL 1: GROUP'S FINAL RESNET (ACTIVE BY DEFAULT) <---
                    # TO DISABLE: Highlight this block and comment it out.
                    classification_result = predict_identity(
                        detection_results["face_image"],
                        gallery_path="models/recognition_gallery_kaixiang.pkl",
                        model_path="models/recognition_triplet_resnet18_kaixiang_final30b_best.pth",
                        distance_threshold=0.006,
                        distance_margin=0.0003,
                        matching_mode="mean",
                    )

                    # ---> MODEL 2: MOBILENETV2 BASELINE (DISABLED BY DEFAULT) <---
                    # TO ACTIVATE: Highlight this block and uncomment it.
                    # live_webcam_embedding = classification_verifier.generate_face_embedding(detection_results["face_image"])
                    # highest_similarity_score = 0.0
                    # best_matching_identity_name = "[unknown]"
                    # for stored_identity_name, stored_database_embedding in custom_identity_gallery.items():
                    #     comparison_output_dictionary = classification_verifier.compare_embeddings(
                    #         live_webcam_embedding=live_webcam_embedding, 
                    #         saved_database_embedding=stored_database_embedding
                    #     )
                    #     current_comparison_score = comparison_output_dictionary["similarity_score"]
                    #     if current_comparison_score > highest_similarity_score:
                    #         highest_similarity_score = current_comparison_score
                    #         best_matching_identity_name = stored_identity_name
                    # classification_result = {
                    #     "best_identity": best_matching_identity_name,
                    #     "similarity_score": highest_similarity_score
                    # }
                    
                    # ==============================================================================

                    color = (0, 255, 0)
                    if classification_result["similarity_score"] < 0.88:
                        label = "[unknown]"
                    else:
                        label = f"{classification_result['best_identity']} {classification_result['similarity_score']}" 
                    
                    # --- EMOTION DETECTION ---
                    if self.emotion_detector.active == True:
                        emotion_result = "Emotion: " + self.emotion_detector.detect_emotion(detection_results["raw_face_image"])
                        cv2.putText(current_video_frame, emotion_result, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # ==============================================================================
                    # --- HD SPATIAL TRACKING ---
                    tracking_result_tuple = attendance_tracker.log_person(
                        identity_name=classification_result['best_identity'],
                        bounding_box_coordinates=[x1, y1, x2, y2]
                    )
                    
                    current_spatial_zone = tracking_result_tuple[1]
                    label = f"{label}\n[{current_spatial_zone}]"
                    # ==============================================================================

                cv2.rectangle(current_video_frame, (x1, y1), (x2, y2), color, 2)
                
                label_lines = label.split('\n')
                y_offset = max(20, y1 - 10) - ((len(label_lines) - 1) * 20)
                
                for line in label_lines:
                    cv2.putText(
                        current_video_frame,
                        line,
                        (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                    y_offset += 25
            
            # ==============================================================================
            # --- HD SPATIAL TRACKING GRID ---
            cv2.line(current_video_frame, (int(camera_frame_width / 3), 0), (int(camera_frame_width / 3), camera_frame_height), (200, 200, 200), 1)
            cv2.line(current_video_frame, (int(camera_frame_width / 3 * 2), 0), (int(camera_frame_width / 3 * 2), camera_frame_height), (200, 200, 200), 1)
            cv2.line(current_video_frame, (0, int(camera_frame_height / 3)), (camera_frame_width, int(camera_frame_height / 3)), (200, 200, 200), 1)
            cv2.line(current_video_frame, (0, int(camera_frame_height / 3 * 2)), (camera_frame_width, int(camera_frame_height / 3 * 2)), (200, 200, 200), 1)
            # ==============================================================================

            cv2.imshow("Face Detection Live Test", current_video_frame)

            keyboard_input = cv2.waitKey(1)
            if keyboard_input & 0xFF == ord('q'):
                break
            elif keyboard_input & 0xFF == 13:
                if "face_image" in detection_results:
                    if liveness_result["liveness"] == "REAL":
                        self.register_employee(detection_results["raw_face_image"], detection_results["face_image"])
                    else:
                        messagebox.showinfo("Anti-spoofing verification", "Face was not clear enough to be verified. Please ensure that your face is fully shown.")
            elif keyboard_input & 0xFF == ord('e'):
                self.emotion_detector.toggle_active()

        # ==============================================================================
        # --- HD SPATIAL TRACKING REPORTING ---
        attendance_tracker.generate_reports()
        # ==============================================================================

        live_webcam_feed.release()
        cv2.destroyAllWindows()

    def start(self):
        self.video_capture()
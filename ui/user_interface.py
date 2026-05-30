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
from anti_spoofing.temporal_liveness_zhongyu import (
    DEFAULT_TEMPORAL_MODEL_PATH,
    DEFAULT_TEMPORAL_THRESHOLD,
    TemporalLivenessDetector,
)

# Authors: Jack (105417647), Patrick (100599029)

LIVENESS_MODEL_PATH = DEFAULT_TEMPORAL_MODEL_PATH
LIVENESS_THRESHOLD = DEFAULT_TEMPORAL_THRESHOLD
LIVENESS_SMOOTH_WINDOW = 15
LIVENESS_MIN_FRAMES = 5
LIVENESS_CONFIRMATIONS = 3
LIVENESS_MAX_STABLE_STD = 0.18
LIVENESS_PROCESS_EVERY = 8
LIVENESS_PROGRESS_BLOCKS = 5
LIVENESS_MIN_FACE_HEIGHT_RATIO = 0.25
LIVENESS_MIN_FACE_WIDTH_RATIO = 0.16
LIVENESS_RESET_IOU_THRESHOLD = 0.35
LIVENESS_RESET_CENTER_SHIFT = 0.25
LIVENESS_RESET_AREA_CHANGE = 0.45


def is_face_close_enough(bbox, frame_shape, min_height_ratio, min_width_ratio):
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape[:2]
    face_height_ratio = (y2 - y1) / max(frame_height, 1)
    face_width_ratio = (x2 - x1) / max(frame_width, 1)
    return (
        face_height_ratio >= min_height_ratio
        and face_width_ratio >= min_width_ratio
    )


def make_too_far_result(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape[:2]
    face_height_ratio = (y2 - y1) / max(frame_height, 1)
    face_width_ratio = (x2 - x1) / max(frame_width, 1)
    return {
        "liveness": "TOO_FAR",
        "confidence": 0.0,
        "raw_real_probability": 0.5,
        "temporal_probability": 0.5,
        "temporal_std": 0.0,
        "stable_frames": 0,
        "stable_enough": "NO",
        "face_height_ratio": round(float(face_height_ratio), 4),
        "face_width_ratio": round(float(face_width_ratio), 4),
        "method": "temporal_liveness_distance_guard",
    }


def format_liveness_label(temporal_result):
    liveness = temporal_result["liveness"]
    if liveness in {"REAL", "SPOOF", "TOO_FAR"}:
        return liveness
    if int(temporal_result.get("stable_frames", 0)) >= LIVENESS_SMOOTH_WINDOW:
        return "RETRY"
    return "CHECKING"


def draw_liveness_overlay(frame, bbox, label, color, temporal_result):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    debug_text = (
        f"temp {float(temporal_result['temporal_probability']):.2f} | "
        f"std {float(temporal_result['temporal_std']):.2f}"
    )
    cv2.putText(
        frame,
        debug_text,
        (x1, max(20, y1 - 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )
    draw_stability_blocks(frame, x1, max(20, y1 - 8), temporal_result, color)


def draw_stability_blocks(frame, x, y, temporal_result, color):
    block_width = 22
    block_height = 8
    block_gap = 5
    stable_frames = int(temporal_result.get("stable_frames", 0))
    if temporal_result.get("liveness") in {"REAL", "SPOOF"}:
        filled_blocks = LIVENESS_PROGRESS_BLOCKS
    else:
        filled_blocks = min(
            LIVENESS_PROGRESS_BLOCKS,
            int(
                (stable_frames * LIVENESS_PROGRESS_BLOCKS + LIVENESS_SMOOTH_WINDOW - 1)
                / LIVENESS_SMOOTH_WINDOW
            ),
        )

    for index in range(LIVENESS_PROGRESS_BLOCKS):
        left = x + index * (block_width + block_gap)
        right = left + block_width
        bottom = y + block_height
        thickness = -1 if index < filled_blocks else 1
        cv2.rectangle(frame, (left, y), (right, bottom), color, thickness)


def is_new_face_track(previous_bbox, current_bbox):
    previous_area = bbox_area(previous_bbox)
    current_area = bbox_area(current_bbox)
    if previous_area <= 0 or current_area <= 0:
        return True

    iou = bbox_iou(previous_bbox, current_bbox)
    center_shift = normalized_center_shift(previous_bbox, current_bbox)
    area_change = abs(current_area - previous_area) / max(previous_area, current_area)

    return (
        iou < LIVENESS_RESET_IOU_THRESHOLD
        or center_shift > LIVENESS_RESET_CENTER_SHIFT
        or area_change > LIVENESS_RESET_AREA_CHANGE
    )


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def bbox_iou(first_bbox, second_bbox):
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox

    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    intersection = bbox_area([inter_x1, inter_y1, inter_x2, inter_y2])
    union = bbox_area(first_bbox) + bbox_area(second_bbox) - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_shift(first_bbox, second_bbox):
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox
    first_center_x = (first_x1 + first_x2) / 2
    first_center_y = (first_y1 + first_y2) / 2
    second_center_x = (second_x1 + second_x2) / 2
    second_center_y = (second_y1 + second_y2) / 2
    center_distance = (
        (second_center_x - first_center_x) ** 2
        + (second_center_y - first_center_y) ** 2
    ) ** 0.5
    average_box_size = (
        (first_x2 - first_x1)
        + (first_y2 - first_y1)
        + (second_x2 - second_x1)
        + (second_y2 - second_y1)
    ) / 4
    return center_distance / max(average_box_size, 1.0)


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

        print("Webcam initialised. Press 'q' in the video window to quit.")

        print("Webcam initialised. Press 'Enter' in the video window to register a new face.")
        temporal_liveness = TemporalLivenessDetector(
            model_path=LIVENESS_MODEL_PATH,
            real_threshold=LIVENESS_THRESHOLD,
            window_size=LIVENESS_SMOOTH_WINDOW,
            min_frames=LIVENESS_MIN_FRAMES,
            required_confirmations=LIVENESS_CONFIRMATIONS,
            max_stable_std=LIVENESS_MAX_STABLE_STD,
        )
        frame_index = 0
        last_processed_bbox = None
        last_detection_results = None
        last_liveness_result = None
        last_label = ""
        last_color = (150, 150, 150)

        # Begin an infinite loop to process the webcam feed frame by frame.
        while live_webcam_feed.isOpened():
            
            # Read the current frame from the webcam.
            successful_read, current_video_frame = live_webcam_feed.read()
            
            # Break the loop if the webcam stops sending frames.
            if not successful_read:
                break

            should_process = frame_index % LIVENESS_PROCESS_EVERY == 0
            frame_index += 1

            if should_process:
                # Use the detector API to extract a cropped image from a detected bounding box
                detection_results = detect_and_crop_face(current_video_frame)
            else:
                detection_results = last_detection_results or {}

            # Draw bounding box
            if "face_image" in detection_results:
                current_bbox = detection_results["bbox"]
                if should_process:
                    if last_processed_bbox is not None and is_new_face_track(
                        last_processed_bbox,
                        current_bbox,
                    ):
                        temporal_liveness.reset()
                    last_processed_bbox = current_bbox
                    last_detection_results = detection_results

                    if not is_face_close_enough(
                        bbox=current_bbox,
                        frame_shape=current_video_frame.shape,
                        min_height_ratio=LIVENESS_MIN_FACE_HEIGHT_RATIO,
                        min_width_ratio=LIVENESS_MIN_FACE_WIDTH_RATIO,
                    ):
                        temporal_liveness.reset()
                        last_liveness_result = make_too_far_result(
                            current_bbox,
                            current_video_frame.shape,
                        )
                        last_label = "TOO_FAR"
                        last_color = (0, 255, 255)
                    else:
                        last_liveness_result = temporal_liveness.update(
                            detection_results["face_image"]
                        )
                        if last_liveness_result["liveness"] == "REAL":
                            classification_result = predict_identity(
                                detection_results["face_image"],
                                gallery_path="models/recognition_gallery_kaixiang.pkl",
                                model_path=(
                                    "models/"
                                    "recognition_triplet_resnet18_kaixiang_final30b_best.pth"
                                ),
                                distance_threshold=0.006,
                                distance_margin=0.0003,
                                matching_mode="mean",
                            )
                            last_color = (0, 255, 0)
                            if classification_result["similarity_score"] < 0.88:
                                last_label = "[unknown]"
                            else:
                                last_label = (
                                    f"{classification_result['best_identity']} "
                                    f"{classification_result['similarity_score']}"
                                )
                        elif last_liveness_result["liveness"] == "SPOOF":
                            last_label = "SPOOF"
                            last_color = (0, 0, 255)
                        else:
                            last_label = format_liveness_label(last_liveness_result)
                            last_color = (0, 255, 255)

                if last_liveness_result is not None:
                    draw_liveness_overlay(
                        frame=current_video_frame,
                        bbox=current_bbox,
                        label=last_label,
                        color=last_color,
                        temporal_result=last_liveness_result,
                    )
            else:
                temporal_liveness.reset()
                last_processed_bbox = None
                last_detection_results = None
                last_liveness_result = None
                last_label = ""
                last_color = (150, 150, 150)

            # Display a new image frame in a new desktop window that includes the drawn bounding boxes and labels.
            cv2.imshow("Face Detection Live Test", current_video_frame)

            # Check if the user presses the 'q' key to terminate the loop.
            keyboard_input = cv2.waitKey(1)
            if keyboard_input & 0xFF == ord('q'):
                break
            # Check if the user presses the 'Enter' key to register a new employee
            elif keyboard_input & 0xFF == 13:
                if "face_image" in detection_results:
                    if (
                        last_liveness_result is not None
                        and last_liveness_result["liveness"] == "REAL"
                    ):
                        # Open a window to register the cropped face in the employee database
                        self.register_employee(detection_results["raw_face_image"], detection_results["face_image"])
                    else:
                        messagebox.showinfo("Anti-spoofing verification", "Face was not clear enough to be verified. Please ensure that your face is fully shown.")

        # Release the webcam hardware and close all created graphical windows.
        live_webcam_feed.release()
        cv2.destroyAllWindows()

    def start(self):
        self.video_capture()

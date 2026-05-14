import os
from ultralytics import YOLO
# Author: Patrick (100599029)

if __name__ == '__main__':
    # Define the relative file path to the dataset configuration file.
    dataset_configuration_file_path = "datasets/data.yaml"

    # Initialise the YOLOv11 Nano model.
    face_detection_model = YOLO("yolo11n.pt")

    # Define the number of epochs for training.
    total_training_epochs = 30

    # Define the image resolution size for the training process.
    image_resolution_size = 640

    # Define the batch size for the GPU.
    gpu_batch_size = 8

    # Define the device ID for the GPU.
    hardware_device_id = 0

    # Define the number of background worker processes for loading data.
    number_of_data_loader_workers = 4

    # Execute the training process using the defined variables.
    training_metrics_history = face_detection_model.train(
        data=dataset_configuration_file_path,
        epochs=total_training_epochs,
        imgsz=image_resolution_size,
        batch=gpu_batch_size,
        device=hardware_device_id,
        workers=number_of_data_loader_workers,
        project="models",
        name="yolo_face_detection_run"
    )
# An API module for making requests to Tanmay's emotion detection model
# Author: Jack (105417647)

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionDetector:
    def __init__(self, model_path="models/emotion_model.h5", active=True):
        self.model = tf.keras.models.load_model(model_path)
        self.active = active

    def toggle_active(self):
        self.active = not self.active

    def detect_emotion(self, face_image: np.ndarray) -> str:
        # The model was trained on 48x48 grayscale images
        gray_img = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        scaled_gray_img = cv2.resize(gray_img, (48, 48), interpolation=cv2.INTER_NEAREST) # Fastest but lower quality interpolation

        # cv2.imshow("Scaled gray image", scaled_gray_img) # For development: Show the input image transformed for the model's prediction
        input_img = np.array([scaled_gray_img], dtype="float32")

        prediction_weights = self.model.predict(input_img, verbose=0)
        predicted_emotion = emotion_labels[np.argmax(prediction_weights)]
        # print(prediction_weights)
        # print(predicted_emotion)
        return predicted_emotion

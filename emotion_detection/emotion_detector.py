# An API module for making requests to Tanmay's emotion detection model
# Author: Jack (105417647)

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionDetector:
    def __init__(self, model_path="models/emotion_model.h5"):
        self.model = tf.keras.models.load_model(model_path)

    def detect_from_image(self, face_image: np.ndarray) -> str:
        # The model was trained on 48x48 grayscale images
        gray_img = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        # scaled_gray_img = cv2.resize(gray_img, (48, 48), interpolation=cv2.INTER_LINEAR)
        scaled_gray_img = cv2.resize(gray_img, (48, 48), interpolation=cv2.INTER_NEAREST) # Fastest but lower quality interpolation
        input_img = np.array([scaled_gray_img], dtype="float32")

        prediction_weights = self.model.predict(input_img, verbose=0)
        predicted_emotion = emotion_labels[np.argmax(prediction_weights)]
        return predicted_emotion

if __name__ == "__main__":
    emotion_detector = EmotionDetector()

    imgs = [
        cv2.imread("datasets/faces_db/jack_marken/0.jpg"),
        cv2.imread("datasets/faces_db/jack_marken/1.jpg"),
        cv2.imread("datasets/faces_db/jack_marken/2.jpg"),
        cv2.imread("datasets/faces_db/jack_marken/3.jpg")
    ]
    
    for img in imgs:
        print(emotion_detector.detect_from_image(img))
        print(emotion_detector.detect_from_image(img))
        print(emotion_detector.detect_from_image(img))
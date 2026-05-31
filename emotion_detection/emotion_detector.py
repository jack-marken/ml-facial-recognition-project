# An API module for making requests to Tanmay's emotion detection model
# Original Author: Jack (105417647)
# Edited By: Patrick (100599029)

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionDetector:
    def __init__(self, model_path="models/emotion_model.h5", active=True):
        # Load the pre-trained Keras model from the provided file path.
        self.model = tf.keras.models.load_model(model_path)
        self.active = active

    def toggle_active(self):
        # Switch the active state between True and False.
        self.active = not self.active

    def detect_emotion(self, face_image_array: np.ndarray) -> str:
        # Convert the input RGB image to a grayscale image.
        grayscale_image = cv2.cvtColor(face_image_array, cv2.COLOR_RGB2GRAY)
        
        # Resize the grayscale image to 48 by 48 pixels to match the input shape of the model.
        resized_grayscale_image = cv2.resize(grayscale_image, (48, 48), interpolation=cv2.INTER_NEAREST)
        
        # Convert the image data type to a 32-bit floating point number.
        float_image_array = np.array(resized_grayscale_image, dtype="float32")
        
        # Normalise the pixel values to a range between 0.0 and 1.0.
        # This is required because the model was trained using rescaled images.
        normalized_image_array = float_image_array / 255.0
        
        # Add a batch dimension at the start of the array to create a shape of (1, 48, 48).
        image_with_batch_dimension = np.expand_dims(normalized_image_array, axis=0)
        
        # Add a channel dimension at the end of the array to create a final shape of (1, 48, 48, 1).
        # This matches the expected input shape of the convolutional neural network.
        final_input_tensor = np.expand_dims(image_with_batch_dimension, axis=-1)

        # Generate probability weights for each emotion category.
        prediction_weights_array = self.model.predict(final_input_tensor, verbose=0)
        
        # Find the index of the highest probability weight.
        highest_probability_index = np.argmax(prediction_weights_array)
        
        # Retrieve the corresponding human-readable emotion label.
        predicted_emotion_string = emotion_labels[highest_probability_index]
        
        return predicted_emotion_string
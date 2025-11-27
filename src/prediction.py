# src/prediction.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# helper that loads a model (path to .keras or .h5) and returns a function to predict single image
def load_model_for_prediction(model_path):
    """
    Load a Keras model from disk and return it.
    """
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image_file_bytes(file_path, target_size=(224, 224)):
    """
    Load image from file path, preprocess for VGG16, and return batch shaped array.
    """
    img = load_img(file_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict_image_from_path(model, image_path, threshold=0.5):
    """
    Predict single image from a saved image path.
    Returns label and confidence.
    """
    arr = preprocess_image_file_bytes(image_path)
    prob = float(model.predict(arr)[0][0])
    label = "PNEUMONIA" if prob > threshold else "NORMAL"
    confidence = prob if prob > threshold else 1.0 - prob
    return {"label": label, "confidence": float(confidence), "probability": prob}

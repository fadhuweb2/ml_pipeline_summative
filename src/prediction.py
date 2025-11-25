# src/prediction.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load TFLite model once
def load_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(file_path, target_size=(224, 224)):
    img = load_img(file_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = (x / 127.5) - 1.0  # Normalize similar to VGG preprocessing
    return x.astype(np.float32)

def predict_image_from_path(interpreter, image_path, threshold=0.5):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_array = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    label = "PNEUMONIA" if prob > threshold else "NORMAL"
    confidence = prob if prob > threshold else 1.0 - prob
    return {"label": label, "confidence": float(confidence), "probability": prob}

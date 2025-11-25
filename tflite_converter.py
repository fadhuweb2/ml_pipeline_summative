import tensorflow as tf

# Paths
keras_model_path = "models/chest_xray_model.keras"
tflite_model_path = "models/chest_xray_model.tflite"

# Load Keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_model_path}")

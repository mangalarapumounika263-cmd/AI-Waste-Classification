import tensorflow as tf
import os

INPUT_MODEL = "models/waste_classifier_mobilenet.keras"
OUTPUT_MODEL = "models/waste_classifier_mobilenet.tflite"

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(INPUT_MODEL)

print("Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Reduce model size while maintaining good accuracy
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types = [
    tf.float16
]

tflite_model = converter.convert()

with open(OUTPUT_MODEL, "wb") as file:
    file.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)

print()
print("==============================")
print("CONVERSION COMPLETE")
print("==============================")
print(f"Model: {OUTPUT_MODEL}")
print(f"Size: {size_mb:.2f} MB")
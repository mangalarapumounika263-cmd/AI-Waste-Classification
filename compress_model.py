import tensorflow as tf
import os

INPUT_MODEL = "models/waste_classifier.keras"
OUTPUT_MODEL = "models/waste_classifier_fp16.keras"

print("Loading model...")
model = tf.keras.models.load_model(INPUT_MODEL)

print("Saving compressed model...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

OUTPUT_FILE = "models/waste_classifier.tflite"

with open(OUTPUT_FILE, "wb") as f:
    f.write(tflite_model)

print("Compressed model saved:")
print(OUTPUT_FILE)

size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"Model size: {size_mb:.2f} MB")
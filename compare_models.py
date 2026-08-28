import tensorflow as tf
import numpy as np
import json
from PIL import Image

KERAS_MODEL = "models/waste_classifier_mobilenet.keras"
TFLITE_MODEL = "models/waste_classifier_mobilenet.tflite"
IMAGE_PATH = "real_test/plastic2.jpg"

with open("models/class_names.json", "r") as f:
    classes = json.load(f)

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((224, 224))

x = np.array(image, dtype=np.float32)
x = np.expand_dims(x, axis=0)

# -----------------------------
# KERAS
# -----------------------------

print("\nLoading Keras model...")

keras_model = tf.keras.models.load_model(
    KERAS_MODEL
)

keras_prediction = keras_model.predict(
    x,
    verbose=0
)[0]

print("\n==============================")
print("KERAS PREDICTION")
print("==============================")

for name, value in sorted(
    zip(classes, keras_prediction),
    key=lambda z: z[1],
    reverse=True
):
    print(
        f"{name:12} {value * 100:.2f}%"
    )

# -----------------------------
# TFLITE
# -----------------------------

print("\nLoading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(
    input_details[0]["index"],
    x
)

interpreter.invoke()

tflite_prediction = interpreter.get_tensor(
    output_details[0]["index"]
)[0]

print("\n==============================")
print("TFLITE PREDICTION")
print("==============================")

for name, value in sorted(
    zip(classes, tflite_prediction),
    key=lambda z: z[1],
    reverse=True
):
    print(
        f"{name:12} {value * 100:.2f}%"
    )
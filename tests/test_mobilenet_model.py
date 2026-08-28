import tensorflow as tf
import numpy as np
import json

MODEL_PATH = "models/waste_classifier_mobilenet.tflite"
CLASS_PATH = "models/class_names.json"

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]["shape"])
print("Input type:", input_details[0]["dtype"])

print("Output shape:", output_details[0]["shape"])

print("Classes:", class_names)

# Create test image
input_shape = input_details[0]["shape"]

dummy_image = np.zeros(
    (
        input_shape[1],
        input_shape[2],
        3
    ),
    dtype=np.float32
)

dummy_image = np.expand_dims(
    dummy_image,
    axis=0
)

# MobileNetV2 preprocessing
dummy_image = (
    dummy_image / 127.5
) - 1.0

interpreter.set_tensor(
    input_details[0]["index"],
    dummy_image
)

interpreter.invoke()

prediction = interpreter.get_tensor(
    output_details[0]["index"]
)[0]

print()
print("Prediction:", prediction)

predicted_index = int(
    np.argmax(prediction)
)

print(
    "Predicted class:",
    class_names[predicted_index]
)

print(
    "Confidence:",
    f"{prediction[predicted_index] * 100:.2f}%"
)

print()
print("Model test successful!")
import tensorflow as tf
import json
import numpy as np

MODEL_PATH = "models/waste_classifier.tflite"
CLASS_PATH = "models/class_names.json"

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details[0]["shape"])
print("Input type:", input_details[0]["dtype"])
print("Output:", output_details[0]["shape"])
print("Classes:", class_names)

input_shape = input_details[0]["shape"]

dummy_image = np.zeros(
    (input_shape[1], input_shape[2], 3),
    dtype=np.float32
)

dummy_image = np.expand_dims(dummy_image, axis=0)

interpreter.set_tensor(
    input_details[0]["index"],
    dummy_image
)

interpreter.invoke()

prediction = interpreter.get_tensor(
    output_details[0]["index"]
)

print("Prediction output:", prediction)
print("Model test successful!")
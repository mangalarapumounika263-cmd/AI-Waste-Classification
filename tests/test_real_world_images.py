import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

MODEL_PATH = "models/waste_classifier_mobilenet.tflite"
IMAGE_DIR = "real_test"

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n========================================")
print("REAL WORLD IMAGE TEST")
print("========================================\n")

files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
]

if not files:
    print("No images found in real_test/")
    exit()

for filename in files:

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    sorted_results = sorted(
        zip(class_names, predictions),
        key=lambda x: x[1],
        reverse=True
    )

    print("=" * 50)
    print(filename)
    print("=" * 50)

    for class_name, probability in sorted_results:

        print(
            f"{class_name:12} "
            f"{probability * 100:6.2f}%"
        )

    print()

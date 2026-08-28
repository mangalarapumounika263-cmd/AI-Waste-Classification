import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/waste_classifier_v2.keras"
TEST_DIR = "real_test"

IMG_SIZE = (224, 224)

# Load model
print("Loading V2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

print("\n" + "=" * 60)
print("V2 REAL WORLD IMAGE TEST")
print("=" * 60)

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)

files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(image_extensions)
]

if not files:
    print("\nNo images found in real_test folder.")
    print("Put your test images inside:")
    print(os.path.abspath(TEST_DIR))
    exit()

for filename in sorted(files):

    image_path = os.path.join(
        TEST_DIR,
        filename
    )

    try:

        image = Image.open(image_path).convert("RGB")

        image = image.resize(IMG_SIZE)

        image_array = np.array(
            image,
            dtype=np.float32
        )

        # MobileNetV2 preprocessing
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Sort predictions from highest to lowest
        sorted_indices = np.argsort(
            predictions
        )[::-1]

        print("\n" + "=" * 50)
        print(filename)
        print("=" * 50)

        for index in sorted_indices:

            print(
                f"{class_names[index]:12} "
                f"{predictions[index] * 100:6.2f}%"
            )

        predicted_index = sorted_indices[0]

        print("\nPrediction:")
        print(
            class_names[predicted_index]
        )

        print(
            f"Confidence: "
            f"{predictions[predicted_index] * 100:.2f}%"
        )

    except Exception as e:

        print(
            f"\nError processing {filename}: {e}"
        )

print("\n" + "=" * 60)
print("V2 REAL WORLD TEST COMPLETED")
print("=" * 60)
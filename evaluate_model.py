import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "models/waste_classifier_mobilenet.keras"

TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

matrix = confusion_matrix(
    y_true,
    y_pred
)

print(matrix)

accuracy = np.mean(
    np.array(y_true) == np.array(y_pred)
)

print("\nValidation Accuracy:")
print(f"{accuracy * 100:.2f}%")
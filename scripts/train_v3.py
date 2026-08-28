import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    KERAS_MODEL_PATH,
    REAL_DATASET_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
)
from src.model import build_v3_model

BATCH_SIZE = 32
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 15
CLASS_WEIGHT_CAP = 2.5


def count_images(root: Path) -> dict:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        counts[class_name] = (
            len([p for p in class_dir.rglob("*") if p.suffix.lower() in extensions])
            if class_dir.exists()
            else 0
        )
    return counts


def print_distribution(title: str, counts: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for class_name in CLASS_NAMES:
        print(f"{class_name:10} {counts.get(class_name, 0):5}")
    print(f"{'total':10} {sum(counts.values()):5}")


def controlled_class_weights(counts: dict) -> dict:
    values = np.array([max(counts[name], 1) for name in CLASS_NAMES], dtype=np.float32)
    total = float(values.sum())
    raw = total / (len(CLASS_NAMES) * values)
    capped = np.clip(raw, 0.5, CLASS_WEIGHT_CAP)
    return {index: float(capped[index]) for index in range(len(CLASS_NAMES))}


def make_dataset(root: Path, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(224, 224),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42 if shuffle else None,
    )


def main() -> None:
    print("=" * 70)
    print("AI WASTE CLASSIFICATION - V3 TRAINING")
    print("=" * 70)
    print("TensorFlow:", tf.__version__)

    train_counts = count_images(TRAIN_DIR)
    validation_counts = count_images(VALIDATION_DIR)
    real_counts = count_images(REAL_DATASET_DIR)
    combined_counts = {
        name: train_counts[name] + real_counts[name]
        for name in CLASS_NAMES
    }

    print_distribution("Training distribution", train_counts)
    print_distribution("Validation distribution", validation_counts)
    print_distribution("Additional real_dataset distribution", real_counts)

    missing = [name for name, count in train_counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"Missing training images for classes: {missing}")

    train_dataset = make_dataset(TRAIN_DIR, shuffle=True)
    if REAL_DATASET_DIR.exists() and sum(real_counts.values()) > 0:
        real_dataset = make_dataset(REAL_DATASET_DIR, shuffle=True)
        train_dataset = train_dataset.concatenate(real_dataset)

    validation_dataset = make_dataset(VALIDATION_DIR, shuffle=False)
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    validation_dataset = validation_dataset.prefetch(autotune)

    class_weights = controlled_class_weights(combined_counts)
    print("\nControlled class weights")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:10} {class_weights[index]:.3f}")

    KERAS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            KERAS_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model, base_model = build_v3_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("\nPHASE 1: training classification head")
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\nPHASE 2: fine-tuning later MobileNetV2 layers")
    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    validation_loss, validation_accuracy = best_model.evaluate(validation_dataset, verbose=0)
    print(f"\nValidation accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Validation loss: {validation_loss:.4f}")

    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(CLASS_NAMES, file, indent=4)


if __name__ == "__main__":
    main()

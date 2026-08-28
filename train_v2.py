import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ============================================================
# AI WASTE CLASSIFICATION - V2 TRAINING
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"
MODEL_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 20

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "waste_classifier_v2.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("AI WASTE CLASSIFICATION - V2 TRAINING")
print("=" * 60)

print("\nTensorFlow:", tf.__version__)


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# Current number of training images
#
# cardboard = 322
# glass     = 400
# metal     = 328
# paper     = 475
# plastic   = 385
# trash     = 109

class_counts = np.array(
    [
        322,
        400,
        328,
        475,
        385,
        109
    ],
    dtype=np.float32
)

total_images = np.sum(class_counts)

num_classes = len(CLASS_NAMES)

class_weights = {}

for i, count in enumerate(class_counts):

    class_weights[i] = (
        total_images /
        (num_classes * count)
    )


print("\nClass weights:")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name:12} "
        f"{class_weights[i]:.3f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.20
        ),

        layers.RandomZoom(
            height_factor=(-0.15, 0.20),
            width_factor=(-0.15, 0.20)
        ),

        layers.RandomTranslation(
            height_factor=0.15,
            width_factor=0.15
        ),

        layers.RandomContrast(
            0.20
        ),

        layers.RandomBrightness(
            0.15
        ),

    ],

    name="data_augmentation"
)


# ============================================================
# LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(

    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)

# Initially freeze MobileNetV2
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        224,
        224,
        3
    )
)


# Data augmentation
x = data_augmentation(inputs)


# MobileNetV2 preprocessing
#
# Converts RGB values from:
#
# 0 - 255
#
# to:
#
# -1 to +1

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


# MobileNetV2 feature extractor
x = base_model(
    x,
    training=False
)


# Global average pooling
x = layers.GlobalAveragePooling2D()(x)


# Dropout
x = layers.Dropout(
    0.40
)(x)


# Dense layer
x = layers.Dense(
    128,
    activation="relu"
)(x)


# More dropout
x = layers.Dropout(
    0.30
)(x)


# Six waste classes
outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax"
)(x)


# Final model
model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# PHASE 1 - COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\nModel created successfully.")


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        verbose=1
    ),

    EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-7,

        verbose=1
    )

]


# ============================================================
# PHASE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n")

print("=" * 60)

print(
    "PHASE 1: "
    "TRAINING CLASSIFICATION HEAD"
)

print("=" * 60)


history1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n")

print("=" * 60)

print(
    "PHASE 2: "
    "FINE-TUNING MOBILENETV2"
)

print("=" * 60)


# Unfreeze MobileNetV2
base_model.trainable = True


# Freeze first 100 layers
for layer in base_model.layers[:100]:

    layer.trainable = False


# ============================================================
# RECOMPILE FOR FINE TUNING
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# FINE-TUNE
# ============================================================

history2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n")

print("=" * 60)

print("V2 MODEL EVALUATION")

print("=" * 60)


loss, accuracy = model.evaluate(
    validation_dataset
)


print(
    f"\nValidation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Validation Loss: "
    f"{loss:.4f}"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)


with open(
    class_path,
    "w"
) as f:

    json.dump(
        CLASS_NAMES,
        f,
        indent=4
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\nModel saved to:")

print(
    MODEL_PATH
)


print("\nClass names saved to:")

print(
    class_path
)


print("\n")

print("=" * 60)

print(
    "V2 TRAINING COMPLETED SUCCESSFULLY!"
)

print("=" * 60)
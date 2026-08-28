from .config import CLASS_NAMES, IMAGE_SIZE


def build_v3_model(trainable_backbone_layers: int = 0):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.12),
            layers.RandomBrightness(0.08),
        ],
        name="controlled_augmentation",
    )

    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False
    if trainable_backbone_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_backbone_layers]:
            layer.trainable = False

    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="waste_classifier_v3"), base_model

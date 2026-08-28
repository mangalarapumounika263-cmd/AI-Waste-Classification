import numpy as np
import sys
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import KERAS_MODEL_PATH, TFLITE_MODEL_PATH


def main() -> None:
    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train or provide the Keras model first: {KERAS_MODEL_PATH}")

    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    TFLITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TFLITE_MODEL_PATH.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    dummy = np.zeros(input_details[0]["shape"], dtype=input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], dummy)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    print("TFLite conversion complete")
    print("Input shape:", input_details[0]["shape"].tolist())
    print("Input dtype:", input_details[0]["dtype"])
    print("Output shape:", list(output.shape))
    if list(input_details[0]["shape"]) != [1, 224, 224, 3] or list(output.shape) != [1, 6]:
        raise RuntimeError("Unexpected TFLite model input or output shape.")


if __name__ == "__main__":
    main()

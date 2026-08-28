import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, TFLITE_MODEL_PATH
from src.predictor import WastePredictor


def main() -> None:
    if not TFLITE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing deployment model: {TFLITE_MODEL_PATH}")

    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = WastePredictor().predict(image)
    print("Predicted class:", result.predicted_class)
    print("Confidence:", f"{result.confidence * 100:.2f}%")
    print("Top 3:", result.top_3_predictions)
    assert result.predicted_class in CLASS_NAMES
    assert len(result.probabilities) == len(CLASS_NAMES)
    assert abs(sum(result.probabilities) - 1.0) < 0.05


if __name__ == "__main__":
    main()

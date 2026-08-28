import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, MODEL_DIR, REAL_DATASET_DIR, REAL_TEST_DIR, REPORTS_DIR, VALIDATION_DIR
from src.predictor import normalize_probabilities

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_CANDIDATES = [
    MODEL_DIR / "waste_classifier_v3.tflite",
    MODEL_DIR / "waste_classifier_mobilenet.tflite",
    MODEL_DIR / "waste_classifier.tflite",
    MODEL_DIR / "waste_classifier_mobilenet.keras",
    MODEL_DIR / "waste_classifier_v2.keras",
    MODEL_DIR / "waste_classifier.keras",
]


def collect_labeled_images(root: Path):
    rows = []
    invalid = []
    if not root.exists():
        return rows, invalid
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        actual = path.parent.name.lower() if path.parent.name.lower() in CLASS_NAMES else None
        if actual is None:
            lower_name = path.name.lower()
            actual = next((name for name in CLASS_NAMES if lower_name.startswith(name)), None)
            if actual is None and lower_name.startswith("cardpoard"):
                actual = "cardboard"
        if actual is None:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            rows.append((path, actual))
        except (UnidentifiedImageError, OSError) as error:
            invalid.append({"filename": str(path), "error": str(error)})
    return rows, invalid


def preprocess_raw(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


class CandidateModel:
    def __init__(self, path: Path):
        self.path = path
        self.kind = path.suffix.lower()
        self.model = None
        self.input_details = None
        self.output_details = None

    def load(self):
        if self.kind == ".tflite":
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter
            self.model = Interpreter(model_path=str(self.path))
            self.model.allocate_tensors()
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.path)

    def predict(self, image_path: Path) -> np.ndarray:
        array = preprocess_raw(image_path)
        if self.kind == ".tflite":
            input_dtype = self.input_details[0]["dtype"]
            self.model.set_tensor(self.input_details[0]["index"], array.astype(input_dtype))
            self.model.invoke()
            output = self.model.get_tensor(self.output_details[0]["index"])
            return normalize_probabilities(output)
        output = self.model.predict(array, verbose=0)[0]
        return normalize_probabilities(output)


def evaluate_candidate(path: Path, dataset_name: str, images: list[tuple[Path, str]]) -> dict:
    candidate = CandidateModel(path)
    candidate.load()
    y_true = []
    y_pred = []
    confidences = []
    for image_path, actual in images:
        probabilities = candidate.predict(image_path)
        prediction = int(np.argmax(probabilities))
        y_true.append(CLASS_NAMES.index(actual))
        y_pred.append(prediction)
        confidences.append(float(probabilities[prediction]))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": path.name,
        "dataset": dataset_name,
        "images": len(images),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "plastic_recall": report["plastic"]["recall"],
        "trash_recall": report["trash"]["recall"],
        "metal_recall": report["metal"]["recall"],
        "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "validation": collect_labeled_images(VALIDATION_DIR)[0],
        "real_dataset": collect_labeled_images(REAL_DATASET_DIR)[0],
        "real_test": collect_labeled_images(REAL_TEST_DIR)[0],
    }
    rows = []
    for model_path in MODEL_CANDIDATES:
        if not model_path.exists():
            continue
        for dataset_name, images in datasets.items():
            if not images:
                continue
            print(f"Evaluating {model_path.name} on {dataset_name} ({len(images)} images)")
            rows.append(evaluate_candidate(model_path, dataset_name, images))

    if not rows:
        print("No comparable models or labeled images found.")
        return

    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "accuracy", "macro_f1", "trash_recall", "plastic_recall"],
        ascending=[True, False, False, False, False],
    )
    frame.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved comparison to {REPORTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()

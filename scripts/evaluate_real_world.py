import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, REAL_TEST_DIR, REPORTS_DIR
from src.predictor import WastePredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FILENAME_CLASS_ALIASES = {"cardpoard": "cardboard", "cardboard": "cardboard"}


def discover_images(root: Path):
    if not root.exists():
        return []
    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parent = path.parent.name.lower()
        actual = parent if parent in CLASS_NAMES else None
        if actual is None:
            lower_name = path.name.lower()
            actual = next((name for name in CLASS_NAMES if lower_name.startswith(name)), None)
        if actual is None:
            actual = next(
                (class_name for alias, class_name in FILENAME_CLASS_ALIASES.items() if lower_name.startswith(alias)),
                None,
            )
        rows.append((path, actual))
    return rows


def main() -> None:
    images = discover_images(REAL_TEST_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not images:
        print("No real-world images found. Add images under real_test/<class>/ and rerun.")
        return

    predictor = WastePredictor()
    y_true = []
    y_pred = []
    rows = []
    low_confidence = []
    incorrect = []

    for image_path, actual in images:
        if actual not in CLASS_NAMES:
            print(f"Skipping {image_path.name}: unable to infer actual class from folder or filename.")
            continue
        with Image.open(image_path) as image:
            result = predictor.predict(image)
        predicted = result.predicted_class
        y_true.append(CLASS_NAMES.index(actual))
        y_pred.append(CLASS_NAMES.index(predicted))
        top_3 = [
            f"{item['class_name']}={item['probability'] * 100:.2f}%"
            for item in result.top_3_predictions
        ]
        row = {
            "filename": str(image_path.relative_to(REAL_TEST_DIR)),
            "actual": actual,
            "predicted": predicted,
            "confidence": round(result.confidence * 100, 4),
            "confidence_level": result.confidence_level,
            "uncertainty": result.uncertainty or "",
            "top_3": "; ".join(top_3),
        }
        rows.append(row)
        if predicted != actual:
            incorrect.append(row)
        if result.confidence_level == "Low" or result.uncertainty:
            low_confidence.append(row)

    if not y_true:
        print("No labeled real-world images were found.")
        return

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    per_class_accuracy = {}
    for index, class_name in enumerate(CLASS_NAMES):
        actual_count = int(matrix[index].sum())
        per_class_accuracy[class_name] = (
            float(matrix[index, index] / actual_count) if actual_count else None
        )

    pd.DataFrame(rows).to_csv(REPORTS_DIR / "real_world_predictions.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        REPORTS_DIR / "real_world_confusion_matrix.csv"
    )
    (REPORTS_DIR / "real_world_classification_report.txt").write_text(report_text, encoding="utf-8")
    (REPORTS_DIR / "real_world_classification_report.json").write_text(
        json.dumps(
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "report": report_dict,
                "per_class_accuracy": per_class_accuracy,
                "incorrect_predictions": incorrect,
                "low_confidence_predictions": low_confidence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report_text)
    print(f"Real-world accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("\nIncorrect predictions:", len(incorrect))
    print("Low-confidence or uncertain predictions:", len(low_confidence))


if __name__ == "__main__":
    main()

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, REPORTS_DIR, VALIDATION_DIR
from src.predictor import WastePredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_validation_images():
    rows = []
    invalid = []
    for class_name in CLASS_NAMES:
        class_dir = VALIDATION_DIR / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    rows.append((path, class_name))
                except (UnidentifiedImageError, OSError) as error:
                    invalid.append({"filename": str(path), "error": str(error)})
    return rows, invalid


def main() -> None:
    images, invalid = collect_validation_images()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Validation image distribution")
    for class_name in CLASS_NAMES:
        print(f"{class_name:10} {sum(1 for _, actual in images if actual == class_name):5}")
    print(f"{'total':10} {len(images):5}")
    if invalid:
        pd.DataFrame(invalid).to_csv(REPORTS_DIR / "invalid_validation_images.csv", index=False)
        print(f"Invalid images: {len(invalid)}")

    if not images:
        print("No validation images found.")
        return

    predictor = WastePredictor()
    y_true = []
    y_pred = []
    prediction_rows = []

    for image_path, actual in images:
        with Image.open(image_path) as image:
            result = predictor.predict(image)
        y_true.append(CLASS_NAMES.index(actual))
        y_pred.append(CLASS_NAMES.index(result.predicted_class))
        prediction_rows.append(
            {
                "filename": str(image_path.relative_to(VALIDATION_DIR)),
                "actual": actual,
                "predicted": result.predicted_class,
                "confidence": round(result.confidence * 100, 4),
                "confidence_level": result.confidence_level,
                "uncertainty": result.uncertainty or "",
            }
        )

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

    pd.DataFrame(prediction_rows).to_csv(REPORTS_DIR / "validation_predictions.csv", index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        REPORTS_DIR / "confusion_matrix.csv"
    )
    (REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (REPORTS_DIR / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )

    print(report_text)
    print("Validation accuracy:", f"{accuracy_score(y_true, y_pred) * 100:.2f}%")


if __name__ == "__main__":
    main()

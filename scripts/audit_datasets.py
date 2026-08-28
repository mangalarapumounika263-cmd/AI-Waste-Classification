import sys
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, REAL_DATASET_DIR, REAL_TEST_DIR, REPORTS_DIR, TRAIN_DIR, VALIDATION_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def audit_root(root: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    invalid = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        images = []
        if class_dir.exists():
            images = [path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            for path in images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError) as error:
                    invalid.append({"dataset": str(root), "class": class_name, "filename": str(path), "error": str(error)})
        rows.append(
            {
                "dataset": root.name,
                "class": class_name,
                "count": len(images),
                "missing_class": not class_dir.exists(),
            }
        )
    return rows, invalid


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_invalid = []

    for root in [TRAIN_DIR, VALIDATION_DIR, REAL_DATASET_DIR, REAL_TEST_DIR]:
        rows, invalid = audit_root(root)
        all_rows.extend(rows)
        all_invalid.extend(invalid)
        print(f"\n{root}")
        for row in rows:
            missing = " missing" if row["missing_class"] else ""
            print(f"{row['class']:10} {row['count']:5}{missing}")
        print(f"{'total':10} {sum(row['count'] for row in rows):5}")

    pd.DataFrame(all_rows).to_csv(REPORTS_DIR / "dataset_audit.csv", index=False)
    pd.DataFrame(all_invalid).to_csv(REPORTS_DIR / "invalid_images.csv", index=False)
    print(f"\nInvalid images: {len(all_invalid)}")
    print(f"Saved audit reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()

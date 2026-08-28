from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
REAL_DATASET_DIR = BASE_DIR / "real_dataset"
REAL_TEST_DIR = BASE_DIR / "real_test"

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
KERAS_MODEL_PATH = MODEL_DIR / "waste_classifier_v3.keras"
TFLITE_MODEL_PATH = MODEL_DIR / "waste_classifier_v3.tflite"
HISTORY_DB_PATH = DATABASE_DIR / "prediction_history.db"

IMAGE_SIZE = (224, 224)
MIN_IMAGE_WIDTH = 96
MIN_IMAGE_HEIGHT = 96
DARK_IMAGE_MEAN_THRESHOLD = 25.0
BRIGHT_IMAGE_MEAN_THRESHOLD = 235.0

MODEL_INCLUDES_PREPROCESSING = True
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.50
UNCERTAIN_MARGIN_THRESHOLD = 0.10
HIGH_ENTROPY_THRESHOLD = 1.45

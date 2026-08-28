import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover - only used in training environments
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:  # pragma: no cover
        Interpreter = None

from .config import (
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    TFLITE_MODEL_PATH,
    UNCERTAIN_MARGIN_THRESHOLD,
)
from .preprocessing import check_image_quality, preprocess_for_mobilenet_v2
from .recommendations import get_recommendation


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    confidence_level: str
    probabilities: List[float]
    top_3_predictions: List[dict]
    uncertainty: Optional[str]
    recommendation: str
    quality_warnings: List[str]


def load_class_names(path: Path = CLASS_NAMES_PATH) -> List[str]:
    if not path.exists():
        return CLASS_NAMES.copy()
    with path.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if class_names != CLASS_NAMES:
        raise ValueError(f"class_names.json must use this order: {CLASS_NAMES}")
    return class_names


def normalize_probabilities(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Model returned an empty prediction.")

    total = float(np.sum(values))
    if np.all(values >= 0) and np.isclose(total, 1.0, atol=0.05):
        probabilities = values / total
    else:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values)

    return probabilities.astype(np.float32)


def confidence_level(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


def uncertainty_reason(class_names: List[str], probabilities: np.ndarray) -> Optional[str]:
    order = np.argsort(probabilities)[::-1]
    top = float(probabilities[order[0]])
    second = float(probabilities[order[1]])
    if top - second < UNCERTAIN_MARGIN_THRESHOLD:
        return (
            f"Uncertain: {class_names[order[0]].title()} and "
            f"{class_names[order[1]].title()} are visually similar in this image."
        )
    if top < MEDIUM_CONFIDENCE_THRESHOLD:
        return "Low-confidence prediction. Try a clearer image with one dominant waste item."
    return None


class WastePredictor:
    def __init__(
        self,
        model_path: Path = TFLITE_MODEL_PATH,
        class_path: Path = CLASS_NAMES_PATH,
    ) -> None:
        self.model_path = model_path
        self.class_names = load_class_names(class_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load(self) -> None:
        if Interpreter is None:
            raise RuntimeError("Install ai-edge-litert to run TFLite inference.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, image: Image.Image) -> PredictionResult:
        if self.interpreter is None:
            self.load()

        quality = check_image_quality(image)
        input_data = preprocess_for_mobilenet_v2(image)
        input_dtype = self.input_details[0]["dtype"]
        if input_dtype != np.float32:
            input_data = input_data.astype(input_dtype)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details[0]["index"])
        probabilities = normalize_probabilities(raw_output)

        if len(probabilities) != len(self.class_names):
            raise ValueError(
                f"Model returned {len(probabilities)} outputs for {len(self.class_names)} classes."
            )

        order = np.argsort(probabilities)[::-1]
        top_index = int(order[0])
        top_3 = [
            {
                "class_name": self.class_names[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in order[:3]
        ]
        predicted_class = self.class_names[top_index]
        confidence = float(probabilities[top_index])
        uncertainty = uncertainty_reason(self.class_names, probabilities)

        return PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            confidence_level=confidence_level(confidence),
            probabilities=[float(value) for value in probabilities],
            top_3_predictions=top_3,
            uncertainty=uncertainty,
            recommendation=get_recommendation(predicted_class),
            quality_warnings=quality.warnings,
        )

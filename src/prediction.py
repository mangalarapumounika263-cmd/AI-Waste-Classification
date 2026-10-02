"""Public prediction service.

`src.predictor` remains as a compatibility module for older scripts; all new
callers should import this module so there is one inference implementation.
"""

from .predictor import (
    PredictionResult,
    WastePredictor,
    confidence_level,
    load_class_names,
    normalize_probabilities,
    uncertainty_reason,
)

__all__ = [
    "PredictionResult",
    "WastePredictor",
    "confidence_level",
    "load_class_names",
    "normalize_probabilities",
    "uncertainty_reason",
]

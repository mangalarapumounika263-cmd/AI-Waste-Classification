import numpy as np

from src.config import CLASS_NAMES
from src.prediction import confidence_level, normalize_probabilities, uncertainty_reason


def test_probabilities_are_normalized_and_top_class_is_retained():
    probabilities = normalize_probabilities(np.array([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0]]))
    assert np.isclose(probabilities.sum(), 1.0)
    assert int(np.argmax(probabilities)) == 0


def test_confidence_bands_and_ambiguous_top_predictions():
    assert confidence_level(0.75) == "High"
    assert confidence_level(0.50) == "Medium"
    assert confidence_level(0.49) == "Low"
    assert uncertainty_reason(CLASS_NAMES, np.array([0.51, 0.45, 0.01, 0.01, 0.01, 0.01]))

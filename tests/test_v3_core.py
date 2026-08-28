import numpy as np
import unittest
from pathlib import Path
from PIL import Image

from src.config import CLASS_NAMES
from src.predictor import confidence_level, load_class_names, normalize_probabilities, uncertainty_reason
from src.preprocessing import check_image_quality, preprocess_for_mobilenet_v2


class V3CoreTests(unittest.TestCase):
    def test_class_loading_falls_back_to_expected_order(self):
        self.assertEqual(load_class_names(Path("missing-class-file.json")), CLASS_NAMES)

    def test_preprocessing_shape_and_range(self):
        image = Image.new("RGB", (320, 240), color=(128, 128, 128))
        array = preprocess_for_mobilenet_v2(image)
        self.assertEqual(array.shape, (1, 224, 224, 3))
        self.assertEqual(array.dtype, np.float32)
        self.assertGreaterEqual(float(array.min()), -1.0)
        self.assertLessEqual(float(array.max()), 1.0)

    def test_probability_normalization_keeps_probabilities(self):
        probabilities = normalize_probabilities(np.array([[0.2, 0.3, 0.1, 0.1, 0.2, 0.1]]))
        self.assertTrue(np.isclose(probabilities.sum(), 1.0))
        self.assertEqual(int(np.argmax(probabilities)), 1)

    def test_probability_normalization_softmaxes_logits(self):
        probabilities = normalize_probabilities(np.array([[4.0, 2.0, 1.0, 0.0, -1.0, -2.0]]))
        self.assertTrue(np.isclose(probabilities.sum(), 1.0))
        self.assertEqual(int(np.argmax(probabilities)), 0)

    def test_confidence_levels(self):
        self.assertEqual(confidence_level(0.81), "High")
        self.assertEqual(confidence_level(0.50), "Medium")
        self.assertEqual(confidence_level(0.49), "Low")

    def test_uncertainty_when_top_two_are_close(self):
        probabilities = np.array([0.52, 0.44, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        reason = uncertainty_reason(CLASS_NAMES, probabilities)
        self.assertIsNotNone(reason)
        self.assertIn("Uncertain", reason)

    def test_image_quality_warns_for_small_image(self):
        image = Image.new("RGB", (20, 20), color=(120, 120, 120))
        quality = check_image_quality(image)
        self.assertTrue(quality.is_valid)
        self.assertTrue(quality.warnings)

    def test_invalid_image_handling(self):
        with self.assertRaises(ValueError):
            preprocess_for_mobilenet_v2(None)


if __name__ == "__main__":
    unittest.main()

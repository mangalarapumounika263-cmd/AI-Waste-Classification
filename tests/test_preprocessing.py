import numpy as np
import pytest
from PIL import Image

from src.preprocessing import check_image_quality, preprocess_for_mobilenet_v2


def test_preprocessing_preserves_required_shape_and_mobilenet_range():
    image = Image.new("RGB", (480, 160), (128, 128, 128))
    array = preprocess_for_mobilenet_v2(image)
    assert array.shape == (1, 224, 224, 3)
    assert array.dtype == np.float32
    assert -1.0 <= float(array.min()) <= float(array.max()) <= 1.0


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        preprocess_for_mobilenet_v2(None)


def test_quality_identifies_dark_and_blurry_image():
    quality = check_image_quality(Image.new("RGB", (224, 224), (0, 0, 0)))
    assert quality.brightness_status == "Too dark"
    assert quality.sharpness_status == "Blurry"
    assert quality.warnings

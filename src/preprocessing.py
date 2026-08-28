from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image, ImageStat

from .config import (
    BRIGHT_IMAGE_MEAN_THRESHOLD,
    DARK_IMAGE_MEAN_THRESHOLD,
    IMAGE_SIZE,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)


@dataclass
class ImageQuality:
    is_valid: bool
    warnings: List[str]
    width: int
    height: int
    mean_brightness: float


def ensure_rgb_image(image: Image.Image) -> Image.Image:
    if image is None:
        raise ValueError("No image was provided.")
    return image.convert("RGB")


def check_image_quality(image: Image.Image) -> ImageQuality:
    rgb_image = ensure_rgb_image(image)
    width, height = rgb_image.size
    warnings: List[str] = []

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        warnings.append(
            f"Image is small ({width}x{height}); image quality may reduce prediction reliability."
        )

    grayscale = rgb_image.convert("L")
    mean_brightness = float(ImageStat.Stat(grayscale).mean[0])
    if mean_brightness < DARK_IMAGE_MEAN_THRESHOLD:
        warnings.append("Image appears extremely dark; image quality may reduce prediction reliability.")
    elif mean_brightness > BRIGHT_IMAGE_MEAN_THRESHOLD:
        warnings.append("Image appears extremely bright; image quality may reduce prediction reliability.")

    return ImageQuality(
        is_valid=True,
        warnings=warnings,
        width=width,
        height=height,
        mean_brightness=mean_brightness,
    )


def preprocess_for_mobilenet_v2(image: Image.Image) -> np.ndarray:
    rgb_image = ensure_rgb_image(image)
    resized = rgb_image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32)
    image_array = (image_array / 127.5) - 1.0
    return np.expand_dims(image_array, axis=0)

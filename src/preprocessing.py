from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image, ImageOps, ImageStat

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
    laplacian_variance: float
    brightness_status: str
    sharpness_status: str
    suitability: str


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
        brightness_status = "Too dark"
        warnings.append("Image appears extremely dark; image quality may reduce prediction reliability.")
    elif mean_brightness > BRIGHT_IMAGE_MEAN_THRESHOLD:
        brightness_status = "Too bright"
        warnings.append("Image appears extremely bright; image quality may reduce prediction reliability.")
    else:
        brightness_status = "Good"

    # OpenCV's variance-of-Laplacian is used when available.  The small NumPy
    # fallback keeps the deployment usable if OpenCV is intentionally omitted.
    gray = np.asarray(grayscale, dtype=np.float32)
    try:
        import cv2

        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except ImportError:
        padded = np.pad(gray, 1, mode="edge")
        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4 * gray
        )
        laplacian_variance = float(laplacian.var())

    if laplacian_variance < 25.0:
        sharpness_status = "Blurry"
        warnings.append("Image appears blurry; use a sharper photo for a more reliable prediction.")
    else:
        sharpness_status = "Good"

    suitability = "Suitable for prediction" if not warnings else "Use with caution"

    return ImageQuality(
        is_valid=True,
        warnings=warnings,
        width=width,
        height=height,
        mean_brightness=mean_brightness,
        laplacian_variance=laplacian_variance,
        brightness_status=brightness_status,
        sharpness_status=sharpness_status,
        suitability=suitability,
    )


def preprocess_for_mobilenet_v2(image: Image.Image, model_includes_preprocessing: bool = False) -> np.ndarray:
    rgb_image = ensure_rgb_image(image)
    # Letterbox to retain object proportions rather than stretching the image.
    resized = ImageOps.pad(rgb_image, IMAGE_SIZE, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    image_array = np.asarray(resized, dtype=np.float32)
    if not model_includes_preprocessing:
        image_array = (image_array / 127.5) - 1.0
    return np.expand_dims(image_array, axis=0)

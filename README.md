# AI Waste Classification System 

An AI-based waste classification application built around MobileNetV2, TensorFlow Lite, and Streamlit. The app classifies one dominant waste item into six categories: cardboard, glass, metal, paper, plastic, and trash.

## Problem and objectives

Waste is commonly mixed or incorrectly sorted because material identification
is inconvenient. This project demonstrates how deep learning and computer
vision can provide an image-based first-pass classification with transparent
confidence and disposal guidance. Its objectives are reliable single-label
classification, quality-aware user feedback, measurable evaluation, and a
deployment-ready dashboard.

## Features

- TensorFlow Lite deployment model for fast local inference
- Professional Streamlit dashboard with classification, analytics, performance, history, guide, and about pages
- High, medium, and low confidence handling
- Top-two margin checks for uncertain predictions
- Aspect-ratio-preserving preprocessing with RGB conversion and MobileNetV2
  preprocessing applied exactly once by the model
- Image-quality checks for small, dark, bright, and blurry images (variance of
  Laplacian through OpenCV when installed)
- SQLite prediction history with CSV export
- Validation and real-world evaluation scripts that save reports under `reports/`

## Architecture

- `app.py`: Streamlit user interface
- `src/predictor.py`: reusable TFLite prediction engine
- `src/preprocessing.py`: image validation and MobileNetV2 preprocessing
- `src/analytics.py`: local SQLite prediction history
- `src/model.py`: MobileNetV2 V3 training architecture
- `scripts/train_v3.py`: two-stage training
- `scripts/evaluate_model.py`: validation evaluation
- `scripts/evaluate_real_world.py`: unseen real-world evaluation
- `scripts/convert_to_tflite.py`: Keras-to-TFLite conversion and shape verification

## Dataset Structure

Training and validation images should be arranged as:

```text
data/train/cardboard
data/train/glass
data/train/metal
data/train/paper
data/train/plastic
data/train/trash
data/validation/cardboard
data/validation/glass
data/validation/metal
data/validation/paper
data/validation/plastic
data/validation/trash
```

Optional additional real-world training images can be placed in `real_dataset/<class>/`. Completely unseen real-world test images should be placed in `real_test/<class>/`. Do not mix `real_test` images into training.

## Model

V3 uses MobileNetV2 with ImageNet weights, `224 x 224 x 3` input, controlled training augmentation, global average pooling, dropout, a compact dense head, and a six-class softmax output.

The class order is always:

```text
cardboard, glass, metal, paper, plastic, trash
```

The currently deployed `waste_classifier_v3.tflite` produced **86.22%
validation accuracy** on the saved 508-image validation evaluation, with macro
F1 of **82.66%**. It is deliberately not described as perfectly accurate.
`reports/model_comparison.csv` records comparisons against the available
models; V3 ties the deployed MobileNet export and has the highest saved
validation accuracy, while V2 has a slightly higher macro F1. Model selection
should be revisited after collecting more real-world data, especially for
trash.

## Methodology and evaluation

Training uses controlled realistic augmentation, class weighting, a frozen
MobileNetV2 training stage, fine-tuning of upper layers, early stopping,
checkpointing, and learning-rate reduction. Validation, real-world evaluation,
confusion matrices, per-class precision/recall/F1, and model comparisons are
saved under `reports/`. Keep training, validation, and unseen real-world test
images separate.

## Run Locally

Install deployment dependencies:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The app uses `models/waste_classifier_v3.tflite` and `models/class_names.json`.

## Train

Training requires TensorFlow and scikit-learn in addition to the deployment requirements:

```bash
pip install tensorflow scikit-learn
python scripts/train_v3.py
python scripts/convert_to_tflite.py
```

The training script calculates class distribution and controlled class weights dynamically from the directories.

## Evaluate

Validation evaluation:

```bash
python scripts/evaluate_model.py
```

Real-world evaluation:

```bash
python scripts/evaluate_real_world.py
```

Reports are written to `reports/`, including classification reports, confusion matrices, real-world predictions, incorrect predictions, and low-confidence/uncertain cases.

## Limitations

This is a single-label classifier for one dominant waste category. It is not an object detector and cannot reliably identify multiple waste objects in one image. Mixed-waste images may produce uncertain results. Recycling rules vary by location, so app recommendations are general guidance only.

Images containing multiple waste materials may require an object-detection or
segmentation model for item-level classification.

## Deployment

The TFLite model and class-name file are lightweight deployment artifacts. For
Streamlit Community Cloud or another Python host, install `requirements.txt`,
include `models/waste_classifier_v3.tflite`, and launch `streamlit run app.py`.
Do not deploy local prediction-history databases or private datasets.

## Future Improvements

- Collect more varied real-world images
- Add camera-based capture on supported deployments
- Add YOLO or another object detector for mixed-waste scenes
- Add calibration analysis for confidence reliability
- Track model versions alongside prediction history

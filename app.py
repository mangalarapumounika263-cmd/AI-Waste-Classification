import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "waste_classifier_mobilenet.tflite"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

    .prediction {
        font-size: 32px;
        font-weight: 700;
        text-align: center;
    }

    .confidence {
        font-size: 20px;
        text-align: center;
    }

    .info-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_classes():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class file not found: {CLASS_PATH}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    interpreter = Interpreter(
        model_path=str(MODEL_PATH)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing:
    # RGB [0,255] -> [-1,1]
    image_array = (image_array / 127.5) - 1.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict(image):
    interpreter, input_details, output_details = load_model()

    input_data = preprocess_image(image)

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    interpreter.set_tensor(
        input_index,
        input_data
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_index
    )

    predictions = np.asarray(
        predictions[0],
        dtype=np.float32
    )

    # Some TFLite models may return logits.
    # Convert to probabilities if required.
    if (
        np.any(predictions < 0)
        or not np.isclose(
            np.sum(predictions),
            1.0,
            atol=0.05
        )
    ):
        exp_predictions = np.exp(
            predictions - np.max(predictions)
        )
        predictions = (
            exp_predictions /
            np.sum(exp_predictions)
        )

    return predictions


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("♻️ About")

    st.write(
        """
        This application uses a trained MobileNetV2
        deep-learning model to classify waste images
        into six categories.
        """
    )

    st.subheader("Supported Categories")

    classes = load_classes()

    for class_name in classes:
        st.write(f"• {class_name.title()}")

    st.divider()

    st.subheader("Model")

    st.write("MobileNetV2")
    st.write("Input: 224 × 224 RGB")
    st.write("Output: 6 classes")


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">♻️ AI Waste Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a waste image and let the AI classify it'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a waste image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an image to start waste classification."
    )

    st.markdown(
        """
        ### How it works

        1. Upload a waste image.
        2. The image is resized to 224 × 224.
        3. MobileNetV2 analyzes the image.
        4. The application displays the predicted category.
        5. All six prediction probabilities are shown.
        """
    )

else:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    with col2:

        st.subheader("AI Analysis")

        with st.spinner("Analyzing image..."):

            try:

                predictions = predict(image)

                classes = load_classes()

                if len(predictions) != len(classes):
                    st.error(
                        "Model output does not match "
                        "the number of classes."
                    )
                    st.stop()

                sorted_indices = np.argsort(
                    predictions
                )[::-1]

                top_index = sorted_indices[0]

                predicted_class = classes[
                    top_index
                ]

                confidence = float(
                    predictions[top_index] * 100
                )

                second_confidence = float(
                    predictions[
                        sorted_indices[1]
                    ] * 100
                )

                # ------------------------------------------------
                # MAIN RESULT
                # ------------------------------------------------

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="prediction">'
                    f'{predicted_class.title()}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="confidence">'
                    f'Confidence: {confidence:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # CONFIDENCE INTERPRETATION
                # ------------------------------------------------

                if confidence >= 80:

                    st.success(
                        "High-confidence prediction."
                    )

                elif confidence >= 60:

                    st.warning(
                        "Moderate-confidence prediction. "
                        "The image may contain visual "
                        "features shared with other waste types."
                    )

                else:

                    st.warning(
                        "Low-confidence prediction. "
                        "The model is uncertain. "
                        "Try a clearer image containing "
                        "one dominant waste item."
                    )

                # ------------------------------------------------
                # ALL PREDICTIONS
                # ------------------------------------------------

                st.subheader(
                    "Prediction Probabilities"
                )

                for index in sorted_indices:

                    class_name = classes[index]

                    probability = float(
                        predictions[index] * 100
                    )

                    st.write(
                        f"**{class_name.title()}** "
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        min(
                            probability / 100,
                            1.0
                        )
                    )

                # ------------------------------------------------
                # TOP TWO
                # ------------------------------------------------

                st.subheader(
                    "Prediction Analysis"
                )

                st.write(
                    f"**Top prediction:** "
                    f"{predicted_class.title()} "
                    f"({confidence:.2f}%)"
                )

                st.write(
                    f"**Second prediction:** "
                    f"{classes[sorted_indices[1]].title()} "
                    f"({second_confidence:.2f}%)"
                )

                difference = (
                    confidence -
                    second_confidence
                )

                st.write(
                    f"**Difference:** "
                    f"{difference:.2f} percentage points"
                )

                if difference < 10:

                    st.warning(
                        "The top two classes are close. "
                        "The prediction should be treated "
                        "as uncertain."
                    )

                # ------------------------------------------------
                # RECOMMENDATION
                # ------------------------------------------------

                st.subheader(
                    "♻️ Recommendation"
                )

                recommendations = {
                    "cardboard":
                        "Place cardboard in the paper/cardboard recycling stream.",

                    "glass":
                        "Place clean glass in the appropriate glass recycling stream.",

                    "metal":
                        "Place metal containers in the metal recycling stream.",

                    "paper":
                        "Place clean and dry paper in paper recycling.",

                    "plastic":
                        "Check the local recycling rules before placing plastic in recycling.",

                    "trash":
                        "Dispose of non-recyclable or contaminated waste in the general waste stream."
                }

                st.info(
                    recommendations.get(
                        predicted_class,
                        "Follow local waste-management guidelines."
                    )
                )

            except Exception as error:

                st.error(
                    "Unable to analyze the image."
                )

                st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Waste Classification System • "
    "MobileNetV2 • TensorFlow Lite"
)
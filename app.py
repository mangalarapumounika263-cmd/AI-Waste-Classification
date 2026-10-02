from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    TFLITE_MODEL_PATH,
    KERAS_MODEL_PATH,
)

from src.predictor import WastePredictor

try:
    from src.history import record_prediction, load_history
    HISTORY_AVAILABLE = True
except Exception:
    HISTORY_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Based Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .prediction-box {
            padding: 22px;
            border-radius: 15px;
            border: 1px solid #ddd;
            background: #f8f9fa;
            margin-bottom: 20px;
        }

        .big-result {
            font-size: 34px;
            font-weight: 800;
        }

        .confidence {
            font-size: 22px;
            font-weight: 600;
        }

        .section-card {
            padding: 18px;
            border-radius: 14px;
            border: 1px solid #e1e1e1;
            background: white;
            margin-bottom: 15px;
        }

        .tip-card {
            padding: 15px;
            border-radius: 12px;
            background: #f5f7f8;
            margin: 8px 0;
        }

        .warning-card {
            padding: 15px;
            border-radius: 12px;
            background: #fff7e6;
            border: 1px solid #f0c36d;
            margin: 10px 0;
        }

        .success-card {
            padding: 15px;
            border-radius: 12px;
            background: #eef8f0;
            border: 1px solid #a9d5b1;
            margin: 10px 0;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e5e5e5;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_image" not in st.session_state:
    st.session_state.prediction_image = None

if "last_source" not in st.session_state:
    st.session_state.last_source = None


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def get_predictor():
    """
    Load the existing TFLite waste classification model.

    WastePredictor expects a Path object, not a string.
    """
    return WastePredictor(TFLITE_MODEL_PATH)


# ============================================================
# HELPERS
# ============================================================

def pretty_name(name):
    return str(name).replace("_", " ").title()


def get_confidence_status(confidence):
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


def confidence_message(confidence):
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "The model has relatively strong confidence in this prediction."
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "The model has moderate confidence. Check the top alternatives before disposing."
    return "The model has low confidence. Try a clearer image containing one main waste item."


def get_category_info(category):
    information = {
        "cardboard": {
            "description": "Paper-based packaging material such as boxes and cartons.",
            "bin": "Paper/cardboard recycling",
            "tips": [
                "Flatten boxes before recycling when possible.",
                "Remove excessive plastic wrapping and tape.",
                "Keep cardboard reasonably dry and clean.",
                "Pizza boxes with heavy food contamination may not be accepted everywhere.",
            ],
            "environment": "Recycling cardboard reduces demand for new paper fibres and can save landfill space.",
        },
        "glass": {
            "description": "Glass containers and other recyclable glass material.",
            "bin": "Glass recycling",
            "tips": [
                "Separate glass from ordinary household waste where collection exists.",
                "Empty and rinse food or beverage containers.",
                "Do not mix broken glass with recyclable containers unless your local system allows it.",
                "Ceramics, mirrors and heat-resistant glass may require different disposal routes.",
            ],
            "environment": "Glass can often be recycled repeatedly without losing its basic material properties.",
        },
        "metal": {
            "description": "Metal packaging and other recyclable metal items.",
            "bin": "Metal recycling",
            "tips": [
                "Empty containers before recycling.",
                "Rinse food residue when practical.",
                "Separate hazardous or electronic components from ordinary metal recycling.",
                "Compress cans only if your local recycling facility recommends doing so.",
            ],
            "environment": "Recycling metals can reduce the need for extracting and processing new raw materials.",
        },
        "paper": {
            "description": "Paper-based material such as sheets, newspapers and some packaging.",
            "bin": "Paper recycling",
            "tips": [
                "Keep recyclable paper dry.",
                "Remove food-contaminated sections.",
                "Avoid mixing heavily laminated or wax-coated materials unless accepted locally.",
                "Reuse paper for notes or packing before recycling.",
            ],
            "environment": "Paper recycling can reduce the amount of waste sent to landfill and reduce demand for virgin fibre.",
        },
        "plastic": {
            "description": "Plastic packaging and other plastic waste.",
            "bin": "Plastic recycling",
            "tips": [
                "Check the recycling symbol and local collection rules.",
                "Empty and rinse containers when appropriate.",
                "Avoid placing plastic film in ordinary recycling unless your local system accepts it.",
                "Reuse suitable containers before recycling them.",
            ],
            "environment": "Correct sorting helps recyclable plastic reach appropriate recovery streams instead of being mixed with general waste.",
        },
        "trash": {
            "description": "General waste that does not clearly belong to the recyclable categories.",
            "bin": "General waste",
            "tips": [
                "Check whether the item has a dedicated collection route before sending it to general waste.",
                "Separate batteries, electronics and hazardous materials.",
                "Avoid mixing recyclable materials into general waste when separate collection is available.",
                "Consider reuse or repair before disposal.",
            ],
            "environment": "Reducing general waste helps conserve resources and reduces pressure on landfill and waste-treatment systems.",
        },
    }

    return information.get(
        category.lower(),
        {
            "description": "Waste material classified by the AI model.",
            "bin": "Check your local waste rules",
            "tips": [
                "Check your local recycling instructions.",
                "Keep recyclable materials separated from food and hazardous waste.",
            ],
            "environment": "Correct waste separation supports better recovery and disposal.",
        },
    )


def calculate_prediction_metrics(result):
    probabilities = np.asarray(result.probabilities, dtype=float)

    order = np.argsort(probabilities)[::-1]

    top1_index = int(order[0])
    top2_index = int(order[1])

    top1 = float(probabilities[top1_index])
    top2 = float(probabilities[top2_index])

    gap = top1 - top2

    entropy = float(
        -np.sum(
            probabilities
            * np.log(np.clip(probabilities, 1e-8, 1.0))
        )
    )

    return {
        "top1": top1,
        "top2": top2,
        "gap": gap,
        "entropy": entropy,
    }


def make_analysis(result):
    metrics = calculate_prediction_metrics(result)

    predicted = pretty_name(result.predicted_class)
    confidence = metrics["top1"]
    second_probability = metrics["top2"]
    gap = metrics["gap"]

    analysis = []

    analysis.append(
        f"The model classified the uploaded image primarily as **{predicted}**."
    )

    analysis.append(
        f"The highest model probability is **{confidence:.1%}**."
    )

    if gap < 0.10:
        analysis.append(
            "The first and second predictions are close, so the image should be treated as uncertain."
        )
    elif gap < 0.20:
        analysis.append(
            "There is a noticeable but not large separation between the first and second predictions."
        )
    else:
        analysis.append(
            "There is a clear separation between the first and second predictions."
        )

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        analysis.append(
            "The model's confidence is in the high-confidence range."
        )
    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        analysis.append(
            "The model's confidence is in the medium range."
        )
    else:
        analysis.append(
            "The model's confidence is low, so a clearer image is recommended."
        )

    return analysis


def render_probability_chart(result):
    probabilities = result.probabilities

    df = pd.DataFrame(
        {
            "Category": [pretty_name(x) for x in CLASS_NAMES],
            "Probability": probabilities,
        }
    )

    df = df.sort_values("Probability", ascending=True)

    st.bar_chart(
        df.set_index("Category"),
        y="Probability",
    )


def save_history_safely(result):
    if not HISTORY_AVAILABLE:
        return False

    try:
        record_prediction(
            result.predicted_class,
            result.confidence,
        )
        return True
    except Exception:
        return False


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("♻️ AI Waste Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Classify Waste",
        "📊 Analysis",
        "📈 Model Insights",
        "🕘 Prediction History",
        "♻️ Waste Guide",
        "🌱 Eco Tips",
        "🌍 Environmental Impact",
        "⚙️ How It Works",
        "ℹ️ About Project",
    ],
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">AI-Based Waste Classification</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Classify waste images using an AI-powered image classification model.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Waste Categories", len(CLASS_NAMES))

    with col2:
        st.metric("Image Size", f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}")

    with col3:
        st.metric("Inference Model", "TFLite")

    st.markdown("---")

    st.subheader("♻️ What this application does")

    st.write(
        """
        This application uses a trained image-classification model to identify
        common waste categories from an uploaded image or camera capture.

        The system provides:
        - predicted waste category
        - confidence level
        - top alternative predictions
        - probability distribution
        - image-quality warnings
        - disposal guidance
        - prediction history
        - environmental information
        """
    )

    st.info(
        "For the most reliable prediction, photograph one main waste item against a clear background."
    )

    st.subheader("Supported Categories")

    cols = st.columns(3)

    for i, category in enumerate(CLASS_NAMES):
        with cols[i % 3]:
            st.markdown(
                f"### {pretty_name(category)}"
            )
            st.write(get_category_info(category)["description"])


# ============================================================
# CLASSIFY WASTE
# ============================================================

elif page == "🔍 Classify Waste":

    st.title("🔍 Analyze Waste")

    st.write(
        "Choose an image source first. The camera will only open after you select Camera."
    )

    source = st.radio(
        "Choose image source",
        ["📁 Upload Image", "📷 Camera"],
        horizontal=True,
    )

    image = None

    if source == "📁 Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.last_source = "Upload"

    else:

        camera_file = st.camera_input(
            "Take a picture of the waste item"
        )

        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")
            st.session_state.last_source = "Camera"

    if image is not None:

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Selected waste image",
                use_container_width=True,
            )

        with col2:
            st.subheader("Image Details")

            st.write(f"**Width:** {image.width}px")
            st.write(f"**Height:** {image.height}px")
            st.write(f"**Source:** {st.session_state.last_source}")

            if image.width < 96 or image.height < 96:
                st.warning(
                    "This image is very small. A larger image may produce a more reliable result."
                )

        st.markdown("---")

        analyze = st.button(
            "🔍 Analyze Waste",
            type="primary",
            use_container_width=True,
        )

        if analyze:

            with st.spinner("Analyzing image..."):

                try:
                    predictor = get_predictor()

                    # IMPORTANT:
                    # WastePredictor.predict() returns PredictionResult.
                    result = predictor.predict(image)

                    st.session_state.prediction_result = result
                    st.session_state.prediction_image = image

                    saved = save_history_safely(result)

                    st.success("Waste analysis completed.")

                    if saved:
                        st.caption("Prediction saved to history.")

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

        result = st.session_state.prediction_result

        if result is not None:

            st.markdown("---")

            st.subheader("🎯 Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Waste",
                    pretty_name(result.predicted_class),
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{result.confidence:.1%}",
                )

            with col3:
                st.metric(
                    "Confidence Level",
                    result.confidence_level,
                )

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="big-result">
                        {pretty_name(result.predicted_class)}
                    </div>
                    <div class="confidence">
                        Confidence: {result.confidence:.1%}
                    </div>
                    <p>{confidence_message(result.confidence)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result.uncertainty:
                st.warning(result.uncertainty)

            st.subheader("📊 Top Predictions")

            top_df = pd.DataFrame(result.top_3_predictions)

            top_df["class_name"] = top_df["class_name"].apply(pretty_name)
            top_df["probability"] = top_df["probability"].map(
                lambda x: f"{x:.2%}"
            )

            top_df.columns = ["Category", "Probability"]

            st.table(top_df)

            st.subheader("📈 Complete Probability Distribution")

            render_probability_chart(result)

            metrics = calculate_prediction_metrics(result)

            st.subheader("🔎 Prediction Strength")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Top Prediction",
                    f"{metrics['top1']:.1%}",
                )

            with c2:
                st.metric(
                    "Second Prediction",
                    f"{metrics['top2']:.1%}",
                )

            with c3:
                st.metric(
                    "Top-1 vs Top-2 Gap",
                    f"{metrics['gap']:.1%}",
                )

            if metrics["gap"] < 0.10:
                st.warning(
                    "The top two classes are close. Treat this prediction cautiously."
                )

            st.subheader("🖼️ Image Quality")

            quality = result.image_quality

            q1, q2, q3 = st.columns(3)

            with q1:
                st.write(
                    f"**Brightness:** {quality.brightness_status}"
                )

            with q2:
                st.write(
                    f"**Sharpness:** {quality.sharpness_status}"
                )

            with q3:
                st.write(
                    f"**Suitability:** {quality.suitability}"
                )

            if result.quality_warnings:

                st.markdown(
                    '<div class="warning-card"><b>⚠️ Image warnings</b></div>',
                    unsafe_allow_html=True,
                )

                for warning in result.quality_warnings:
                    st.warning(warning)

            category_info = get_category_info(result.predicted_class)

            st.subheader("♻️ Disposal Guidance")

            st.success(
                f"Suggested waste stream: **{category_info['bin']}**"
            )

            st.write(category_info["description"])

            st.subheader("🌱 What you can do")

            for tip in category_info["tips"]:
                st.markdown(
                    f"- {tip}"
                )

            st.subheader("🌍 Environmental Note")

            st.info(category_info["environment"])

            st.subheader("🧠 AI Analysis")

            for sentence in make_analysis(result):
                st.write(f"• {sentence}")

            st.caption(
                "The displayed probabilities are model outputs, not guaranteed real-world certainty."
            )

    else:

        st.info(
            "Select Upload Image or Camera and provide a waste image to begin."
        )


# ============================================================
# ANALYSIS
# ============================================================

elif page == "📊 Analysis":

    st.title("📊 Waste Analysis")

    result = st.session_state.prediction_result

    if result is None:

        st.info(
            "No prediction is available yet. Go to 🔍 Classify Waste and analyze an image first."
        )

    else:

        st.subheader("Current Analysis")

        metrics = calculate_prediction_metrics(result)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Classification",
                pretty_name(result.predicted_class),
            )

        with col2:
            st.metric(
                "Confidence",
                f"{result.confidence:.1%}",
            )

        with col3:
            st.metric(
                "Top-2 Gap",
                f"{metrics['gap']:.1%}",
            )

        with col4:
            st.metric(
                "Confidence Level",
                result.confidence_level,
            )

        st.markdown("---")

        st.subheader("🔬 Detailed Interpretation")

        for sentence in make_analysis(result):
            st.write(f"• {sentence}")

        if metrics["gap"] < 0.10:
            st.warning(
                "The model's two strongest classes are close together. "
                "Use a clearer image with one item and a simple background."
            )
        elif metrics["gap"] < 0.20:
            st.info(
                "The prediction has a moderate separation from the second-best class."
            )
        else:
            st.success(
                "The prediction has a relatively clear separation from the second-best class."
            )

        st.subheader("📈 Probability Distribution")

        render_probability_chart(result)

        st.subheader("🥇 Top 3 Candidates")

        top3 = pd.DataFrame(result.top_3_predictions)

        top3["Category"] = top3["class_name"].apply(pretty_name)
        top3["Probability"] = top3["probability"].map(
            lambda x: f"{x:.2%}"
        )

        st.dataframe(
            top3[["Category", "Probability"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("💡 How to Improve the Next Prediction")

        suggestions = [
            "Use a well-lit image.",
            "Keep the main waste item clearly visible.",
            "Avoid heavy shadows and reflections.",
            "Avoid extremely close or blurry photographs.",
            "Use a simple background where possible.",
            "If the object contains several materials, photograph the dominant material separately.",
        ]

        for suggestion in suggestions:
            st.write(f"✓ {suggestion}")

        if result.uncertainty:
            st.warning(result.uncertainty)


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "📈 Model Insights":

    st.title("📈 Model Insights")

    st.write(
        """
        This page focuses on the model that is actually being used by the
        application instead of displaying invented or unrelated performance
        numbers.
        """
    )

    st.subheader("🤖 Active Model")

    model_exists = Path(TFLITE_MODEL_PATH).exists()
    keras_exists = Path(KERAS_MODEL_PATH).exists()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Inference Format",
            "TFLite",
        )

    with c2:
        st.metric(
            "Input Size",
            f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}",
        )

    with c3:
        st.metric(
            "Output Classes",
            len(CLASS_NAMES),
        )

    st.markdown("---")

    st.subheader("📦 Model Files")

    st.write(
        f"**TFLite model:** `{Path(TFLITE_MODEL_PATH).name}`"
    )

    if model_exists:
        st.success("TFLite model found and available for inference.")
    else:
        st.error("TFLite model file was not found.")

    st.write(
        f"**Keras model:** `{Path(KERAS_MODEL_PATH).name}`"
    )

    if keras_exists:
        st.info("Keras training model is present.")
    else:
        st.warning("Keras model file was not found.")

    st.subheader("🏷️ Supported Classes")

    class_df = pd.DataFrame(
        {
            "Class": [pretty_name(x) for x in CLASS_NAMES],
            "Model Output Index": list(range(len(CLASS_NAMES))),
        }
    )

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🎯 Confidence Interpretation")

    confidence_df = pd.DataFrame(
        {
            "Range": [
                f"{HIGH_CONFIDENCE_THRESHOLD:.0%} and above",
                f"{MEDIUM_CONFIDENCE_THRESHOLD:.0%} – {HIGH_CONFIDENCE_THRESHOLD:.0%}",
                f"Below {MEDIUM_CONFIDENCE_THRESHOLD:.0%}",
            ],
            "Meaning": [
                "High confidence",
                "Medium confidence",
                "Low confidence",
            ],
        }
    )

    st.table(confidence_df)

    st.warning(
        """
        A high probability does not automatically mean the prediction is
        correct. Model confidence should be interpreted together with image
        quality and the difference between the top predictions.
        """
    )

    st.subheader("🧪 Important Model Limitation")

    st.write(
        """
        The model is trained on a fixed set of six waste categories. Real-world
        photographs can contain different lighting, backgrounds, camera angles,
        object combinations and materials that may not closely match the
        training images.

        Therefore, improving real-world accuracy requires evaluation on
        representative real images and, when necessary, better training data,
        preprocessing or fine-tuning.
        """
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "🕘 Prediction History":

    st.title("🕘 Prediction History")

    if not HISTORY_AVAILABLE:

        st.warning(
            "Prediction history module is not available in the current project."
        )

    else:

        try:
            history = load_history()

            if history is None:
                history = []

            if isinstance(history, pd.DataFrame):

                df = history.copy()

            else:

                df = pd.DataFrame(history)

            if df.empty:

                st.info(
                    "No predictions have been recorded yet."
                )

            else:

                st.subheader("📋 Recorded Predictions")

                # Make column names easier to read without assuming
                # one exact database schema.
                display_df = df.copy()

                display_df.columns = [
                    str(column).replace("_", " ").title()
                    for column in display_df.columns
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("---")

                st.subheader("📊 History Summary")

                # Try to identify common column names.
                lower_columns = {
                    str(column).lower(): column
                    for column in df.columns
                }

                category_column = None
                confidence_column = None

                for name in [
                    "predicted_class",
                    "class_name",
                    "category",
                    "prediction",
                ]:
                    if name in lower_columns:
                        category_column = lower_columns[name]
                        break

                for name in [
                    "confidence",
                    "probability",
                ]:
                    if name in lower_columns:
                        confidence_column = lower_columns[name]
                        break

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Total Predictions",
                        len(df),
                    )

                with c2:

                    if confidence_column is not None:

                        numeric_conf = pd.to_numeric(
                            df[confidence_column],
                            errors="coerce",
                        )

                        if numeric_conf.notna().any():

                            avg_conf = numeric_conf.mean()

                            if avg_conf <= 1:
                                avg_conf_text = f"{avg_conf:.1%}"
                            else:
                                avg_conf_text = f"{avg_conf:.1f}%"

                            st.metric(
                                "Average Confidence",
                                avg_conf_text,
                            )
                        else:
                            st.metric(
                                "Average Confidence",
                                "N/A",
                            )

                    else:
                        st.metric(
                            "Average Confidence",
                            "N/A",
                        )

                with c3:

                    if category_column is not None:

                        counts = df[category_column].value_counts()

                        if not counts.empty:
                            st.metric(
                                "Most Predicted",
                                pretty_name(counts.index[0]),
                            )
                        else:
                            st.metric(
                                "Most Predicted",
                                "N/A",
                            )

                    else:
                        st.metric(
                            "Most Predicted",
                            "N/A",
                        )

                if category_column is not None:

                    st.subheader("📈 Predictions by Category")

                    category_counts = (
                        df[category_column]
                        .astype(str)
                        .value_counts()
                        .rename_axis("Category")
                        .reset_index(name="Predictions")
                    )

                    category_counts["Category"] = category_counts[
                        "Category"
                    ].apply(pretty_name)

                    st.bar_chart(
                        category_counts.set_index("Category")
                    )

        except Exception as error:

            st.error(
                f"Could not load prediction history: {error}"
            )

            st.info(
                "The classifier itself can still work even if the history database has a separate issue."
            )


# ============================================================
# WASTE GUIDE
# ============================================================

elif page == "♻️ Waste Guide":

    st.title("♻️ Waste Guide")

    st.write(
        "Select a category to see practical handling and disposal guidance."
    )

    category = st.selectbox(
        "Choose waste category",
        CLASS_NAMES,
    )

    info = get_category_info(category)

    st.subheader(pretty_name(category))

    st.write(info["description"])

    st.success(
        f"Suggested waste stream: {info['bin']}"
    )

    st.subheader("Recommended Practices")

    for tip in info["tips"]:
        st.write(f"✓ {tip}")

    st.subheader("Environmental Information")

    st.info(info["environment"])

    st.caption(
        "Always follow the waste-separation rules provided by your local municipality or collection service."
    )


# ============================================================
# ECO TIPS
# ============================================================

elif page == "🌱 Eco Tips":

    st.title("🌱 Eco Tips")

    tips = [
        ("♻️ Reduce", "Avoid buying products with unnecessary packaging."),
        ("🔄 Reuse", "Reuse containers, boxes and bags when safe and practical."),
        ("🛠️ Repair", "Repair usable products instead of replacing them immediately."),
        ("📦 Flatten", "Flatten cardboard boxes to save collection and storage space."),
        ("🧴 Clean", "Empty and rinse recyclable containers when local rules recommend it."),
        ("🗂️ Separate", "Keep different waste materials separated."),
        ("🚫 Avoid contamination", "Food contamination can make recyclable material harder to process."),
        ("🔋 Batteries", "Keep batteries separate from normal household waste."),
        ("💻 Electronics", "Use authorised e-waste collection channels for electronic products."),
        ("🍃 Compost", "Compost suitable organic waste instead of mixing it with dry recyclables."),
        ("🛍️ Carry reusable bags", "Reduce single-use packaging by carrying reusable bags."),
        ("🥤 Reusable bottles", "Use reusable bottles where practical."),
        ("📄 Save paper", "Use digital documents when appropriate and reuse paper before recycling."),
        ("🚰 Reduce waste", "Buy only the amount of food and products you expect to use."),
        ("🌍 Local rules", "Check your municipality's recycling instructions because collection rules differ by location."),
    ]

    for title, description in tips:
        st.markdown(
            f"""
            <div class="tip-card">
                <b>{title}</b><br>
                {description}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ENVIRONMENTAL IMPACT
# ============================================================

elif page == "🌍 Environmental Impact":

    st.title("🌍 Environmental Impact")

    st.write(
        """
        Correct waste classification is one part of a larger waste-management
        process. Separating materials can make recycling and recovery easier
        when suitable collection systems are available.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("♻️ Benefits of Separation")

        benefits = [
            "Improves sorting at the source.",
            "Helps recyclable materials enter appropriate recovery streams.",
            "Reduces unnecessary mixing of different waste types.",
            "Can reduce the amount of recyclable material sent to general waste.",
            "Encourages awareness of responsible disposal.",
        ]

        for benefit in benefits:
            st.write(f"✓ {benefit}")

    with col2:

        st.subheader("🌱 Sustainable Habits")

        habits = [
            "Reduce consumption where possible.",
            "Reuse products before disposal.",
            "Repair usable items.",
            "Separate recyclable materials.",
            "Dispose of hazardous and electronic waste through appropriate channels.",
        ]

        for habit in habits:
            st.write(f"✓ {habit}")

    st.info(
        "The environmental benefit depends on local collection, recycling and waste-treatment infrastructure."
    )


# ============================================================
# HOW IT WORKS
# ============================================================

elif page == "⚙️ How It Works":

    st.title("⚙️ How It Works")

    st.write(
        "The application follows a simple image-classification pipeline."
    )

    steps = [
        (
            "1️⃣",
            "Choose Image Source",
            "The user selects either an uploaded image or the camera."
        ),
        (
            "2️⃣",
            "Provide Waste Image",
            "The image is loaded into the application."
        ),
        (
            "3️⃣",
            "Image Quality Check",
            "The system checks basic image characteristics such as brightness and sharpness."
        ),
        (
            "4️⃣",
            "Preprocessing",
            "The image is prepared for the model using the project's configured preprocessing pipeline."
        ),
        (
            "5️⃣",
            "AI Classification",
            "The TFLite model produces probabilities for the six supported waste classes."
        ),
        (
            "6️⃣",
            "Prediction Analysis",
            "The application identifies the highest probability class and compares it with alternatives."
        ),
        (
            "7️⃣",
            "Confidence & Uncertainty",
            "The application reports confidence and warns when predictions are close or uncertain."
        ),
        (
            "8️⃣",
            "Waste Guidance",
            "The predicted category is connected to practical disposal and environmental guidance."
        ),
        (
            "9️⃣",
            "Prediction History",
            "The prediction can be recorded so previous classifications can be reviewed."
        ),
    ]

    for symbol, title, description in steps:

        st.markdown(
            f"""
            <div class="section-card">
                <h3>{symbol} {title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("🧠 Model Pipeline")

    st.code(
        """
Image
  ↓
Quality Check
  ↓
Preprocessing
  ↓
TFLite Model
  ↓
6 Class Probabilities
  ↓
Top Prediction
  ↓
Confidence + Uncertainty
  ↓
Waste Guidance
  ↓
Prediction History
        """,
        language="text",
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.subheader("AI-Based Waste Classification")

    st.write(
        """
        AI-Based Waste Classification is a computer-vision project designed
        to classify common waste materials from images.

        The application combines image preprocessing, a trained image
        classification model, confidence analysis and waste-management
        guidance in a Streamlit interface.
        """
    )

    st.subheader("🎯 Project Objectives")

    objectives = [
        "Classify common waste categories using computer vision.",
        "Provide an easy-to-use image analysis interface.",
        "Show model confidence and alternative predictions.",
        "Help users understand appropriate waste separation.",
        "Maintain a record of previous predictions.",
    ]

    for objective in objectives:
        st.write(f"✓ {objective}")

    st.subheader("🏷️ Waste Categories")

    for category in CLASS_NAMES:
        st.write(f"• {pretty_name(category)}")

    st.subheader("⚠️ Important Note")

    st.write(
        """
        This application is an AI-assisted classification system. Its output
        should be treated as a prediction rather than a guaranteed determination
        of material composition. Local waste-management rules should always be
        followed for final disposal decisions.
        """
    )

    st.caption(
        f"Model input resolution: {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}"
    )
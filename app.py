import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.analytics import clear_history, load_history, record_prediction
from src.config import CLASS_NAMES, REPORTS_DIR, TFLITE_MODEL_PATH
from src.predictor import WastePredictor
from src.recommendations import WASTE_GUIDE

st.set_page_config(
    page_title="AI Waste Classification V3",
    page_icon="Recycle",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
    }
    .status-note {
        border-left: 4px solid #64748b;
        padding: 0.7rem 0.9rem;
        background: #f8fafc;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor() -> WastePredictor:
    predictor = WastePredictor()
    predictor.load()
    return predictor


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def read_json_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def render_probability_bars(top_predictions: list[dict]) -> None:
    for item in top_predictions:
        probability = float(item["probability"])
        st.write(f"**{item['class_name'].title()}** {percent(probability)}")
        st.progress(min(max(probability, 0.0), 1.0))


def render_all_probability_bars(probabilities: list[float]) -> None:
    rows = sorted(
        zip(CLASS_NAMES, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    for class_name, probability in rows:
        st.write(f"**{class_name.title()}** {percent(float(probability))}")
        st.progress(min(max(float(probability), 0.0), 1.0))


def dashboard_page() -> None:
    st.title("AI Waste Classification Dashboard")
    st.caption("Intelligent waste recognition and responsible disposal guidance.")
    history = load_history()

    if history.empty:
        st.info("No predictions yet.")
        return

    total = len(history)
    avg_confidence = float(history["confidence"].mean())
    high_count = int((history["confidence_level"] == "High").sum())
    uncertain_count = int(history["uncertainty"].fillna("").astype(bool).sum())
    most_common = history["predicted_class"].mode().iloc[0]
    latest = history.iloc[0]

    cols = st.columns(5)
    low_count = int((history["confidence_level"] == "Low").sum())
    cols[0].metric("Total Predictions", total)
    cols[1].metric("Average Confidence", percent(avg_confidence))
    cols[2].metric("High Confidence", high_count)
    cols[3].metric("Low Confidence", low_count)
    cols[4].metric("Most Common Waste Type", most_common.title())

    st.markdown(f"Latest prediction: **{latest['predicted_class'].title()}** at {latest['timestamp']}")

    left, right = st.columns(2)
    with left:
        st.subheader("Category Distribution")
        st.bar_chart(history["predicted_class"].value_counts())
    with right:
        st.subheader("Confidence Distribution")
        st.bar_chart(history["confidence_level"].value_counts())

    st.subheader("Recent History")
    st.dataframe(
        history[["timestamp", "filename", "predicted_class", "confidence", "confidence_level", "uncertainty"]]
        .head(10),
        use_container_width=True,
        hide_index=True,
    )


def classify_page() -> None:
    st.title("Analyze Waste")
    if not TFLITE_MODEL_PATH.exists():
        st.error(f"Deployment model is missing: {TFLITE_MODEL_PATH}")
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

    if uploaded is None:
        st.info("Upload a waste image containing one dominant item.")
        return

    try:
        image = Image.open(uploaded)
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
        return

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)

    if not analyze:
        return

    with right:
        with st.spinner("Analyzing image..."):
            try:
                result = get_predictor().predict(image)
            except Exception as error:
                st.error(f"Unable to analyze this image: {error}")
                return

        status_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[result.confidence_level]
        st.metric("Prediction", result.predicted_class.upper())
        st.metric("Confidence", percent(result.confidence))
        st.metric("Confidence status", f"{status_icon} {result.confidence_level} confidence")

        if result.quality_warnings:
            for warning in result.quality_warnings:
                st.warning(warning)

        if result.uncertainty:
            st.warning(result.uncertainty)
            st.warning(
                "This model is designed to classify one dominant waste category. "
                "Images containing multiple waste objects may produce uncertain results."
            )
        elif result.confidence_level == "High":
            st.success("High-confidence prediction.")
        else:
            st.warning("Moderate confidence: image may contain mixed or unclear materials.")

        st.subheader("Probability Chart")
        render_all_probability_bars(result.probabilities)

        st.subheader("Top 3 Predictions")
        render_probability_bars(result.top_3_predictions)

        st.subheader("Why this result?")
        st.write(
            "The production model selected the class with the highest output probability. "
            "Review the alternatives when confidence is medium, low, or marked uncertain."
        )

        st.subheader("Recommendation")
        st.info(result.recommendation)

        record_prediction(
            filename=uploaded.name,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            confidence_level=result.confidence_level,
            uncertainty=result.uncertainty,
            top_3_predictions=result.top_3_predictions,
        )


def analytics_page() -> None:
    st.title("Analytics")
    history = load_history()
    if history.empty:
        st.info("No predictions yet.")
        return

    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    st.subheader("Predictions by Category")
    st.bar_chart(history["predicted_class"].value_counts())

    st.subheader("Average Confidence by Category")
    st.bar_chart(history.groupby("predicted_class")["confidence"].mean())

    st.subheader("Confidence Level Distribution")
    st.bar_chart(history["confidence_level"].value_counts())

    st.subheader("Recent Prediction Trend")
    trend = history.dropna(subset=["timestamp"]).set_index("timestamp").resample("D").size()
    st.line_chart(trend)


def model_performance_page() -> None:
    st.title("Model Insights")
    validation_report = read_json_report(REPORTS_DIR / "classification_report.json")
    real_report = read_json_report(REPORTS_DIR / "real_world_classification_report.json")

    st.write("Model name: `waste_classifier_v3.tflite`")
    st.write("Model type: MobileNetV2 transfer-learning classifier converted to TensorFlow Lite")
    st.write("Input size: `224 x 224 x 3`")
    st.write(f"Number of classes: `{len(CLASS_NAMES)}`")
    st.write("Class names: " + ", ".join(name.title() for name in CLASS_NAMES))

    if validation_report is None:
        st.info("Validation evaluation has not been performed yet.")
    else:
        cols = st.columns(4)
        cols[0].metric("Validation accuracy", percent(validation_report.get("accuracy", 0.0)))
        cols[1].metric("Macro precision", percent(validation_report["macro avg"]["precision"]))
        cols[2].metric("Macro recall", percent(validation_report["macro avg"]["recall"]))
        cols[3].metric("Macro F1", percent(validation_report["macro avg"]["f1-score"]))

        per_class = pd.DataFrame(
            [
                {
                    "class": name,
                    "precision": validation_report[name]["precision"],
                    "recall": validation_report[name]["recall"],
                    "f1": validation_report[name]["f1-score"],
                    "support": validation_report[name]["support"],
                }
                for name in CLASS_NAMES
                if name in validation_report
            ]
        )
        st.subheader("Per-Class Validation Performance")
        st.dataframe(per_class, use_container_width=True, hide_index=True)

    matrix_path = REPORTS_DIR / "confusion_matrix.csv"
    if matrix_path.exists():
        st.subheader("Validation Confusion Matrix")
        st.dataframe(pd.read_csv(matrix_path, index_col=0), use_container_width=True)

    st.subheader("Real-World Performance")
    if real_report is None:
        st.info("Real-world evaluation has not been performed yet.")
    else:
        cols = st.columns(4)
        report = real_report["report"]
        cols[0].metric("Real-world accuracy", percent(real_report.get("accuracy", 0.0)))
        cols[1].metric("Macro precision", percent(report["macro avg"]["precision"]))
        cols[2].metric("Macro recall", percent(report["macro avg"]["recall"]))
        cols[3].metric("Macro F1", percent(report["macro avg"]["f1-score"]))
        predictions_path = REPORTS_DIR / "real_world_predictions.csv"
        if predictions_path.exists():
            st.dataframe(pd.read_csv(predictions_path), use_container_width=True, hide_index=True)


def history_page() -> None:
    st.title("Prediction History")
    history = load_history()
    if history.empty:
        st.info("No predictions yet.")
        return

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear history"):
            clear_history()
            st.rerun()
    with col2:
        csv = history.to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", csv, "prediction_history.csv", "text/csv")

    st.dataframe(history, use_container_width=True, hide_index=True)


def waste_guide_page() -> None:
    st.title("Recycling Guide")
    st.caption("General guidance only. Recycling rules vary by location.")
    for class_name in CLASS_NAMES:
        guide = WASTE_GUIDE[class_name]
        with st.expander(class_name.title(), expanded=False):
            st.write("Examples: " + ", ".join(guide["examples"]))
            st.write("Recycling: " + guide["recycling"])
            st.write("Contamination: " + guide["contamination"])
            st.write("Disposal: " + guide["disposal"])


def about_page() -> None:
    st.title("About")
    st.write(
        "This V3 system uses a MobileNetV2 TensorFlow Lite classifier for six single-label "
        "waste categories: cardboard, glass, metal, paper, plastic, and trash."
    )
    st.write(
        "The classifier predicts one dominant category. It is not an object detector and does "
        "not locate multiple items in an image. YOLO-style object detection would be a suitable "
        "future direction for mixed-waste scenes."
    )
    st.write(f"Deployment model: `{TFLITE_MODEL_PATH}`")


PAGES = {
    "Dashboard": dashboard_page,
    "Classify Waste": classify_page,
    "Analytics": analytics_page,
    "Model Performance": model_performance_page,
    "Prediction History": history_page,
    "Waste Guide": waste_guide_page,
    "About": about_page,
}

with st.sidebar:
    st.header("Navigation")
    labels = {
        "Dashboard": "🏠 Dashboard",
        "Classify Waste": "🔍 Analyze Waste",
        "Analytics": "📊 Analytics",
        "Model Performance": "🧠 Model Insights",
        "Prediction History": "📜 Prediction History",
        "Waste Guide": "♻️ Recycling Guide",
        "About": "ℹ️ About",
    }
    selected = st.radio("Page", list(labels.values()), label_visibility="collapsed")
    page_name = next(key for key, value in labels.items() if value == selected)
    st.divider()
    st.caption("Model classes")
    st.write(", ".join(name.title() for name in CLASS_NAMES))

PAGES[page_name]()

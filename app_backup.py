import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/waste_classifier.tflite"
CLASS_PATH = "models/class_names.json"

# ============================================================
# WASTE INFORMATION
# ============================================================

WASTE_INFO = {
    "cardboard": {
        "icon": "📦",
        "title": "Cardboard",
        "type": "Recyclable",
        "disposal": "Flatten cardboard boxes and place them in the paper/cardboard recycling collection.",
        "tips": "Keep cardboard clean and dry. Remove excessive tape, plastic, and food contamination."
    },
    "glass": {
        "icon": "🍾",
        "title": "Glass",
        "type": "Recyclable",
        "disposal": "Place glass containers in the designated glass recycling collection.",
        "tips": "Rinse containers and handle broken glass carefully. Follow local recycling rules."
    },
    "metal": {
        "icon": "🥫",
        "title": "Metal",
        "type": "Recyclable",
        "disposal": "Place clean metal cans and containers in the appropriate recycling collection.",
        "tips": "Empty and rinse cans before recycling whenever possible."
    },
    "paper": {
        "icon": "📄",
        "title": "Paper",
        "type": "Recyclable",
        "disposal": "Place clean and dry paper in the paper recycling collection.",
        "tips": "Avoid mixing heavily contaminated or wet paper with recyclable paper."
    },
    "plastic": {
        "icon": "🧴",
        "title": "Plastic",
        "type": "Usually Recyclable",
        "disposal": "Check your local recycling rules and place accepted plastic containers in the recycling collection.",
        "tips": "Empty and rinse containers. Different areas accept different plastic types."
    },
    "trash": {
        "icon": "🗑️",
        "title": "Trash",
        "type": "General Waste",
        "disposal": "Place non-recyclable waste in the appropriate general waste collection.",
        "tips": "Before disposing, check whether the item can be reused, repaired, composted, or recycled."
    }
}

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as file:
        return json.load(file)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image, interpreter, input_details, output_details, class_names):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    # Model contains Rescaling layer, so keep pixel values 0-255.
    interpreter.set_tensor(
        input_details[0]["index"],
        image_array
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[predicted_index])

    return predicted_class, confidence, predictions


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("♻️ Waste AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📷 Classify Waste",
            "📊 Analytics",
            "🕐 History",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.caption("AI Waste Classification System")
    st.caption("6-Class Image Classification")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    interpreter, input_details, output_details = load_model()
    class_names = load_classes()

except Exception as e:

    st.error("Unable to load the AI model.")

    st.code(str(e))

    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("♻️ AI-Based Waste Classification")

    st.subheader(
        "Intelligent waste identification and disposal assistance"
    )

    st.markdown(
        """
        Upload an image of waste and let the AI model identify
        its category. The system provides the predicted category,
        confidence score, waste information, and disposal guidance.
        """
    )

    st.markdown("---")

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Waste Categories",
            "6"
        )

    with col2:
        st.metric(
            "Predictions",
            len(st.session_state.history)
        )

    with col3:
        st.metric(
            "Model Input",
            "224 × 224"
        )

    with col4:
        st.metric(
            "Model Type",
            "CNN"
        )

    st.markdown("---")

    st.subheader("Supported Waste Categories")

    cols = st.columns(6)

    for index, class_name in enumerate(class_names):

        info = WASTE_INFO.get(
            class_name,
            {
                "icon": "♻️",
                "title": class_name.title()
            }
        )

        with cols[index]:
            st.markdown(
                f"""
                <div style="
                    border:1px solid #ddd;
                    border-radius:10px;
                    padding:15px;
                    text-align:center;
                    min-height:120px;
                ">
                    <h2>{info['icon']}</h2>
                    <b>{info['title']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.info(
        "Go to 'Classify Waste' from the sidebar to analyze an image."
    )


# ============================================================
# CLASSIFICATION PAGE
# ============================================================

elif page == "📷 Classify Waste":

    st.title("📷 Classify Waste")

    st.write(
        "Upload a waste image and the AI model will classify it."
    )

    uploaded_file = st.file_uploader(
        "Choose a waste image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("AI Analysis")

            analyze = st.button(
                "🔍 Analyze Waste",
                type="primary",
                use_container_width=True
            )

            if analyze:

                with st.spinner("Analyzing image..."):

                    predicted_class, confidence, predictions = predict_image(
                        image,
                        interpreter,
                        input_details,
                        output_details,
                        class_names
                    )

                info = WASTE_INFO.get(
                    predicted_class,
                    {
                        "icon": "♻️",
                        "title": predicted_class.title(),
                        "type": "Unknown",
                        "disposal": "Follow local waste-management guidance.",
                        "tips": "Check local recycling rules."
                    }
                )

                st.success("Classification completed!")

                st.markdown(
                    f"""
                    ## {info['icon']} {info['title']}
                    """
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(
                    confidence
                )

                st.markdown(
                    f"**Waste Type:** {info['type']}"
                )

                st.markdown("---")

                st.subheader("♻️ Disposal Recommendation")

                st.write(
                    info["disposal"]
                )

                st.subheader("💡 Recycling Tip")

                st.write(
                    info["tips"]
                )

                # Probability chart

                st.markdown("---")

                st.subheader("📊 Model Probabilities")

                probability_data = {
                    class_names[i]: float(predictions[i])
                    for i in range(len(class_names))
                }

                st.bar_chart(
                    probability_data
                )

                # History

                st.session_state.history.append(
                    {
                        "time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "filename": uploaded_file.name,
                        "prediction": predicted_class,
                        "confidence": confidence
                    }
                )

    else:

        st.info(
            "Upload a JPG, JPEG, PNG, or WEBP image to begin."
        )


# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Waste Classification Analytics")

    history = st.session_state.history

    if not history:

        st.info(
            "No predictions available yet. "
            "Classify some images first."
        )

    else:

        total = len(history)

        counts = {}

        for item in history:

            category = item["prediction"]

            counts[category] = counts.get(
                category,
                0
            ) + 1

        average_confidence = (
            sum(
                item["confidence"]
                for item in history
            ) / total
        )

        most_common = max(
            counts,
            key=counts.get
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Predictions",
                total
            )

        with col2:
            st.metric(
                "Most Detected",
                most_common.title()
            )

        with col3:
            st.metric(
                "Average Confidence",
                f"{average_confidence * 100:.2f}%"
            )

        st.markdown("---")

        st.subheader("Category Distribution")

        chart_data = {
            category.title(): count
            for category, count in counts.items()
        }

        st.bar_chart(chart_data)

        st.markdown("---")

        st.subheader("Prediction Details")

        for category in class_names:

            count = counts.get(
                category,
                0
            )

            st.write(
                f"**{category.title()}**: {count}"
            )


# ============================================================
# HISTORY PAGE
# ============================================================

elif page == "🕐 History":

    st.title("🕐 Prediction History")

    history = st.session_state.history

    if not history:

        st.info(
            "No classification history yet."
        )

    else:

        for item in reversed(history):

            st.markdown(
                f"""
                **{item['prediction'].title()}**

                🕐 {item['time']}

                📁 {item['filename']}

                🎯 Confidence: {item['confidence'] * 100:.2f}%

                ---
                """
            )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
        ## AI Waste Classification System

        This application uses a convolutional neural network
        trained to classify waste images into six categories:

        - 📦 Cardboard
        - 🍾 Glass
        - 🥫 Metal
        - 📄 Paper
        - 🧴 Plastic
        - 🗑️ Trash

        ### Technology Stack

        **Frontend:** Streamlit

        **Machine Learning:** TensorFlow / TensorFlow Lite

        **Image Processing:** Pillow / NumPy

        **Model Input:** 224 × 224 RGB image

        ### Purpose

        The system aims to assist users in identifying waste
        categories and providing appropriate disposal and
        recycling guidance.

        ### Note

        Predictions are AI-based and should be treated as
        assistance rather than a replacement for local
        waste-management regulations.
        """
    )

    st.markdown("---")

    st.subheader("Model Classes")

    for class_name in class_names:

        info = WASTE_INFO.get(class_name)

        if info:

            st.write(
                f"{info['icon']} **{info['title']}**"
            )
"""
Facial Expression Analysis Page
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import plotly.graph_objects as go
import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fusion.inference_wrappers import predict_face  # noqa: E402
from fusion.label_mapper import STANDARD_EMOTION_LABELS  # noqa: E402
from utils.input_handlers import capture_image, load_image  # noqa: E402


st.set_page_config(
    page_title="Facial Expression Analysis",
    page_icon="🎭",
    layout="wide",
)

st.markdown(
    """
<style>
    .page-title {
        color: #2d3748;
        font-size: 2.5rem;
        font-weight: 750;
        text-align: center;
        margin: 1rem 0 0.25rem 0;
    }
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _render_face_result(prediction: dict[str, Any]) -> None:
    label = str(prediction.get("label", "unavailable"))
    confidence = prediction.get("confidence", 0.0)
    metadata = prediction.get("metadata") if isinstance(prediction.get("metadata"), dict) else {}
    probabilities = prediction.get("probabilities") if isinstance(prediction.get("probabilities"), dict) else {}

    if label == "unavailable":
        st.error("No face prediction available.")
        reason = metadata.get("reason")
        if reason:
            st.info(str(reason))
        return

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin: 0 0 0.75rem 0;">Detected Emotion</h3>
            <h1 style="margin: 0;">{label.upper()}</h1>
            <p style="font-size: 1.25rem; margin: 0.75rem 0 0 0;">Confidence: {_format_percent(confidence)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    values = [float(probabilities.get(emotion, 0.0)) * 100.0 for emotion in STANDARD_EMOTION_LABELS]
    fig = go.Figure(
        data=[
            go.Bar(
                x=STANDARD_EMOTION_LABELS,
                y=values,
                marker_color=["#ef4444", "#14b8a6", "#0ea5e9", "#f59e0b", "#64748b", "#6366f1", "#a855f7"],
            )
        ]
    )
    fig.update_layout(
        yaxis_title="Probability (%)",
        height=340,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Confidence Scores")
    for emotion in STANDARD_EMOTION_LABELS:
        prob = float(probabilities.get(emotion, 0.0))
        c1, c2, c3 = st.columns([3, 6, 2])
        c1.markdown(f"**{emotion.capitalize()}**")
        c2.progress(prob)
        c3.markdown(f"**{_format_percent(prob)}**")


st.markdown('<h1 class="page-title">🎭 Facial Expression Analysis</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Select an input mode, provide one image, then run prediction manually.</p>',
    unsafe_allow_html=True,
)

st.session_state.setdefault("face_page_prediction", None)

with st.sidebar:
    st.markdown("### Configuration")
    confidence_threshold = st.slider(
        "Face Detection Threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.85,
        step=0.01,
        help="Minimum MTCNN confidence required before emotion prediction.",
    )

input_col, result_col = st.columns([1, 1], gap="large")
selected_image = None

with input_col:
    st.subheader("Face Input")
    mode = st.radio(
        "Select Input Mode",
        ["Upload Image", "Capture from Camera"],
        horizontal=True,
        key="face_input_mode",
    )

    if mode == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="face_upload_image",
        )
        if uploaded_image is not None:
            try:
                selected_image = load_image(uploaded_image)
                st.image(selected_image, caption="Selected image", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not load image: {exc}")
    else:
        camera_image = st.camera_input("Capture Image", key="face_camera_image")
        if camera_image is not None:
            try:
                selected_image = capture_image(camera_image)
                st.image(selected_image, caption="Captured image", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read captured image: {exc}")

    run_prediction = st.button(
        "Run Face Prediction",
        disabled=selected_image is None,
        type="primary",
        use_container_width=True,
    )

with result_col:
    st.subheader("Results")
    if run_prediction and selected_image is not None:
        with st.spinner("Running face detection and emotion prediction..."):
            st.session_state["face_page_prediction"] = predict_face(
                selected_image,
                confidence_threshold=confidence_threshold,
            )

    prediction = st.session_state.get("face_page_prediction")
    if isinstance(prediction, dict):
        _render_face_result(prediction)
    else:
        st.info("No automatic detection runs here. Select or capture an image, then click Run Face Prediction.")

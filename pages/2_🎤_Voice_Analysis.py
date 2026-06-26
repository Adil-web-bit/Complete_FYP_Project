"""
Voice Emotion Analysis Page
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

from fusion.inference_wrappers import predict_voice  # noqa: E402
from fusion.label_mapper import STANDARD_EMOTION_LABELS  # noqa: E402
from utils.input_handlers import record_audio  # noqa: E402


st.set_page_config(
    page_title="Voice Analysis",
    page_icon="🎤",
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


def _render_voice_result(prediction: dict[str, Any]) -> None:
    label = str(prediction.get("label", "unavailable"))
    confidence = prediction.get("confidence", 0.0)
    metadata = prediction.get("metadata") if isinstance(prediction.get("metadata"), dict) else {}
    probabilities = prediction.get("probabilities") if isinstance(prediction.get("probabilities"), dict) else {}

    if label == "unavailable":
        st.error("No voice prediction available.")
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
                x=values,
                y=STANDARD_EMOTION_LABELS,
                orientation="h",
                marker_color="#667eea",
                text=[f"{value:.1f}%" for value in values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Probability (%)",
        height=340,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Confidence Scores")
    for emotion in STANDARD_EMOTION_LABELS:
        prob = float(probabilities.get(emotion, 0.0))
        c1, c2, c3 = st.columns([3, 6, 2])
        c1.markdown(f"**{emotion.capitalize()}**")
        c2.progress(prob)
        c3.markdown(f"**{_format_percent(prob)}**")


st.markdown('<h1 class="page-title">🎤 Voice Emotion Analysis</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Select an input mode, provide one audio sample, then run prediction manually.</p>',
    unsafe_allow_html=True,
)

st.session_state.setdefault("voice_page_prediction", None)

with st.sidebar:
    st.markdown("### Configuration")
    recording_seconds = st.slider(
        "Recording duration target",
        min_value=3,
        max_value=5,
        value=4,
        step=1,
        help="Record roughly this many seconds of clear speech before stopping the browser recorder.",
    )
    st.caption("Audio inference uses the existing 22050 Hz librosa preprocessing pipeline.")

input_col, result_col = st.columns([1, 1], gap="large")
selected_audio = None

with input_col:
    st.subheader("Voice Input")
    mode = st.radio(
        "Select Input Mode",
        ["Upload Audio", "Record Voice"],
        horizontal=True,
        key="voice_input_mode",
    )

    if mode == "Upload Audio":
        uploaded_audio = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "ogg", "flac"],
            key="voice_upload_audio",
        )
        if uploaded_audio is not None:
            selected_audio = uploaded_audio
            st.audio(uploaded_audio)
            st.info(f"File: `{uploaded_audio.name}` | Size: {uploaded_audio.size / 1024:.2f} KB")
    else:
        recorded_audio = st.audio_input("Start Recording", key="voice_record_audio")
        if recorded_audio is not None:
            selected_audio = record_audio(recorded_audio, seconds=recording_seconds)
            st.audio(selected_audio)
            st.caption(f"Target duration: {recording_seconds} seconds")

    run_prediction = st.button(
        "Run Voice Prediction",
        disabled=selected_audio is None,
        type="primary",
        use_container_width=True,
    )

with result_col:
    st.subheader("Results")
    if run_prediction and selected_audio is not None:
        with st.spinner("Running voice feature extraction and emotion prediction..."):
            st.session_state["voice_page_prediction"] = predict_voice(selected_audio)

    prediction = st.session_state.get("voice_page_prediction")
    if isinstance(prediction, dict):
        _render_voice_result(prediction)
    else:
        st.info("No automatic detection runs here. Upload or record audio, then click Run Voice Prediction.")

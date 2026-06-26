"""
Multimodal Human Behavior Analysis Page
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent.parent
TYPING_MODULE_DIR = SCRIPT_DIR / "Typing_scrolling"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TYPING_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(TYPING_MODULE_DIR))

from components.monitoring_component import monitoring_component  # noqa: E402
from fusion.fusion_engine import FusionEngine  # noqa: E402
from fusion.inference_wrappers import predict_behavior, predict_face, predict_voice  # noqa: E402
from fusion.label_mapper import STANDARD_EMOTION_LABELS  # noqa: E402
from fusion.prediction_schema import build_unified_prediction, new_session_id, utc_timestamp  # noqa: E402
from fusion.session_store import load_multimodal_history, save_multimodal_session  # noqa: E402
from utils.input_handlers import capture_image, load_image, record_audio  # noqa: E402
from src.utils.session_payload import (  # noqa: E402
    calculate_payload_duration_seconds,
    get_payload_event_counts,
    is_stopped_payload,
    normalize_component_payload,
    payload_has_events,
)


st.set_page_config(
    page_title="Multimodal Analysis",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
<style>
    .main-title {
        color: #2d3748;
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #718096;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-panel {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        background: #ffffff;
    }
    .muted {
        color: #718096;
        font-size: 0.9rem;
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


def _render_probability_chart(title: str, probabilities: dict[str, Any]) -> None:
    values = []
    for label in STANDARD_EMOTION_LABELS:
        try:
            values.append(float(probabilities.get(label, 0.0)) * 100.0)
        except (TypeError, ValueError):
            values.append(0.0)

    fig = go.Figure(
        data=[
            go.Bar(
                x=STANDARD_EMOTION_LABELS,
                y=values,
                marker_color=[
                    "#ef4444",
                    "#14b8a6",
                    "#0ea5e9",
                    "#f59e0b",
                    "#64748b",
                    "#6366f1",
                    "#a855f7",
                ],
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title="Probability (%)",
        xaxis_title="",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_modality_summary(title: str, prediction: dict[str, Any]) -> None:
    label = str(prediction.get("label", "unavailable"))
    confidence = prediction.get("confidence", 0.0)
    metadata = prediction.get("metadata") if isinstance(prediction.get("metadata"), dict) else {}

    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.subheader(title)
    c1, c2 = st.columns(2)
    c1.metric("Prediction", label.upper())
    c2.metric("Confidence", _format_percent(confidence))
    reason = metadata.get("reason")
    if reason:
        st.caption(str(reason))
    st.markdown("</div>", unsafe_allow_html=True)


def _build_behavior_session(
    *,
    session_id: str,
    user_id: str,
    payload: dict[str, Any],
    consent_given: bool,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "user_id": user_id.strip() or "anonymous",
        "timestamp_utc": utc_timestamp(),
        "consent_given": bool(consent_given),
        "behavior_label": None,
        "typed_text": str(payload.get("typed_text") or ""),
        "typing_events": payload.get("typing_events") if isinstance(payload.get("typing_events"), list) else [],
        "scroll_events": payload.get("scroll_events") if isinstance(payload.get("scroll_events"), list) else [],
        "started_at_ms": payload.get("started_at_ms"),
        "stopped_at_ms": payload.get("stopped_at_ms"),
        "duration_seconds": calculate_payload_duration_seconds(payload),
        "source": "streamlit_multimodal_component",
    }


def _run_controlled_multimodal_prediction(
    *,
    session_id: str,
    user_id: str,
    face_image: Image.Image,
    voice_audio,
    behavior_payload: dict[str, Any] | None,
    consent_given: bool,
    input_mode: str,
    target_label: str | None = None,
) -> dict[str, Any]:
    behavior_prediction: dict[str, Any] | None = None

    if (
        consent_given
        and isinstance(behavior_payload, dict)
        and is_stopped_payload(behavior_payload)
        and payload_has_events(behavior_payload)
    ):
        behavior_session = _build_behavior_session(
            session_id=session_id,
            user_id=user_id,
            payload=behavior_payload,
            consent_given=consent_given,
        )
        behavior_prediction = predict_behavior(behavior_session)

    face_prediction = predict_face(face_image)
    voice_prediction = predict_voice(voice_audio)

    final_prediction = FusionEngine().fuse(
        face=face_prediction,
        voice=voice_prediction,
        behavior=behavior_prediction,
    )

    unified = build_unified_prediction(
        session_id=session_id,
        timestamp=utc_timestamp(),
        face=face_prediction,
        voice=voice_prediction,
        behavior=behavior_prediction or {},
        final_prediction=final_prediction,
    )
    unified["input_mode"] = input_mode
    if target_label:
        unified["target_label"] = target_label
        unified["final_human_state_label"] = target_label
    return unified


st.markdown('<div class="main-title">Multimodal Human Behavior Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Combine facial expression, voice tone, and typing behavior with weighted late fusion.</div>',
    unsafe_allow_html=True,
)

st.session_state.setdefault("multimodal_session_id", new_session_id())
st.session_state.setdefault("multimodal_payload", None)
st.session_state.setdefault("multimodal_prediction", None)
st.session_state.setdefault("multimodal_saved_path", None)

top_left, top_right = st.columns([2, 1])
with top_left:
    user_id = st.text_input("User ID", value="anonymous", key="multimodal_user_id").strip()
with top_right:
    if st.button("New Session", use_container_width=True):
        st.session_state["multimodal_session_id"] = new_session_id()
        st.session_state["multimodal_payload"] = None
        st.session_state["multimodal_prediction"] = None
        st.session_state["multimodal_saved_path"] = None
        st.rerun()

st.caption(f"Session ID: `{st.session_state['multimodal_session_id']}`")

consent_given = st.checkbox(
    "I have consent to collect typing events inside this page.",
    value=False,
    key="multimodal_consent",
)

face_input_image = None
voice_input_audio = None
face_input_mode = "Upload Image"
voice_input_mode = "Upload Audio"

input_col1, input_col2 = st.columns(2, gap="large")

with input_col1:
    st.subheader("Face Input")
    face_input_mode = st.radio(
        "Select Face Input Mode",
        ["Upload Image", "Capture from Camera"],
        horizontal=True,
        key="multimodal_face_input_mode",
    )
    if face_input_mode == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload facial image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="multimodal_image",
        )
        if uploaded_image is not None:
            try:
                face_input_image = load_image(uploaded_image)
                st.image(face_input_image, caption="Selected face input", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not load face image: {exc}")
    else:
        captured_image = st.camera_input(
            "Capture Image",
            key="multimodal_camera_image",
        )
        if captured_image is not None:
            try:
                face_input_image = capture_image(captured_image)
                st.image(face_input_image, caption="Captured face input", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read captured image: {exc}")

with input_col2:
    st.subheader("Voice Input")
    voice_input_mode = st.radio(
        "Select Voice Input Mode",
        ["Upload Audio", "Record Voice"],
        horizontal=True,
        key="multimodal_voice_input_mode",
    )
    if voice_input_mode == "Upload Audio":
        uploaded_audio = st.file_uploader(
            "Upload voice audio",
            type=["wav", "mp3", "ogg", "flac"],
            key="multimodal_audio",
        )
        if uploaded_audio is not None:
            voice_input_audio = uploaded_audio
            st.audio(uploaded_audio)
    else:
        recording_duration = st.slider(
            "Recording duration target",
            min_value=3,
            max_value=5,
            value=4,
            step=1,
            key="multimodal_recording_duration",
        )
        recorded_audio = st.audio_input(
            "Start Recording",
            key="multimodal_record_voice",
        )
        if recorded_audio is not None:
            voice_input_audio = record_audio(recorded_audio, seconds=recording_duration)
            st.audio(voice_input_audio)
            st.caption(f"Target duration: {recording_duration} seconds")

st.subheader("Behavior Input")
st.caption("Only interactions inside this embedded monitoring widget are captured.")
component_payload = monitoring_component(key="multimodal_monitoring_widget")
normalized_payload = normalize_component_payload(component_payload)

if normalized_payload is not None:
    st.session_state["multimodal_payload"] = normalized_payload

payload = st.session_state.get("multimodal_payload")
counts = get_payload_event_counts(payload if isinstance(payload, dict) else None)
duration = calculate_payload_duration_seconds(payload if isinstance(payload, dict) else None)

summary_cols = st.columns(3)
summary_cols[0].metric("Typing events", counts.get("typing", 0))
summary_cols[1].metric("Duration", f"{duration:.1f}s" if duration is not None else "0.0s")
summary_cols[2].metric(
    "Status",
    str(payload.get("monitoring_status", "idle")) if isinstance(payload, dict) else "idle",
)

can_predict = face_input_image is not None and voice_input_audio is not None
behavior_ready = (
    consent_given
    and isinstance(payload, dict)
    and is_stopped_payload(payload)
    and payload_has_events(payload)
)

if not can_predict:
    st.info("Select face and voice input modes, provide both inputs, then click Run Multimodal Prediction.")
elif not behavior_ready:
    st.info("Behavior input is optional here. Start and stop the monitoring widget with consent if you want behavior included.")

save_result = st.checkbox("Save combined prediction JSON", value=True, key="multimodal_save_json")
target_choice = st.selectbox(
    "Optional human target label for fusion training",
    ["Unlabeled", *STANDARD_EMOTION_LABELS],
    index=0,
    key="multimodal_target_label",
)
target_label = None if target_choice == "Unlabeled" else str(target_choice)

if st.button("Run Multimodal Prediction", disabled=not can_predict, type="primary"):
    input_mode = (
        f"face:{face_input_mode.lower().replace(' ', '_')};"
        f"voice:{voice_input_mode.lower().replace(' ', '_')}"
    )
    with st.spinner("Running face, voice, optional behavior, and fusion inference..."):
        prediction = _run_controlled_multimodal_prediction(
            session_id=st.session_state["multimodal_session_id"],
            user_id=user_id,
            face_image=face_input_image,
            voice_audio=voice_input_audio,
            behavior_payload=payload if isinstance(payload, dict) else None,
            consent_given=consent_given,
            input_mode=input_mode,
            target_label=target_label,
        )
        st.session_state["multimodal_prediction"] = prediction
        st.session_state["multimodal_saved_path"] = None
        if save_result:
            saved_path = save_multimodal_session(prediction)
            st.session_state["multimodal_saved_path"] = str(saved_path)

prediction = st.session_state.get("multimodal_prediction")
if isinstance(prediction, dict):
    st.subheader("Results")
    final = prediction.get("final_prediction", {})
    final_cols = st.columns(3)
    final_cols[0].metric("Final emotion", str(final.get("label", "neutral")).upper())
    final_cols[1].metric("Confidence", _format_percent(final.get("confidence", 0.0)))
    final_cols[2].metric("Fusion status", str(final.get("status", "unknown")))

    saved_path = st.session_state.get("multimodal_saved_path")
    if saved_path:
        st.success(f"Saved combined session JSON: `{saved_path}`")

    st.divider()
    face_col, voice_col, behavior_col = st.columns(3)
    with face_col:
        _render_modality_summary("Face", prediction.get("face", {}))
    with voice_col:
        _render_modality_summary("Voice", prediction.get("voice", {}))
    with behavior_col:
        _render_modality_summary("Behavior", prediction.get("behavior", {}))

    st.subheader("Probability Dashboard")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        _render_probability_chart(
            "Fused emotion scores",
            final.get("scores", {}) if isinstance(final.get("scores"), dict) else {},
        )
        _render_probability_chart(
            "Face probabilities",
            prediction.get("face", {}).get("probabilities", {}),
        )
    with chart_col2:
        _render_probability_chart(
            "Voice probabilities",
            prediction.get("voice", {}).get("probabilities", {}),
        )
        _render_probability_chart(
            "Behavior influence scores",
            prediction.get("behavior", {}).get("probabilities", {}),
        )

    with st.expander("Fusion Breakdown", expanded=True):
        st.json(final.get("modality_breakdown", {}))

    with st.expander("Unified Prediction JSON"):
        st.json(prediction)

with st.expander("Recent Multimodal Session History", expanded=False):
    history = load_multimodal_history(limit=10)
    if not history:
        st.write("No saved multimodal sessions yet.")
    else:
        for item in history:
            final = item.get("final_prediction", {}) if isinstance(item, dict) else {}
            st.write(
                f"`{item.get('timestamp', '')}` | `{item.get('session_id', '')}` | "
                f"**{str(final.get('label', 'unknown')).upper()}** "
                f"({_format_percent(final.get('confidence', 0.0))})"
            )

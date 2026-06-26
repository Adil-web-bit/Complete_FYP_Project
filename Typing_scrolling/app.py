from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from components.monitoring_component import monitoring_component
from src.config import ensure_project_directories
from src.data_collection.monitoring_session import MonitoringSessionCollector
from src.data_collection.public_dataset_loader import PublicDatasetLoader
from src.model.predictor import BehaviorPredictor
from src.model.public_dataset_trainer import PublicDatasetTrainer
from src.utils.session_payload import (
    calculate_payload_duration_seconds,
    get_payload_event_counts,
    is_stopped_payload,
    normalize_component_payload,
    payload_has_events,
)
from src.utils.validators import validate_non_empty_text


def _format_confidence(confidence: Any) -> str:
    try:
        value = float(confidence)
    except Exception:
        return "n/a"

    if value <= 1.0:
        value *= 100.0
    value = max(0.0, min(100.0, value))
    return f"{value:.1f}%"


def _render_feature_table(features: dict[str, Any], keys: list[str], *, title: str) -> None:
    st.write(title)
    rows: list[dict[str, Any]] = []
    for key in keys:
        raw = features.get(key, 0.0)
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        rows.append({"Feature": key, "Value": value})

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.table(rows)


def _render_schema_preview(preview: dict[str, Any], *, title: str) -> None:
    with st.expander(title, expanded=True):
        status = str(preview.get("status", "error"))
        message = str(preview.get("message", ""))
        if status == "ok":
            st.success(f"status: {status}")
        else:
            st.error(f"status: {status}")
        if message:
            st.write(message)

        c1, c2, c3 = st.columns(3)
        c1.metric("Original rows", str(preview.get("original_rows", 0)))
        c2.metric("Original cols", str(preview.get("original_columns", 0)))
        c3.metric("Final rows", str(preview.get("final_rows", 0)))

        st.write(f"Detected label column: `{preview.get('detected_label_column')}`")

        st.write("Original label counts")
        st.json(preview.get("original_label_counts") or {})

        st.write("Mapped label counts")
        st.json(preview.get("mapped_label_counts") or {})

        st.write(f"Unmapped/dropped label rows: `{preview.get('unmapped_or_dropped_label_rows')}`")

        st.write(f"Final feature count: `{preview.get('final_feature_count')}`")
        st.write("Final feature columns")
        st.code("\n".join(list(preview.get("final_feature_columns") or [])) or "(none)")

        st.write("Dropped identity/text columns")
        st.code("\n".join(list(preview.get("dropped_identity_text_columns") or [])) or "(none)")

        st.write("Constant columns removed")
        st.code("\n".join(list(preview.get("constant_columns_removed") or [])) or "(none)")

        raw_cols = list(preview.get("original_column_names") or [])
        if raw_cols:
            st.write("Raw column names (first 5)")
            st.code("\n".join(list(preview.get("original_column_names_head5") or [])) or "(none)")

        records = preview.get("preview_records") or []
        if isinstance(records, list) and records:
            st.write("Final training dataframe preview (first 5 rows)")
            st.dataframe(records, use_container_width=True)
        else:
            st.write("Final training dataframe preview (empty)")


def _render_demo() -> None:
    st.title("Human Behavior Predictor")
    st.warning("Experimental demo. Not a medical or psychological diagnosis.")

    st.session_state.setdefault("latest_captured_payload", None)
    st.session_state.setdefault("latest_prediction", None)
    st.session_state.setdefault("latest_temp_session", None)

    consent_given = st.checkbox(
        "I confirm this is my own session or I have explicit permission from the participant, and I consent to capturing "
        "typing interactions inside this demo component.",
        key="home_consent",
    )
    user_id = st.text_input("User ID", value="anonymous", key="home_user_id").strip()

    if st.button("Reset", key="home_reset"):
        st.session_state["latest_captured_payload"] = None
        st.session_state["latest_prediction"] = None
        st.session_state["latest_temp_session"] = None
        st.rerun()

    st.caption("Privacy: only interactions inside the embedded monitoring widget are captured (not system-wide).")

    raw_component_value = monitoring_component(key="monitoring_component")

    component_error_message: str | None = None
    if isinstance(raw_component_value, dict):
        # Some Streamlit component failures return an error dict rather than the expected payload.
        # Ignore these in normal flow.
        if "ERROR" in raw_component_value or "error" in raw_component_value:
            msg = raw_component_value.get("message")
            component_error_message = str(msg) if isinstance(msg, str) and msg.strip() else "Component error."
        else:
            msg = raw_component_value.get("message")
            if isinstance(msg, str) and "src property must be a valid json object" in msg.lower():
                component_error_message = msg

    normalized = None
    if component_error_message is None and isinstance(raw_component_value, dict):
        normalized = normalize_component_payload(raw_component_value)
    if normalized is not None and normalized.get("monitoring_status") == "stopped":
        st.session_state["latest_captured_payload"] = normalized

    payload = st.session_state.get("latest_captured_payload")
    counts = get_payload_event_counts(payload if isinstance(payload, dict) else None)
    duration_s = calculate_payload_duration_seconds(payload if isinstance(payload, dict) else None)

    st.subheader("Captured session summary")
    if component_error_message is not None:
        st.error(f"Monitoring widget error: {component_error_message}")
        st.info(
            "If the widget area is blank, the Streamlit component bridge may not have loaded. "
            "Try refreshing the page. The manual JSON fallback is available in the Advanced expander."
        )
    if isinstance(payload, dict):
        status = str(payload.get("monitoring_status", "idle"))
        typed_text = str(payload.get("typed_text", ""))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", status)
        c2.metric("Text length", str(len(typed_text)))
        c3.metric("Typing events", str(counts.get("typing", 0)))
        c4.metric("Duration (s)", f"{float(duration_s):.1f}" if isinstance(duration_s, (int, float)) else "n/a")
    else:
        st.info("Use the widget Start/Stop controls to capture a session, then predict.")

    can_predict = (
        consent_given
        and validate_non_empty_text(user_id)
        and isinstance(payload, dict)
        and is_stopped_payload(payload)
        and payload_has_events(payload)
    )

    if st.button("Predict Mood", key="home_predict", disabled=not can_predict):
        try:
            if not isinstance(payload, dict):
                raise ValueError("No captured payload available.")
            collector = MonitoringSessionCollector()
            temp_session = collector.create_record(
                user_id=user_id,
                consent_given=True,
                behavior_label=None,
                typed_text=str(payload.get("typed_text", "")),
                typing_events=payload.get("typing_events", [])
                if isinstance(payload.get("typing_events", []), list)
                else [],
                scroll_events=payload.get("scroll_events", [])
                if isinstance(payload.get("scroll_events", []), list)
                else [],
                started_at_ms=payload.get("started_at_ms") if isinstance(payload.get("started_at_ms"), int) else None,
                stopped_at_ms=payload.get("stopped_at_ms") if isinstance(payload.get("stopped_at_ms"), int) else None,
            )
            predictor = BehaviorPredictor()
            result = predictor.predict_from_session(temp_session)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.session_state["latest_temp_session"] = temp_session
            st.session_state["latest_prediction"] = result

    result = st.session_state.get("latest_prediction")
    if isinstance(result, dict):
        st.subheader("Prediction result")
        predicted_label = str(result.get("predicted_label", "normal"))
        confidence_str = _format_confidence(result.get("confidence"))
        model_type = str(result.get("model_type", "unknown"))

        if model_type == "trained_ml":
            st.success("Prediction generated by trained model.")
        else:
            st.info("Prediction generated by fallback logic.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted mood", predicted_label)
        c2.metric("Confidence", confidence_str)
        c3.metric("Model", model_type)

        source = result.get("source")
        dataset_type = result.get("dataset_type")
        if source is not None or dataset_type is not None:
            st.caption(f"Source: `{source}` | Dataset: `{dataset_type}`")

        reason = result.get("reason")
        if reason is not None:
            st.write(f"Reason: {reason}")

    with st.expander("Technical details"):
        if isinstance(result, dict) and isinstance(result.get("features"), dict):
            features: dict[str, Any] = result["features"]

            key_features = [
                "typed_text_length",
                "total_key_events",
                "backspace_count",
                "keys_per_second",
                "avg_key_interval_ms",
                "total_duration_seconds",
            ]
            emosurv_features = ["D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"]

            _render_feature_table(features, key_features, title="A) Key extracted features")
            _render_feature_table(features, emosurv_features, title="B) EmoSurv timing features")
        else:
            st.write("No extracted features available yet.")

        with st.expander("C) Raw debug JSON (developer)", expanded=False):
            st.caption("Developer/debug info. Contains raw event arrays and may be large.")
            st.write("Raw normalized payload")
            st.json(payload if isinstance(payload, dict) else None)

            st.write("Temporary session JSON")
            st.json(st.session_state.get("latest_temp_session"))

    with st.expander("Advanced fallback: paste captured JSON manually"):
        if component_error_message is None:
            st.caption("Fallback is hidden because the monitoring widget bridge appears healthy.")
        else:
            raw_text = st.text_area(
                "Paste captured payload JSON here",
                key="manual_payload_json",
                height=180,
                placeholder='{"typed_text":"...","typing_events":[...],"started_at_ms":...,"stopped_at_ms":...,"monitoring_status":"stopped"}',
            )
            if st.button("Use pasted JSON", key="manual_payload_use"):
                try:
                    parsed = json.loads(raw_text)
                    if not isinstance(parsed, dict):
                        raise ValueError("JSON must be an object.")
                    normalized_manual = normalize_component_payload(parsed)
                    if normalized_manual is None:
                        raise ValueError("Payload could not be normalized.")
                    st.session_state["latest_captured_payload"] = normalized_manual
                    st.success("Pasted payload loaded.")
                except Exception as exc:
                    st.error(f"Invalid JSON: {exc}")


def _render_training() -> None:
    st.title("Training")
    st.info(
        "This page is for training or replacing the demo model. The Home page will automatically use the latest saved model."
    )

    st.subheader("Dataset guidance")
    st.write(
        "- Frequency Dataset: easiest baseline, fewer features.\n"
        "- Fixed Text Typing Dataset: recommended next, richer timing features.\n"
        "- Free Text Typing Dataset: also useful, noisier but realistic.\n"
        "- Participants Information Dataset: do not use alone for mood prediction."
    )

    st.subheader("Current model status")
    try:
        from src.config import MODEL_FILE  # local import to keep startup lightweight

        if MODEL_FILE.exists():
            st.success(f"Model exists: `{MODEL_FILE}`")
            try:
                import joblib

                artifact = joblib.load(MODEL_FILE)
                if isinstance(artifact, dict):
                    source = artifact.get("source")
                    dataset_type = artifact.get("dataset_type")
                    training_count = artifact.get("training_count")
                    validation_accuracy = artifact.get("validation_accuracy")
                    feature_names = artifact.get("feature_names")
                    st.write(f"source: `{source}`")
                    st.write(f"dataset_type: `{dataset_type}`")
                    st.write(f"training_count: `{training_count}`")
                    st.write(f"validation_accuracy: `{validation_accuracy}`")
                    st.write(
                        f"feature_names count: `{len(feature_names) if isinstance(feature_names, list) else 0}`"
                    )
                else:
                    st.warning("Model file exists but could not be read as a dict artifact.")
            except Exception as exc:
                st.warning(f"Could not load model artifact: {exc}")
        else:
            st.warning("Model does not exist yet. Training will create/overwrite it.")
    except Exception as exc:
        st.warning(f"Model status check failed: {exc}")

    loader = PublicDatasetLoader()
    trainer = PublicDatasetTrainer(loader=loader)

    st.subheader("A) Upload CSV")
    uploaded = st.file_uploader("Upload a .csv file", type=["csv"], key="training_upload_csv")
    if uploaded is not None:
        try:
            from src.config import PUBLIC_DATA_DIR

            uploaded_path = PUBLIC_DATA_DIR / "uploaded_public_dataset.csv"
            uploaded_path.parent.mkdir(parents=True, exist_ok=True)
            uploaded_path.write_bytes(uploaded.getvalue())

            df = loader.load_csv(uploaded_path)
            preview = loader.preview_schema(df)
            _render_schema_preview(preview, title="Uploaded CSV Schema Preview")

            if str(preview.get("status")) == "ok":
                if st.button("Train Uploaded CSV", key="train_uploaded_csv"):
                    report = trainer.train_from_csv(uploaded_path)
                    if report.get("status") == "trained":
                        st.success(report.get("message", "Trained."))
                    else:
                        st.warning(report.get("message", "Training did not complete successfully."))
                    st.json(report)
            else:
                st.warning("Fix the CSV issues above before training.")
        except Exception as exc:
            st.error(f"Failed to preview uploaded CSV: {exc}")

    st.subheader("B) Train from local CSV path")
    local_path_str = st.text_input(
        "Local CSV path",
        value="data/public/public_keystroke_dataset.csv",
        key="training_local_csv_path",
    ).strip()

    if st.button("Preview Local CSV", key="preview_local_csv"):
        try:
            from src.config import get_project_paths

            root_dir = get_project_paths().root_dir
            local_path = Path(local_path_str).expanduser()
            if not local_path.is_absolute():
                local_path = (root_dir / local_path).resolve()

            if not local_path.exists():
                raise FileNotFoundError(
                    f"File not found: {local_path}. Put a CSV there or update the path."
                )

            df_local = loader.load_csv(local_path)
            preview_local = loader.preview_schema(df_local)
            st.session_state["training_local_preview"] = preview_local
            st.session_state["training_local_resolved_path"] = str(local_path)
        except Exception as exc:
            st.session_state["training_local_preview"] = {"status": "error", "message": str(exc)}
            st.session_state["training_local_resolved_path"] = None

    preview_local_cached = st.session_state.get("training_local_preview")
    resolved_local_path = st.session_state.get("training_local_resolved_path")
    if isinstance(preview_local_cached, dict) and preview_local_cached:
        _render_schema_preview(preview_local_cached, title="Local CSV Schema Preview")

        if str(preview_local_cached.get("status")) == "ok" and isinstance(resolved_local_path, str):
            if st.button("Train from Local CSV", key="train_local_csv"):
                try:
                    report = trainer.train_from_csv(Path(resolved_local_path))
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
                else:
                    if report.get("status") == "trained":
                        st.success(report.get("message", "Trained."))
                    else:
                        st.warning(report.get("message", "Training did not complete successfully."))
                    st.json(report)
        else:
            st.warning("Fix the CSV issues above before training.")

    st.caption("Developer-only: training overwrites `models/behavior_model.joblib`.")


def _render_about() -> None:
    st.header("About")
    st.write("A local-first demo that captures typing only inside the embedded component.")
    st.write("Explicit consent is required. Do not use this to secretly monitor anyone.")
    st.write("Predictions are experimental and not medical/psychological diagnoses.")
    st.write(
        "If `models/behavior_model.joblib` exists, the app uses it automatically. Otherwise it uses a rule-based fallback."
    )


def main() -> None:
    st.set_page_config(page_title="Human Behavior Predictor", layout="centered")
    try:
        ensure_project_directories()
    except OSError as exc:
        st.error(f"Failed to initialize project directories: {exc}")
        st.stop()

    pages: dict[str, Callable[[], None]] = {"Home": _render_demo, "Training": _render_training, "About": _render_about}
    selection = st.sidebar.selectbox("Navigation", list(pages.keys()), index=0)
    pages[selection]()


if __name__ == "__main__":
    main()

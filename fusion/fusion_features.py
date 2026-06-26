from __future__ import annotations

from typing import Any

from .label_mapper import STANDARD_EMOTION_LABELS, normalize_behavior_label


BEHAVIOR_LABELS: list[str] = [
    "normal",
    "focused",
    "stressed",
    "rushed",
    "distracted",
]

BEHAVIOR_NUMERIC_FEATURES: list[str] = [
    "typed_text_length",
    "total_key_events",
    "backspace_count",
    "unique_keys_count",
    "typing_duration_seconds",
    "keys_per_second",
    "avg_key_interval_ms",
    "std_key_interval_ms",
    "total_scroll_events",
    "scroll_duration_seconds",
    "total_scroll_distance",
    "avg_scroll_speed",
    "max_scroll_speed",
    "scroll_direction_changes",
    "total_duration_seconds",
    "D1U1",
    "D1U2",
    "D1D2",
    "U1D2",
    "U1U2",
    "D1U3",
    "D1D3",
]

FACE_PROBABILITY_COLUMNS: list[str] = [
    f"face_{label}_prob" for label in STANDARD_EMOTION_LABELS
]
VOICE_PROBABILITY_COLUMNS: list[str] = [
    f"voice_{label}_prob" for label in STANDARD_EMOTION_LABELS
]
BEHAVIOR_LABEL_COLUMNS: list[str] = [
    f"behavior_label_{label}" for label in BEHAVIOR_LABELS
]
BEHAVIOR_FEATURE_COLUMNS: list[str] = [
    "behavior_confidence",
    *[f"behavior_{name}" for name in BEHAVIOR_NUMERIC_FEATURES],
    *BEHAVIOR_LABEL_COLUMNS,
]

FUSION_FEATURE_COLUMNS: list[str] = [
    *FACE_PROBABILITY_COLUMNS,
    *VOICE_PROBABILITY_COLUMNS,
    *BEHAVIOR_FEATURE_COLUMNS,
]

TARGET_COLUMN = "target_label"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _probability(prediction: dict[str, Any], label: str) -> float:
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, dict):
        return 0.0
    return _safe_float(probabilities.get(label), default=0.0)


def target_from_session(session: dict[str, Any]) -> tuple[str | None, str]:
    """
    Extract the supervised target label from a saved multimodal session.

    Preferred labels are explicit human labels. If none exists, the previous
    heuristic fusion result is used as a weak label so the pipeline can be
    demonstrated before collecting manually labeled data.
    """
    explicit_candidates = (
        "target_label",
        "final_human_state_label",
        "human_state_label",
        "ground_truth_label",
        "label",
    )
    for key in explicit_candidates:
        value = session.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower(), key

    final_prediction = session.get("final_prediction")
    if isinstance(final_prediction, dict):
        for key in explicit_candidates:
            value = final_prediction.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower(), f"final_prediction.{key}"
        value = final_prediction.get("label")
        if isinstance(value, str) and value.strip():
            return value.strip().lower(), "weak_final_prediction.label"

    return None, "missing"


def build_feature_row(session: dict[str, Any]) -> dict[str, Any]:
    face = session.get("face") if isinstance(session.get("face"), dict) else {}
    voice = session.get("voice") if isinstance(session.get("voice"), dict) else {}
    behavior = session.get("behavior") if isinstance(session.get("behavior"), dict) else {}
    behavior_features = behavior.get("features") if isinstance(behavior.get("features"), dict) else {}
    behavior_label = normalize_behavior_label(behavior.get("label"))

    row: dict[str, Any] = {}
    for label in STANDARD_EMOTION_LABELS:
        row[f"face_{label}_prob"] = _probability(face, label)
        row[f"voice_{label}_prob"] = _probability(voice, label)

    row["behavior_label"] = behavior_label
    row["behavior_confidence"] = _safe_float(behavior.get("confidence"), default=0.0)

    for feature_name in BEHAVIOR_NUMERIC_FEATURES:
        row[f"behavior_{feature_name}"] = _safe_float(
            behavior_features.get(feature_name),
            default=0.0,
        )

    for label in BEHAVIOR_LABELS:
        row[f"behavior_label_{label}"] = 1.0 if behavior_label == label else 0.0

    target, source = target_from_session(session)
    row[TARGET_COLUMN] = target
    row["target_source"] = source
    row["session_id"] = str(session.get("session_id") or "")
    row["timestamp"] = str(session.get("timestamp") or "")
    return row

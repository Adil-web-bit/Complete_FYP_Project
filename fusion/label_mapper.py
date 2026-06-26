from __future__ import annotations

from typing import Any


STANDARD_EMOTION_LABELS: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

FACE_LABELS: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

VOICE_LABEL_ALIASES: dict[str, str] = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fearful": "fear",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "calmness": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprised": "surprise",
    "surprise": "surprise",
}

BEHAVIOR_EMOTION_SCORES: dict[str, dict[str, float]] = {
    "focused": {
        "angry": 0.02,
        "disgust": 0.01,
        "fear": 0.03,
        "happy": 0.64,
        "neutral": 0.25,
        "sad": 0.03,
        "surprise": 0.02,
    },
    "normal": {
        "angry": 0.03,
        "disgust": 0.02,
        "fear": 0.04,
        "happy": 0.14,
        "neutral": 0.70,
        "sad": 0.05,
        "surprise": 0.02,
    },
    "stressed": {
        "angry": 0.35,
        "disgust": 0.05,
        "fear": 0.30,
        "happy": 0.02,
        "neutral": 0.08,
        "sad": 0.17,
        "surprise": 0.03,
    },
    "rushed": {
        "angry": 0.22,
        "disgust": 0.03,
        "fear": 0.36,
        "happy": 0.04,
        "neutral": 0.12,
        "sad": 0.10,
        "surprise": 0.13,
    },
    "distracted": {
        "angry": 0.08,
        "disgust": 0.03,
        "fear": 0.18,
        "happy": 0.05,
        "neutral": 0.34,
        "sad": 0.27,
        "surprise": 0.05,
    },
}


def clean_label(label: Any) -> str:
    return str(label or "").strip().lower().replace(" ", "_")


def normalize_face_label(label: Any) -> str:
    label_key = clean_label(label)
    if label_key == "surprised":
        return "surprise"
    return label_key if label_key in STANDARD_EMOTION_LABELS else "neutral"


def normalize_voice_label(label: Any) -> str:
    return VOICE_LABEL_ALIASES.get(clean_label(label), "neutral")


def normalize_behavior_label(label: Any) -> str:
    label_key = clean_label(label)
    return label_key if label_key in BEHAVIOR_EMOTION_SCORES else "normal"


def map_probabilities(
    raw_probabilities: dict[str, Any],
    label_mapper,
) -> dict[str, float]:
    mapped = {label: 0.0 for label in STANDARD_EMOTION_LABELS}
    for raw_label, raw_value in raw_probabilities.items():
        mapped_label = label_mapper(raw_label)
        if mapped_label not in mapped:
            continue
        try:
            mapped[mapped_label] += max(0.0, float(raw_value))
        except (TypeError, ValueError):
            continue
    return mapped


def behavior_to_emotion_scores(label: Any) -> dict[str, float]:
    behavior_label = normalize_behavior_label(label)
    scores = BEHAVIOR_EMOTION_SCORES.get(behavior_label, BEHAVIOR_EMOTION_SCORES["normal"])
    return {label: float(scores.get(label, 0.0)) for label in STANDARD_EMOTION_LABELS}


def sentiment_bucket_for_behavior(label: Any) -> str:
    behavior_label = normalize_behavior_label(label)
    if behavior_label == "focused":
        return "positive"
    if behavior_label == "normal":
        return "neutral"
    if behavior_label == "distracted":
        return "slightly_negative"
    return "negative"

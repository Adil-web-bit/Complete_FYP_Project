from __future__ import annotations

from typing import Any


def clamp_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = float(default)
    return max(0.0, min(1.0, confidence))


def normalize_probabilities(probabilities: dict[str, Any], labels: list[str]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for label in labels:
        try:
            normalized[label] = max(0.0, float(probabilities.get(label, 0.0)))
        except (TypeError, ValueError):
            normalized[label] = 0.0

    total = sum(normalized.values())
    if total <= 0.0:
        fallback = 1.0 / float(len(labels)) if labels else 0.0
        return {label: fallback for label in labels}

    return {label: value / total for label, value in normalized.items()}


def apply_confidence_floor(probabilities: dict[str, float], confidence: float) -> dict[str, float]:
    """Keep calibrated probability shape while reflecting weak modality confidence."""
    confidence = clamp_confidence(confidence)
    if not probabilities:
        return {}
    uniform = 1.0 / float(len(probabilities))
    return {
        label: (value * confidence) + (uniform * (1.0 - confidence))
        for label, value in probabilities.items()
    }

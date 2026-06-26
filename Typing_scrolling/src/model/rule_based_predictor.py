from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALLOWED_LABELS: tuple[str, ...] = (
    "normal",
    "focused",
    "stressed",
    "rushed",
    "distracted",
)


def _f(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = features.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class RuleBasedBehaviorPredictor:
    """
    Safe baseline predictor for MVP demos (non-medical, heuristic-based).
    """

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        backspace_count = _f(features, "backspace_count")
        keys_per_second = _f(features, "keys_per_second")
        avg_key_interval_ms = _f(features, "avg_key_interval_ms")
        total_scroll_events = _f(features, "total_scroll_events")
        avg_scroll_speed = _f(features, "avg_scroll_speed")
        typed_text_length = _f(features, "typed_text_length")

        # Heuristics (intentionally simple and conservative)
        if typed_text_length < 5 and total_scroll_events < 2:
            return {
                "predicted_label": "normal",
                "confidence": 0.35,
                "reason": "Very little interaction data; defaulting to normal.",
                "model_type": "rule_based_baseline",
            }

        if keys_per_second >= 4.5 and avg_key_interval_ms > 0 and avg_key_interval_ms <= 220:
            return {
                "predicted_label": "rushed",
                "confidence": 0.72,
                "reason": "High typing speed with short average key intervals suggests rushed behavior.",
                "model_type": "rule_based_baseline",
            }

        if backspace_count >= 10 and avg_key_interval_ms > 0 and avg_key_interval_ms <= 260:
            return {
                "predicted_label": "stressed",
                "confidence": 0.68,
                "reason": "Frequent backspaces with relatively fast typing suggests stress or correction-heavy input.",
                "model_type": "rule_based_baseline",
            }

        if keys_per_second >= 2.0 and backspace_count <= 3 and avg_key_interval_ms > 0 and avg_key_interval_ms <= 450:
            return {
                "predicted_label": "focused",
                "confidence": 0.62,
                "reason": "Moderate-to-high typing speed with low corrections suggests focused behavior.",
                "model_type": "rule_based_baseline",
            }

        if total_scroll_events >= 25 and avg_scroll_speed >= 1200:
            return {
                "predicted_label": "distracted",
                "confidence": 0.6,
                "reason": "High volume and speed of scrolling suggests distracted or scanning behavior.",
                "model_type": "rule_based_baseline",
            }

        return {
            "predicted_label": "normal",
            "confidence": 0.5,
            "reason": "Signals are mixed; defaulting to normal baseline.",
            "model_type": "rule_based_baseline",
        }

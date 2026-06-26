from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .confidence_calibrator import clamp_confidence


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return uuid4().hex


def modality_prediction(
    *,
    modality: str,
    label: str,
    confidence: float,
    probabilities: dict[str, float],
    raw_label: str | None = None,
    features: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "modality": modality,
        "label": label,
        "raw_label": raw_label or label,
        "confidence": clamp_confidence(confidence),
        "probabilities": {str(k): float(v) for k, v in probabilities.items()},
        "features": features or {},
        "metadata": metadata or {},
    }


def empty_modality_prediction(modality: str, reason: str) -> dict[str, Any]:
    return modality_prediction(
        modality=modality,
        label="unavailable",
        raw_label="unavailable",
        confidence=0.0,
        probabilities={},
        metadata={"available": False, "reason": reason},
    )


def build_unified_prediction(
    *,
    session_id: str,
    face: dict[str, Any] | None,
    voice: dict[str, Any] | None,
    behavior: dict[str, Any] | None,
    final_prediction: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "timestamp": timestamp or utc_timestamp(),
        "face": face or {},
        "voice": voice or {},
        "behavior": behavior or {},
        "final_prediction": final_prediction,
        "confidence": clamp_confidence(final_prediction.get("confidence", 0.0)),
    }

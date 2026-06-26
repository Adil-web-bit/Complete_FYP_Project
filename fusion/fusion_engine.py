from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .confidence_calibrator import normalize_probabilities
from .fusion_features import FUSION_FEATURE_COLUMNS, build_feature_row
from .label_mapper import STANDARD_EMOTION_LABELS
from .train_fusion_model import BEST_MODEL_FILE


DEFAULT_WEIGHTS: dict[str, float] = {
    "face": 0.4,
    "voice": 0.4,
    "behavior": 0.2,
}


@dataclass(slots=True)
class FusionEngine:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    model_file: Path = BEST_MODEL_FILE
    use_trained_model: bool = True

    def fuse(
        self,
        *,
        face: dict[str, Any] | None = None,
        voice: dict[str, Any] | None = None,
        behavior: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.use_trained_model:
            trained_result = self._fuse_with_trained_model(
                face=face,
                voice=voice,
                behavior=behavior,
            )
            if trained_result is not None:
                return trained_result

        return self._weighted_fallback(face=face, voice=voice, behavior=behavior)

    def _weighted_fallback(
        self,
        *,
        face: dict[str, Any] | None = None,
        voice: dict[str, Any] | None = None,
        behavior: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        modalities = {
            "face": face,
            "voice": voice,
            "behavior": behavior,
        }

        active_weights: dict[str, float] = {}
        for modality, prediction in modalities.items():
            if not prediction:
                continue
            probabilities = prediction.get("probabilities")
            if not isinstance(probabilities, dict) or not probabilities:
                continue
            active_weights[modality] = max(0.0, float(self.weights.get(modality, 0.0)))

        total_weight = sum(active_weights.values())
        if total_weight <= 0.0:
            return {
                "label": "neutral",
                "confidence": 0.0,
                "scores": {label: 0.0 for label in STANDARD_EMOTION_LABELS},
                "weights": {},
                "modality_breakdown": {},
                "status": "no_modalities_available",
            }

        normalized_weights = {
            modality: weight / total_weight for modality, weight in active_weights.items()
        }

        fused_scores = {label: 0.0 for label in STANDARD_EMOTION_LABELS}
        modality_breakdown: dict[str, Any] = {}

        for modality, weight in normalized_weights.items():
            prediction = modalities[modality] or {}
            probs = normalize_probabilities(
                prediction.get("probabilities", {}),
                STANDARD_EMOTION_LABELS,
            )
            contribution = {label: probs[label] * weight for label in STANDARD_EMOTION_LABELS}
            for label, value in contribution.items():
                fused_scores[label] += value
            modality_breakdown[modality] = {
                "label": prediction.get("label"),
                "confidence": float(prediction.get("confidence", 0.0)),
                "weight": weight,
                "probabilities": probs,
                "weighted_contribution": contribution,
            }

        final_label = max(fused_scores, key=fused_scores.get)
        final_confidence = float(fused_scores[final_label])

        return {
            "label": final_label,
            "confidence": max(0.0, min(1.0, final_confidence)),
            "scores": fused_scores,
            "weights": normalized_weights,
            "modality_breakdown": modality_breakdown,
            "status": "weighted_fallback",
            "fusion_method": "weighted_late_fusion",
        }

    def _fuse_with_trained_model(
        self,
        *,
        face: dict[str, Any] | None,
        voice: dict[str, Any] | None,
        behavior: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not self.model_file.exists():
            return None

        try:
            artifact = joblib.load(self.model_file)
        except Exception:
            return None

        if not isinstance(artifact, dict):
            return None
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns")
        if model is None or not isinstance(feature_columns, list):
            return None

        try:
            pseudo_session = {
                "face": face or {},
                "voice": voice or {},
                "behavior": behavior or {},
                "final_prediction": {},
            }
            row = build_feature_row(pseudo_session)
            features = {
                str(column): float(row.get(str(column), 0.0))
                for column in feature_columns
            }
            x = pd.DataFrame([features], columns=[str(c) for c in feature_columns])

            if not hasattr(model, "predict_proba"):
                return None
            probabilities = model.predict_proba(x)[0]
            label_encoder = artifact.get("label_encoder")
            if label_encoder is not None:
                classes = [str(c) for c in label_encoder.inverse_transform(range(len(probabilities)))]
            else:
                classes = [str(c) for c in getattr(model, "classes_", artifact.get("classes", []))]

            if len(classes) != len(probabilities):
                classes = [str(c) for c in artifact.get("classes", [])]
            if len(classes) != len(probabilities):
                return None

            scores = {label: 0.0 for label in STANDARD_EMOTION_LABELS}
            extra_scores: dict[str, float] = {}
            for label, probability in zip(classes, probabilities):
                value = max(0.0, float(probability))
                if label in scores:
                    scores[label] = value
                else:
                    extra_scores[label] = value

            all_scores = {**scores, **extra_scores}
            final_label = max(all_scores, key=all_scores.get)
            final_confidence = float(all_scores[final_label])

            return {
                "label": final_label,
                "confidence": max(0.0, min(1.0, final_confidence)),
                "scores": all_scores,
                "weights": {},
                "modality_breakdown": {
                    "face": {"label": (face or {}).get("label"), "confidence": (face or {}).get("confidence")},
                    "voice": {"label": (voice or {}).get("label"), "confidence": (voice or {}).get("confidence")},
                    "behavior": {"label": (behavior or {}).get("label"), "confidence": (behavior or {}).get("confidence")},
                },
                "status": "ml_fused",
                "fusion_method": "trained_fusion_model",
                "model_name": artifact.get("model_name"),
                "model_file": str(self.model_file),
            }
        except Exception:
            return None


def fuse_predictions(
    face: dict[str, Any] | None = None,
    voice: dict[str, Any] | None = None,
    behavior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return FusionEngine().fuse(face=face, voice=voice, behavior=behavior)

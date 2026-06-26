from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.config import MODEL_FILE
from src.data_collection.monitoring_session import ALLOWED_BEHAVIOR_LABELS
from src.features.feature_extractor import FeatureExtractor
from src.model.rule_based_predictor import RuleBasedBehaviorPredictor


@dataclass(slots=True)
class BehaviorPredictor:
    """
    Behavior predictor that falls back to rule-based predictions when no model exists.
    """

    model_file: Path = MODEL_FILE
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    rule_based: RuleBasedBehaviorPredictor = field(default_factory=RuleBasedBehaviorPredictor)

    def _load_artifact(self) -> Optional[dict[str, Any]]:
        if not self.model_file.exists():
            return None
        try:
            import joblib
        except Exception:
            return None

        try:
            artifact = joblib.load(self.model_file)
        except Exception:
            return None

        if not isinstance(artifact, dict):
            return None
        if "pipeline" not in artifact or "feature_names" not in artifact:
            return None
        if not isinstance(artifact.get("feature_names"), list):
            return None
        return artifact

    def predict_from_session(self, session: dict[str, Any]) -> dict[str, Any]:
        features = self.extractor.extract_from_session(session)
        artifact = self._load_artifact()
        if artifact is not None:
            try:
                import pandas as pd

                pipeline = artifact["pipeline"]
                feature_names = [str(n) for n in artifact["feature_names"]]
                row = {name: float(features.get(name, 0.0)) for name in feature_names}
                x = pd.DataFrame([row], columns=feature_names)
                predicted_label = str(pipeline.predict(x)[0])
                confidence = 0.6
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(x)[0]
                    confidence = float(max(proba)) if len(proba) else 0.6

                if predicted_label not in ALLOWED_BEHAVIOR_LABELS:
                    predicted_label = "normal"
                    confidence = min(confidence, 0.5)

                return {
                    "predicted_label": predicted_label,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": "Predicted by trained ML model.",
                    "model_type": "trained_ml",
                    "source": artifact.get("source"),
                    "dataset_type": artifact.get("dataset_type"),
                    "features": features,
                }
            except Exception:
                # fall through to rule-based fallback
                pass

        baseline = self.rule_based.predict(features)
        return {
            "predicted_label": str(baseline.get("predicted_label", "normal")),
            "confidence": float(baseline.get("confidence", 0.5)),
            "reason": str(baseline.get("reason", "Rule-based fallback.")),
            "model_type": "rule_based_fallback",
            "features": features,
        }

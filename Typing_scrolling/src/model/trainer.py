from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EVALUATION_REPORT_FILE, MODEL_FILE, TRAINING_REPORT_FILE
from src.data_collection.monitoring_session import (
    ALLOWED_BEHAVIOR_LABELS,
    MonitoringSessionCollector,
)
from src.features.feature_extractor import FeatureExtractor


@dataclass(slots=True)
class ModelTrainer:
    """
    Train a supervised classifier from locally collected labeled monitoring sessions.
    """

    model_file: Path = MODEL_FILE
    training_report_file: Path = TRAINING_REPORT_FILE
    evaluation_report_file: Path = EVALUATION_REPORT_FILE
    sessions_collector: MonitoringSessionCollector = field(default_factory=MonitoringSessionCollector)
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    min_labeled_sessions: int = 10

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def load_labeled_sessions(self) -> list[dict[str, Any]]:
        sessions = self.sessions_collector.load_records()
        labeled: list[dict[str, Any]] = []
        for session in sessions:
            if not isinstance(session, dict):
                continue
            label = session.get("behavior_label")
            if not (isinstance(label, str) and label in ALLOWED_BEHAVIOR_LABELS):
                continue
            typing_events = session.get("typing_events", [])
            scroll_events = session.get("scroll_events", [])
            typing_count = len(typing_events) if isinstance(typing_events, list) else 0
            scroll_count = len(scroll_events) if isinstance(scroll_events, list) else 0
            if (typing_count + scroll_count) <= 0:
                continue
            labeled.append(session)
        return labeled

    def build_training_dataframe(self, sessions: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for session in sessions:
            if not isinstance(session, dict):
                continue
            label = session.get("behavior_label")
            if not isinstance(label, str) or label not in ALLOWED_BEHAVIOR_LABELS:
                continue
            features = self.extractor.extract_from_session(session)
            row: dict[str, Any] = {k: float(v) for k, v in features.items()}
            row["behavior_label"] = label
            rows.append(row)
        return pd.DataFrame(rows)

    def train(self) -> dict[str, Any]:
        trained_at_utc = self._utc_now_iso()
        try:
            labeled_sessions = self.load_labeled_sessions()
            labeled_count = len(labeled_sessions)
            label_counts: dict[str, int] = {}
            for session in labeled_sessions:
                label = session.get("behavior_label")
                if isinstance(label, str):
                    label_counts[label] = label_counts.get(label, 0) + 1

            if labeled_count < self.min_labeled_sessions:
                report = {
                    "status": "not_enough_data",
                    "message": "Not enough labeled sessions to train a model.",
                    "training_count": labeled_count,
                    "required_minimum": self.min_labeled_sessions,
                    "label_counts": label_counts,
                    "feature_names": [],
                    "validation_accuracy": None,
                    "model_file": str(self.model_file),
                    "evaluation_report_file": str(self.evaluation_report_file),
                    "trained_at_utc": trained_at_utc,
                }
                self._save_report(report)
                return report

            if len(label_counts) < 2:
                report = {
                    "status": "not_enough_label_diversity",
                    "message": "Need at least 2 unique labels to train a classifier.",
                    "training_count": labeled_count,
                    "label_counts": label_counts,
                    "feature_names": [],
                    "validation_accuracy": None,
                    "model_file": str(self.model_file),
                    "evaluation_report_file": str(self.evaluation_report_file),
                    "trained_at_utc": trained_at_utc,
                }
                self._save_report(report)
                return report

            df = self.build_training_dataframe(labeled_sessions)
            if df.empty or "behavior_label" not in df.columns:
                report = {
                    "status": "error",
                    "message": "No valid labeled rows were produced for training.",
                    "training_count": 0,
                    "label_counts": label_counts,
                    "feature_names": [],
                    "validation_accuracy": None,
                    "model_file": str(self.model_file),
                    "evaluation_report_file": str(self.evaluation_report_file),
                    "trained_at_utc": trained_at_utc,
                }
                self._save_report(report)
                return report

            y = df["behavior_label"].astype(str)
            x = df.drop(columns=["behavior_label"])
            feature_names = list(x.columns.astype(str))

            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

            validation_accuracy: Optional[float] = None
            did_validation = False
            evaluation: dict[str, Any] = {
                "trained_at_utc": trained_at_utc,
                "training_count": int(labeled_count),
                "validation_count": 0,
                "label_counts": label_counts,
                "feature_names": feature_names,
                "validation_accuracy": None,
                "classification_report": None,
                "confusion_matrix": None,
                "labels_order": sorted(label_counts.keys()),
                "notes": "",
            }

            # Validation is only meaningful if each class can appear in both train and test.
            min_class_count = min(label_counts.values())
            if labeled_count >= 20 and min_class_count >= 3:
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import classification_report, confusion_matrix

                    x_train, x_test, y_train, y_test = train_test_split(
                        x,
                        y,
                        test_size=0.2,
                        random_state=42,
                        stratify=y,
                    )
                    pipeline.fit(x_train, y_train)
                    validation_accuracy = float(pipeline.score(x_test, y_test))
                    y_pred = pipeline.predict(x_test)
                    labels_order = sorted(label_counts.keys())
                    evaluation.update(
                        {
                            "status": "validated",
                            "validation_count": int(len(y_test)),
                            "validation_accuracy": validation_accuracy,
                            "classification_report": classification_report(
                                y_test,
                                y_pred,
                                labels=labels_order,
                                output_dict=True,
                                zero_division=0,
                            ),
                            "confusion_matrix": confusion_matrix(
                                y_test,
                                y_pred,
                                labels=labels_order,
                            ).tolist(),
                            "labels_order": labels_order,
                            "notes": "Validation performed with train_test_split(test_size=0.2, stratify=y).",
                        }
                    )
                    did_validation = True
                except Exception:
                    did_validation = False

            if not did_validation:
                pipeline.fit(x, y)
                evaluation.update(
                    {
                        "status": "validation_skipped",
                        "validation_count": 0,
                        "validation_accuracy": None,
                        "classification_report": None,
                        "confusion_matrix": None,
                        "notes": "Validation skipped due to small dataset / insufficient per-class counts.",
                    }
                )

            artifact = {
                "pipeline": pipeline,
                "feature_names": feature_names,
                "labels": sorted(label_counts.keys()),
                "trained_at_utc": trained_at_utc,
                "training_count": int(labeled_count),
                "validation_accuracy": validation_accuracy,
            }

            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                import joblib

                joblib.dump(artifact, self.model_file)
            except Exception as exc:
                report = {
                    "status": "error",
                    "message": f"Failed to save model artifact: {exc}",
                    "training_count": labeled_count,
                    "label_counts": label_counts,
                    "feature_names": feature_names,
                    "validation_accuracy": validation_accuracy,
                    "model_file": str(self.model_file),
                    "evaluation_report_file": str(self.evaluation_report_file),
                    "trained_at_utc": trained_at_utc,
                }
                self._save_report(report)
                return report

            report = {
                "status": "trained",
                "message": "Model trained and saved.",
                "training_count": labeled_count,
                "label_counts": label_counts,
                "feature_names": feature_names,
                "validation_accuracy": validation_accuracy,
                "model_file": str(self.model_file),
                "evaluation_report_file": str(self.evaluation_report_file),
                "trained_at_utc": trained_at_utc,
            }
            self._save_evaluation_report(evaluation)
            self._save_report(report)
            return report

        except Exception as exc:
            report = {
                "status": "error",
                "message": f"Training failed: {exc}",
                "training_count": 0,
                "label_counts": {},
                "feature_names": [],
                "validation_accuracy": None,
                "model_file": str(self.model_file),
                "evaluation_report_file": str(self.evaluation_report_file),
                "trained_at_utc": trained_at_utc,
            }
            self._save_report(report)
            return report

    def _save_report(self, report: dict[str, Any]) -> None:
        self.training_report_file.parent.mkdir(parents=True, exist_ok=True)
        with self.training_report_file.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

    def _save_evaluation_report(self, report: dict[str, Any]) -> None:
        self.evaluation_report_file.parent.mkdir(parents=True, exist_ok=True)
        with self.evaluation_report_file.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

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

from src.config import (
    MODEL_FILE,
    PUBLIC_DATASET_FILE,
    PUBLIC_EVALUATION_REPORT_FILE,
    PUBLIC_TRAINING_REPORT_FILE,
)
from src.data_collection.public_dataset_loader import PublicDatasetLoader


@dataclass(slots=True)
class PublicDatasetTrainer:
    model_file: Path = MODEL_FILE
    public_training_report_file: Path = PUBLIC_TRAINING_REPORT_FILE
    public_evaluation_report_file: Path = PUBLIC_EVALUATION_REPORT_FILE
    default_dataset_path: Path = PUBLIC_DATASET_FILE
    loader: PublicDatasetLoader = field(default_factory=PublicDatasetLoader)
    min_rows: int = 10

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def train_synthetic_typing_demo(self, n_rows: int = 250) -> dict[str, Any]:
        df = self.loader.generate_synthetic_typing_demo_dataset(n_rows=int(n_rows))
        return self.train_from_dataframe(df, source_name="synthetic_typing_demo")

    def train_from_csv(self, path: Path) -> dict[str, Any]:
        df = self.loader.load_csv(path)
        return self.train_from_dataframe(df, source_name=str(path))

    def train_from_dataframe(self, df: pd.DataFrame, source_name: str = "public_dataset") -> dict[str, Any]:
        trained_at_utc = self._utc_now_iso()
        try:
            feature_df = self.loader.build_feature_dataframe(df)
            if feature_df.shape[0] < self.min_rows:
                report = self._report(
                    status="not_enough_data",
                    message="Not enough rows after cleaning to train.",
                    trained_at_utc=trained_at_utc,
                    training_count=int(feature_df.shape[0]),
                    validation_count=0,
                    label_counts=self._label_counts(feature_df),
                    feature_names=[],
                    validation_accuracy=None,
                    source=source_name,
                    notes="Need at least 10 rows after cleaning.",
                )
                self._save_reports(report, evaluation=None)
                return report

            label_counts = self._label_counts(feature_df)
            if len(label_counts) < 2:
                report = self._report(
                    status="not_enough_label_diversity",
                    message="Need at least 2 unique labels after mapping to train.",
                    trained_at_utc=trained_at_utc,
                    training_count=int(feature_df.shape[0]),
                    validation_count=0,
                    label_counts=label_counts,
                    feature_names=[],
                    validation_accuracy=None,
                    source=source_name,
                    notes="Label mapping may have dropped rows; ensure dataset has multiple labels.",
                )
                self._save_reports(report, evaluation=None)
                return report

            y = feature_df["behavior_label"].astype(str)
            x = feature_df.drop(columns=["behavior_label"])
            feature_names = list(x.columns.astype(str))

            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

            validation_accuracy: Optional[float] = None
            evaluation: dict[str, Any] = {
                "status": "validation_skipped",
                "source": source_name,
                "dataset_type": "typing_public_dataset",
                "trained_at_utc": trained_at_utc,
                "training_count": int(feature_df.shape[0]),
                "validation_count": 0,
                "label_counts": label_counts,
                "feature_names": feature_names,
                "validation_accuracy": None,
                "classification_report": None,
                "confusion_matrix": None,
                "labels_order": sorted(label_counts.keys()),
                "notes": "Validation skipped due to small dataset / insufficient per-class counts.",
            }

            did_validation = False
            min_class_count = min(label_counts.values())
            if feature_df.shape[0] >= 20 and min_class_count >= 3:
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
                                y_test, y_pred, labels=labels_order, output_dict=True, zero_division=0
                            ),
                            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_order).tolist(),
                            "labels_order": labels_order,
                            "notes": "Validation performed with train_test_split(test_size=0.2, stratify=y).",
                        }
                    )
                    did_validation = True
                except Exception:
                    did_validation = False

            if not did_validation:
                pipeline.fit(x, y)

            artifact = {
                "pipeline": pipeline,
                "feature_names": feature_names,
                "labels": sorted(label_counts.keys()),
                "trained_at_utc": trained_at_utc,
                "training_count": int(feature_df.shape[0]),
                "validation_accuracy": validation_accuracy,
                "source": source_name,
                "dataset_type": "typing_public_dataset",
                "label_mapping_note": "Labels are mapped into app labels: neutral->normal, calmness/happiness->focused, anger->stressed, sadness->distracted.",
            }

            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            import joblib

            joblib.dump(artifact, self.model_file)

            report = self._report(
                status="trained",
                message="Public dataset model trained and saved.",
                trained_at_utc=trained_at_utc,
                training_count=int(feature_df.shape[0]),
                validation_count=int(evaluation.get("validation_count", 0)),
                label_counts=label_counts,
                feature_names=feature_names,
                validation_accuracy=validation_accuracy,
                source=source_name,
                notes="Public typing datasets may not match browser-captured data. Use for demo/baseline only.",
            )
            self._save_reports(report, evaluation=evaluation)
            return report

        except Exception as exc:
            report = self._report(
                status="error",
                message=f"Public training failed: {exc}",
                trained_at_utc=trained_at_utc,
                training_count=0,
                validation_count=0,
                label_counts={},
                feature_names=[],
                validation_accuracy=None,
                source=source_name,
                notes="Fix dataset format and retry.",
            )
            self._save_reports(report, evaluation=None)
            return report

    @staticmethod
    def _label_counts(df: pd.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        if "behavior_label" not in df.columns:
            return counts
        for v in df["behavior_label"].astype(str).tolist():
            counts[v] = counts.get(v, 0) + 1
        return counts

    def _save_reports(self, training_report: dict[str, Any], evaluation: dict[str, Any] | None) -> None:
        self.public_training_report_file.parent.mkdir(parents=True, exist_ok=True)
        with self.public_training_report_file.open("w", encoding="utf-8") as f:
            json.dump(training_report, f, ensure_ascii=False, indent=2)
        if evaluation is not None:
            self.public_evaluation_report_file.parent.mkdir(parents=True, exist_ok=True)
            with self.public_evaluation_report_file.open("w", encoding="utf-8") as f:
                json.dump(evaluation, f, ensure_ascii=False, indent=2)

    def _report(
        self,
        status: str,
        message: str,
        trained_at_utc: str,
        training_count: int,
        validation_count: int,
        label_counts: dict[str, int],
        feature_names: list[str],
        validation_accuracy: float | None,
        source: str,
        notes: str,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "message": message,
            "source": source,
            "dataset_type": "typing_public_dataset",
            "trained_at_utc": trained_at_utc,
            "training_count": int(training_count),
            "validation_count": int(validation_count),
            "label_counts": label_counts,
            "feature_names": feature_names,
            "validation_accuracy": validation_accuracy,
            "model_file": str(self.model_file),
            "public_training_report_file": str(self.public_training_report_file),
            "public_evaluation_report_file": str(self.public_evaluation_report_file),
            "notes": notes,
        }


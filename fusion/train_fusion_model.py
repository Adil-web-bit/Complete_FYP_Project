from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .dataset_builder import DEFAULT_DATASET_FILE, FusionDatasetBuilder
from .fusion_features import FUSION_FEATURE_COLUMNS, TARGET_COLUMN
from .synthetic_data_generator import (
    ALLOWED_SYNTHETIC_SIZES,
    SYNTHETIC_DATASET_FILE,
    SyntheticFusionDataGenerator,
)


MODEL_DIR = Path(__file__).resolve().parent / "models"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
BEST_MODEL_FILE = MODEL_DIR / "best_fusion_model.joblib"
EVALUATION_REPORT_FILE = REPORT_DIR / "fusion_evaluation.json"


@dataclass(slots=True)
class FusionModelTrainer:
    dataset_builder: FusionDatasetBuilder = field(default_factory=FusionDatasetBuilder)
    synthetic_generator: SyntheticFusionDataGenerator = field(default_factory=SyntheticFusionDataGenerator)
    best_model_file: Path = BEST_MODEL_FILE
    evaluation_report_file: Path = EVALUATION_REPORT_FILE
    min_rows: int = 10

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def train_from_sessions(self, synthetic_size: int | None = 1000) -> dict[str, Any]:
        real_df = self.dataset_builder.build_training_dataframe()
        if not real_df.empty:
            real_df = real_df.copy()
            real_df["data_source"] = "real"
            real_df["sample_weight"] = real_df.get("target_source", pd.Series(dtype=str)).astype(str).map(
                lambda source: 5.0 if not source.startswith("weak_") else 3.0
            )

        synthetic_df = pd.DataFrame()
        if synthetic_size is not None and int(synthetic_size) > 0:
            synthetic_df = self.synthetic_generator.generate_dataframe(int(synthetic_size))
            self.synthetic_generator.output_file.parent.mkdir(parents=True, exist_ok=True)
            synthetic_df.to_csv(self.synthetic_generator.output_file, index=False)

        combined = self._merge_real_and_synthetic(real_df, synthetic_df)
        self.dataset_builder.save_dataset(combined)
        source = {
            "real_sessions": str(self.dataset_builder.session_dir),
            "synthetic_dataset": str(self.synthetic_generator.output_file) if not synthetic_df.empty else None,
            "combined_dataset": str(self.dataset_builder.dataset_file),
        }
        return self.train_from_dataframe(combined, source=json.dumps(source))

    def train_from_csv(self, path: Path = DEFAULT_DATASET_FILE) -> dict[str, Any]:
        df = pd.read_csv(path)
        return self.train_from_dataframe(df, source=str(path))

    def train_from_dataframe(self, df: pd.DataFrame, source: str = "dataframe") -> dict[str, Any]:
        trained_at = self._utc_now_iso()
        report_base: dict[str, Any] = {
            "status": "not_trained",
            "trained_at_utc": trained_at,
            "source": source,
            "dataset_file": str(self.dataset_builder.dataset_file),
            "best_model_file": str(self.best_model_file),
            "feature_columns": FUSION_FEATURE_COLUMNS,
            "model_comparison": [],
            "best_model": None,
            "best_metrics": None,
            "synthetic_dataset_file": str(SYNTHETIC_DATASET_FILE),
        }

        if df.empty or TARGET_COLUMN not in df.columns:
            return self._save_report(
                report_base
                | {
                    "status": "not_enough_data",
                    "message": "No fusion training rows found. Save multimodal sessions with target labels first.",
                    "training_count": 0,
                }
            )

        train_df = df.dropna(subset=[TARGET_COLUMN]).copy()
        train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(str).str.strip().str.lower()
        train_df = train_df[train_df[TARGET_COLUMN] != ""].reset_index(drop=True)

        for column in FUSION_FEATURE_COLUMNS:
            if column not in train_df.columns:
                train_df[column] = 0.0
            train_df[column] = pd.to_numeric(train_df[column], errors="coerce").fillna(0.0)

        label_counts = train_df[TARGET_COLUMN].value_counts().to_dict()
        source_counts = train_df.get("data_source", pd.Series(["unknown"] * len(train_df))).astype(str).value_counts().to_dict()
        target_source_counts = train_df.get("target_source", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        if int(train_df.shape[0]) < self.min_rows:
            return self._save_report(
                report_base
                | {
                    "status": "not_enough_data",
                    "message": f"Need at least {self.min_rows} labeled fusion rows.",
                    "training_count": int(train_df.shape[0]),
                    "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                    "class_distribution": {str(k): int(v) for k, v in label_counts.items()},
                    "data_source_counts": {str(k): int(v) for k, v in source_counts.items()},
                }
            )
        if len(label_counts) < 2:
            return self._save_report(
                report_base
                | {
                    "status": "not_enough_label_diversity",
                    "message": "Need at least 2 target labels to train a fusion classifier.",
                    "training_count": int(train_df.shape[0]),
                    "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                    "class_distribution": {str(k): int(v) for k, v in label_counts.items()},
                    "data_source_counts": {str(k): int(v) for k, v in source_counts.items()},
                }
            )

        x = train_df[FUSION_FEATURE_COLUMNS]
        y = train_df[TARGET_COLUMN]
        sample_weight = pd.to_numeric(train_df.get("sample_weight", 1.0), errors="coerce").fillna(1.0)

        stratify = y if min(label_counts.values()) >= 2 else None
        split_values = train_test_split(
            x,
            y,
            sample_weight,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        x_train, x_test, y_train, y_test, w_train, _w_test = split_values

        candidates = self._candidate_models()
        model_comparison: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for name, estimator, label_encoder in candidates:
            try:
                if label_encoder is not None:
                    y_train_fit = label_encoder.fit_transform(y_train)
                    self._fit_estimator(name, estimator, x_train, y_train_fit, w_train)
                    encoded_pred = estimator.predict(x_test)
                    y_pred = label_encoder.inverse_transform(encoded_pred)
                    proba_classes = list(label_encoder.classes_)
                else:
                    self._fit_estimator(name, estimator, x_train, y_train, w_train)
                    y_pred = estimator.predict(x_test)
                    proba_classes = [str(c) for c in getattr(estimator, "classes_", sorted(label_counts.keys()))]

                metrics = self._metrics(y_test, y_pred, labels=sorted(label_counts.keys()))
                entry = {
                    "model_name": name,
                    "status": "trained",
                    **metrics,
                }
                model_comparison.append(entry)

                score = (metrics["f1_weighted"], metrics["accuracy"])
                if best is None or score > (best["metrics"]["f1_weighted"], best["metrics"]["accuracy"]):
                    best = {
                        "model_name": name,
                        "estimator": estimator,
                        "label_encoder": label_encoder,
                        "proba_classes": proba_classes,
                        "metrics": metrics,
                    }
            except Exception as exc:
                model_comparison.append(
                    {
                        "model_name": name,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        if best is None:
            return self._save_report(
                report_base
                | {
                    "status": "error",
                    "message": "All fusion model candidates failed.",
                    "training_count": int(train_df.shape[0]),
                    "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                    "model_comparison": model_comparison,
                }
            )

        final_estimator = best["estimator"]
        final_label_encoder = best["label_encoder"]
        if final_label_encoder is not None:
            final_label_encoder = LabelEncoder().fit(y)
            final_estimator = self._candidate_by_name(best["model_name"], final_label_encoder)
            self._fit_estimator(best["model_name"], final_estimator, x, final_label_encoder.transform(y), sample_weight)
            final_classes = list(final_label_encoder.classes_)
        else:
            final_estimator = self._candidate_by_name(best["model_name"], None)
            self._fit_estimator(best["model_name"], final_estimator, x, y, sample_weight)
            final_classes = [str(c) for c in getattr(final_estimator, "classes_", sorted(label_counts.keys()))]

        artifact = {
            "model": final_estimator,
            "model_name": best["model_name"],
            "feature_columns": FUSION_FEATURE_COLUMNS,
            "labels": sorted(label_counts.keys()),
            "classes": final_classes,
            "label_encoder": final_label_encoder,
            "trained_at_utc": trained_at,
            "training_count": int(train_df.shape[0]),
            "data_source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "class_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "metrics": best["metrics"],
        }
        self.best_model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.best_model_file)

        report = report_base | {
            "status": "trained",
            "message": "Fusion model trained and best model saved.",
            "training_count": int(train_df.shape[0]),
            "validation_count": int(len(y_test)),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "class_distribution": {str(k): int(v) for k, v in label_counts.items()},
            "data_source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "model_comparison": model_comparison,
            "best_model": best["model_name"],
            "best_metrics": best["metrics"],
            "target_source_counts": {str(k): int(v) for k, v in target_source_counts.items()},
        }
        return self._save_report(report)

    @staticmethod
    def _merge_real_and_synthetic(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if real_df is not None and not real_df.empty:
            frames.append(real_df)
        if synthetic_df is not None and not synthetic_df.empty:
            frames.append(synthetic_df)
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, sort=False)
        for column in FUSION_FEATURE_COLUMNS:
            if column not in combined.columns:
                combined[column] = 0.0
        if "data_source" not in combined.columns:
            combined["data_source"] = "unknown"
        if "sample_weight" not in combined.columns:
            combined["sample_weight"] = 1.0
        combined["sample_weight"] = pd.to_numeric(combined["sample_weight"], errors="coerce").fillna(1.0)
        return combined

    @staticmethod
    def _fit_estimator(name: str, estimator: Any, x, y, sample_weight) -> None:
        if name == "Logistic Regression":
            estimator.fit(x, y, clf__sample_weight=sample_weight)
        else:
            estimator.fit(x, y, sample_weight=sample_weight)

    def _candidate_models(self):
        models: list[tuple[str, Any, LabelEncoder | None]] = [
            (
                "Logistic Regression",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=1000,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                None,
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
                None,
            ),
        ]

        try:
            from xgboost import XGBClassifier

            models.append(
                (
                    "XGBoost",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                    LabelEncoder(),
                )
            )
        except Exception:
            models.append(("XGBoost", _UnavailableEstimator("xgboost is not installed or could not be imported."), None))

        return models

    def _candidate_by_name(self, name: str, label_encoder: LabelEncoder | None):
        for candidate_name, estimator, _ in self._candidate_models():
            if candidate_name == name:
                return estimator
        raise ValueError(f"Unknown model candidate: {name}")

    @staticmethod
    def _metrics(y_true, y_pred, labels: list[str]) -> dict[str, Any]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "labels_order": labels,
        }

    def _save_report(self, report: dict[str, Any]) -> dict[str, Any]:
        self.evaluation_report_file.parent.mkdir(parents=True, exist_ok=True)
        with self.evaluation_report_file.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        return report


class _UnavailableEstimator:
    def __init__(self, message: str):
        self.message = message

    def fit(self, *_args, **_kwargs):
        raise RuntimeError(self.message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal fusion model.")
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=1000,
        choices=ALLOWED_SYNTHETIC_SIZES,
        help="Number of synthetic fusion samples to generate and merge with real sessions.",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Train only from real saved multimodal sessions.",
    )
    args = parser.parse_args()
    synthetic_size = None if args.no_synthetic else int(args.synthetic_size)
    report = FusionModelTrainer().train_from_sessions(synthetic_size=synthetic_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

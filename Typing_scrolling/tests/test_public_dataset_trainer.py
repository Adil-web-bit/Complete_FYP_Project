from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.data_collection.public_dataset_loader import PublicDatasetLoader  # noqa: E402
from src.model.public_dataset_trainer import PublicDatasetTrainer  # noqa: E402
from src.model.predictor import BehaviorPredictor  # noqa: E402


def test_public_trainer_synthetic_trains_and_writes_reports(tmp_path: Path) -> None:
    trainer = PublicDatasetTrainer(
        model_file=tmp_path / "behavior_model.joblib",
        public_training_report_file=tmp_path / "public_training_report.json",
        public_evaluation_report_file=tmp_path / "public_evaluation_report.json",
    )
    report = trainer.train_synthetic_typing_demo(n_rows=250)
    assert report["status"] == "trained"
    assert Path(report["model_file"]).exists()
    assert Path(report["public_training_report_file"]).exists()
    assert Path(report["public_evaluation_report_file"]).exists()


def test_public_trainer_not_enough_data(tmp_path: Path) -> None:
    df = pd.DataFrame({"emotion": ["neutral"] * 5, "x": [1, 2, 3, 4, 5]})
    trainer = PublicDatasetTrainer(
        model_file=tmp_path / "behavior_model.joblib",
        public_training_report_file=tmp_path / "public_training_report.json",
        public_evaluation_report_file=tmp_path / "public_evaluation_report.json",
    )
    report = trainer.train_from_dataframe(df, source_name="tiny")
    assert report["status"] == "not_enough_data"


def test_public_trainer_not_enough_label_diversity(tmp_path: Path) -> None:
    df = pd.DataFrame({"emotion": ["neutral"] * 15, "x": list(range(15))})
    trainer = PublicDatasetTrainer(
        model_file=tmp_path / "behavior_model.joblib",
        public_training_report_file=tmp_path / "public_training_report.json",
        public_evaluation_report_file=tmp_path / "public_evaluation_report.json",
    )
    report = trainer.train_from_dataframe(df, source_name="one_label")
    assert report["status"] == "not_enough_label_diversity"


def test_behavior_predictor_can_predict_with_public_artifact(tmp_path: Path) -> None:
    trainer = PublicDatasetTrainer(
        model_file=tmp_path / "behavior_model.joblib",
        public_training_report_file=tmp_path / "public_training_report.json",
        public_evaluation_report_file=tmp_path / "public_evaluation_report.json",
    )
    trainer.train_synthetic_typing_demo(n_rows=250)

    predictor = BehaviorPredictor(model_file=tmp_path / "behavior_model.joblib")
    # Use session-style features: predictor will align to artifact feature_names and fill missing with 0.
    session = {
        "typed_text": "hello",
        "typing_events": [{"timestamp_ms": 1000, "key": "h"}, {"timestamp_ms": 1100, "key": "e"}],
        "scroll_events": [],
        "started_at_ms": 1000,
        "stopped_at_ms": 2000,
    }
    result = predictor.predict_from_session(session)
    assert result["model_type"] in ("trained_ml", "rule_based_fallback")
    assert "predicted_label" in result


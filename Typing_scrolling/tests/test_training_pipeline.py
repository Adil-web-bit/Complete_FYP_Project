from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.data_collection.monitoring_session import (  # noqa: E402
    ALLOWED_BEHAVIOR_LABELS,
    MonitoringSessionCollector,
)
from src.model.predictor import BehaviorPredictor  # noqa: E402
from src.model.trainer import ModelTrainer  # noqa: E402


def _make_typing_events(start_ms: int, n: int, key: str = "a") -> list[dict]:
    return [
        {
            "type": "keydown",
            "key": key if i != 1 else "Backspace",
            "code": "KeyA",
            "timestamp_ms": start_ms + i * 100,
            "text_length": i,
        }
        for i in range(n)
    ]


def _make_scroll_events(start_ms: int, positions: list[int]) -> list[dict]:
    return [
        {
            "type": "scroll",
            "scroll_top": float(pos),
            "scroll_height": 2000.0,
            "client_height": 400.0,
            "timestamp_ms": start_ms + i * 200,
        }
        for i, pos in enumerate(positions)
    ]


def _write_labeled_sessions(
    collector: MonitoringSessionCollector,
    labels: list[str],
) -> None:
    for i, label in enumerate(labels):
        record = collector.create_record(
            user_id="u1",
            consent_given=True,
            behavior_label=label,
            typed_text="hello world",
            typing_events=_make_typing_events(1000 + i * 1000, n=5 + (i % 3), key="a"),
            scroll_events=_make_scroll_events(2000 + i * 1000, positions=[0, 50 + i, 20]),
            started_at_ms=1000 + i * 1000,
            stopped_at_ms=6000 + i * 1000,
        )
        collector.save_record(record)


def test_model_trainer_build_training_dataframe(tmp_path: Path) -> None:
    sessions_file = tmp_path / "sessions.jsonl"
    collector = MonitoringSessionCollector(sessions_file=sessions_file)
    _write_labeled_sessions(collector, [ALLOWED_BEHAVIOR_LABELS[0], ALLOWED_BEHAVIOR_LABELS[1]])

    trainer = ModelTrainer(
        sessions_collector=collector,
        model_file=tmp_path / "model.joblib",
        training_report_file=tmp_path / "training_report.json",
        evaluation_report_file=tmp_path / "evaluation_report.json",
    )
    sessions = trainer.load_labeled_sessions()
    df = trainer.build_training_dataframe(sessions)
    assert "behavior_label" in df.columns
    assert len(df) == 2
    assert df.drop(columns=["behavior_label"]).select_dtypes(include="number").shape[1] > 0


def test_model_trainer_train_not_enough_data(tmp_path: Path) -> None:
    sessions_file = tmp_path / "sessions.jsonl"
    collector = MonitoringSessionCollector(sessions_file=sessions_file)
    _write_labeled_sessions(collector, [ALLOWED_BEHAVIOR_LABELS[0]] * 9)

    trainer = ModelTrainer(
        sessions_collector=collector,
        model_file=tmp_path / "model.joblib",
        training_report_file=tmp_path / "training_report.json",
        evaluation_report_file=tmp_path / "evaluation_report.json",
    )
    report = trainer.train()
    assert report["status"] == "not_enough_data"
    assert report["training_count"] == 9


def test_model_trainer_train_not_enough_label_diversity(tmp_path: Path) -> None:
    sessions_file = tmp_path / "sessions.jsonl"
    collector = MonitoringSessionCollector(sessions_file=sessions_file)
    _write_labeled_sessions(collector, [ALLOWED_BEHAVIOR_LABELS[0]] * 10)

    trainer = ModelTrainer(
        sessions_collector=collector,
        model_file=tmp_path / "model.joblib",
        training_report_file=tmp_path / "training_report.json",
        evaluation_report_file=tmp_path / "evaluation_report.json",
    )
    report = trainer.train()
    assert report["status"] == "not_enough_label_diversity"


def test_model_trainer_can_train_and_save_model(tmp_path: Path) -> None:
    sessions_file = tmp_path / "sessions.jsonl"
    collector = MonitoringSessionCollector(sessions_file=sessions_file)
    labels = [ALLOWED_BEHAVIOR_LABELS[0]] * 5 + [ALLOWED_BEHAVIOR_LABELS[2]] * 5
    _write_labeled_sessions(collector, labels)

    model_file = tmp_path / "model.joblib"
    report_file = tmp_path / "training_report.json"
    eval_file = tmp_path / "evaluation_report.json"
    trainer = ModelTrainer(
        sessions_collector=collector,
        model_file=model_file,
        training_report_file=report_file,
        evaluation_report_file=eval_file,
    )
    report = trainer.train()
    assert report["status"] == "trained"
    assert model_file.exists()
    assert report_file.exists()
    assert eval_file.exists()
    import json

    evaluation = json.loads(eval_file.read_text(encoding="utf-8"))
    assert evaluation["status"] in ("validated", "validation_skipped")
    assert "label_counts" in evaluation
    assert "feature_names" in evaluation


def test_behavior_predictor_uses_trained_model_when_available(tmp_path: Path) -> None:
    sessions_file = tmp_path / "sessions.jsonl"
    collector = MonitoringSessionCollector(sessions_file=sessions_file)
    labels = [ALLOWED_BEHAVIOR_LABELS[0]] * 5 + [ALLOWED_BEHAVIOR_LABELS[1]] * 5
    _write_labeled_sessions(collector, labels)

    model_file = tmp_path / "model.joblib"
    trainer = ModelTrainer(
        sessions_collector=collector,
        model_file=model_file,
        training_report_file=tmp_path / "training_report.json",
        evaluation_report_file=tmp_path / "evaluation_report.json",
    )
    report = trainer.train()
    assert report["status"] == "trained"

    predictor = BehaviorPredictor(model_file=model_file)
    session = collector.load_records()[0]
    result = predictor.predict_from_session(session)
    assert result["model_type"] == "trained_ml"
    assert "predicted_label" in result
    assert "features" in result


def test_model_trainer_writes_validation_metrics_when_possible(tmp_path: Path) -> None:
    sessions_file = tmp_path / "sessions.jsonl"
    collector = MonitoringSessionCollector(sessions_file=sessions_file)
    labels = [ALLOWED_BEHAVIOR_LABELS[0]] * 12 + [ALLOWED_BEHAVIOR_LABELS[1]] * 12
    _write_labeled_sessions(collector, labels)

    eval_file = tmp_path / "evaluation_report.json"
    trainer = ModelTrainer(
        sessions_collector=collector,
        model_file=tmp_path / "model.joblib",
        training_report_file=tmp_path / "training_report.json",
        evaluation_report_file=eval_file,
    )
    report = trainer.train()
    assert report["status"] == "trained"
    assert eval_file.exists()

    import json

    evaluation = json.loads(eval_file.read_text(encoding="utf-8"))
    assert evaluation["status"] in ("validated", "validation_skipped")
    if evaluation["status"] == "validated":
        assert evaluation["classification_report"] is not None
        assert evaluation["confusion_matrix"] is not None

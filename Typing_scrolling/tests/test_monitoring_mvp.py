from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.data_collection.monitoring_session import (  # noqa: E402
    ALLOWED_BEHAVIOR_LABELS,
    MonitoringSessionCollector,
)
from src.features.feature_extractor import FeatureExtractor  # noqa: E402
from src.model.predictor import BehaviorPredictor  # noqa: E402
from src.model.rule_based_predictor import RuleBasedBehaviorPredictor  # noqa: E402


def test_monitoring_session_create_record_valid(tmp_path: Path) -> None:
    collector = MonitoringSessionCollector(sessions_file=tmp_path / "sessions.jsonl")
    record = collector.create_record(
        user_id="u1",
        consent_given=True,
        behavior_label=ALLOWED_BEHAVIOR_LABELS[0],
        typed_text="hello",
        typing_events=[],
        scroll_events=[],
        started_at_ms=1000,
        stopped_at_ms=4000,
    )
    assert record["user_id"] == "u1"
    assert record["consent_given"] is True
    assert record["behavior_label"] == ALLOWED_BEHAVIOR_LABELS[0]
    assert record["duration_seconds"] == 3.0
    assert record["source"] == "streamlit_js_component"


def test_monitoring_session_rejects_missing_user_id(tmp_path: Path) -> None:
    collector = MonitoringSessionCollector(sessions_file=tmp_path / "sessions.jsonl")
    with pytest.raises(ValueError):
        collector.create_record(
            user_id="   ",
            consent_given=True,
            behavior_label=None,
            typed_text="",
            typing_events=[],
            scroll_events=[],
            started_at_ms=None,
            stopped_at_ms=None,
        )


def test_monitoring_session_rejects_consent_false(tmp_path: Path) -> None:
    collector = MonitoringSessionCollector(sessions_file=tmp_path / "sessions.jsonl")
    with pytest.raises(ValueError):
        collector.create_record(
            user_id="u1",
            consent_given=False,
            behavior_label=None,
            typed_text="",
            typing_events=[],
            scroll_events=[],
            started_at_ms=None,
            stopped_at_ms=None,
        )


def test_monitoring_session_rejects_invalid_label(tmp_path: Path) -> None:
    collector = MonitoringSessionCollector(sessions_file=tmp_path / "sessions.jsonl")
    with pytest.raises(ValueError):
        collector.create_record(
            user_id="u1",
            consent_given=True,
            behavior_label="invalid",
            typed_text="",
            typing_events=[],
            scroll_events=[],
            started_at_ms=None,
            stopped_at_ms=None,
        )


def test_feature_extractor_handles_empty_events() -> None:
    features = FeatureExtractor().extract_from_session(
        {
            "typed_text": "",
            "typing_events": [],
            "scroll_events": [],
        }
    )
    assert features["typed_text_length"] == 0.0
    assert features["total_key_events"] == 0.0
    assert features["total_scroll_events"] == 0.0
    assert features["total_duration_seconds"] >= 0.0


def test_feature_extractor_typing_features() -> None:
    session = {
        "typed_text": "abc",
        "typing_events": [
            {"type": "keydown", "key": "a", "code": "KeyA", "timestamp_ms": 1000, "text_length": 0},
            {"type": "keydown", "key": "Backspace", "code": "Backspace", "timestamp_ms": 1500, "text_length": 1},
            {"type": "keydown", "key": "b", "code": "KeyB", "timestamp_ms": 2000, "text_length": 1},
        ],
        "scroll_events": [],
        "started_at_ms": 1000,
        "stopped_at_ms": 3000,
    }
    features = FeatureExtractor().extract_from_session(session)
    assert features["typed_text_length"] == 3.0
    assert features["total_key_events"] == 3.0
    assert features["backspace_count"] == 1.0
    assert features["typing_duration_seconds"] == 1.0  # 2000-1000 ms
    assert features["keys_per_second"] == 3.0
    assert features["avg_key_interval_ms"] == 500.0


def test_feature_extractor_scrolling_features() -> None:
    session = {
        "typed_text": "",
        "typing_events": [],
        "scroll_events": [
            {"type": "scroll", "scroll_top": 0, "scroll_height": 2000, "client_height": 400, "timestamp_ms": 1000},
            {"type": "scroll", "scroll_top": 100, "scroll_height": 2000, "client_height": 400, "timestamp_ms": 2000},
            {"type": "scroll", "scroll_top": 50, "scroll_height": 2000, "client_height": 400, "timestamp_ms": 3000},
        ],
        "started_at_ms": 1000,
        "stopped_at_ms": 4000,
    }
    features = FeatureExtractor().extract_from_session(session)
    assert features["total_scroll_events"] == 3.0
    assert features["scroll_duration_seconds"] == 2.0
    assert features["total_scroll_distance"] == 150.0  # |100-0| + |50-100|
    assert features["avg_scroll_speed"] == 75.0
    assert features["scroll_direction_changes"] == 1.0


def test_rule_based_predictor_returns_valid_output() -> None:
    predictor = RuleBasedBehaviorPredictor()
    out = predictor.predict(
        {
            "typed_text_length": 100.0,
            "total_key_events": 60.0,
            "backspace_count": 20.0,
            "keys_per_second": 5.0,
            "avg_key_interval_ms": 180.0,
            "total_scroll_events": 0.0,
            "avg_scroll_speed": 0.0,
        }
    )
    assert isinstance(out["predicted_label"], str)
    assert 0.0 <= float(out["confidence"]) <= 1.0
    assert isinstance(out["reason"], str)


def test_behavior_predictor_falls_back_when_no_model_file(tmp_path: Path) -> None:
    predictor = BehaviorPredictor(model_file=tmp_path / "missing.joblib")
    session = {
        "typed_text": "hello",
        "typing_events": [{"timestamp_ms": 1000, "key": "h"}, {"timestamp_ms": 1100, "key": "e"}],
        "scroll_events": [],
        "started_at_ms": 1000,
        "stopped_at_ms": 2000,
    }
    result = predictor.predict_from_session(session)
    assert result["model_type"] == "rule_based_fallback"
    assert "predicted_label" in result
    assert "features" in result

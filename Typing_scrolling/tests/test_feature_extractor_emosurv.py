from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.features.feature_extractor import FeatureExtractor  # noqa: E402


def test_emosurv_features_exist_for_empty_events() -> None:
    features = FeatureExtractor().extract_from_session({"typing_events": [], "scroll_events": []})
    for name in ("D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"):
        assert name in features
        assert features[name] == 0.0


def test_emosurv_known_three_key_sequence() -> None:
    typing_events = [
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp_ms": 100, "text_length": 0},
        {"type": "keyup", "key": "a", "code": "KeyA", "timestamp_ms": 150, "text_length": 1},
        {"type": "keydown", "key": "b", "code": "KeyB", "timestamp_ms": 180, "text_length": 1},
        {"type": "keyup", "key": "b", "code": "KeyB", "timestamp_ms": 220, "text_length": 2},
        {"type": "keydown", "key": "c", "code": "KeyC", "timestamp_ms": 260, "text_length": 2},
        {"type": "keyup", "key": "c", "code": "KeyC", "timestamp_ms": 310, "text_length": 3},
    ]
    features = FeatureExtractor().extract_from_session({"typing_events": typing_events, "scroll_events": []})

    assert features["D1U1"] == pytest.approx((50.0 + 40.0 + 50.0) / 3.0, rel=1e-6)
    assert features["D1U2"] == pytest.approx(((220.0 - 100.0) + (310.0 - 180.0)) / 2.0, rel=1e-6)
    assert features["D1D2"] == pytest.approx(((180.0 - 100.0) + (260.0 - 180.0)) / 2.0, rel=1e-6)
    assert features["U1D2"] == pytest.approx(((180.0 - 150.0) + (260.0 - 220.0)) / 2.0, rel=1e-6)
    assert features["U1U2"] == pytest.approx(((220.0 - 150.0) + (310.0 - 220.0)) / 2.0, rel=1e-6)
    assert features["D1U3"] == pytest.approx(310.0 - 100.0, rel=1e-6)
    assert features["D1D3"] == pytest.approx(260.0 - 100.0, rel=1e-6)


def test_emosurv_keydown_only_sequence() -> None:
    typing_events = [
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp_ms": 100, "text_length": 0},
        {"type": "keydown", "key": "b", "code": "KeyB", "timestamp_ms": 180, "text_length": 1},
        {"type": "keydown", "key": "c", "code": "KeyC", "timestamp_ms": 260, "text_length": 2},
    ]
    features = FeatureExtractor().extract_from_session({"typing_events": typing_events, "scroll_events": []})

    assert features["D1D2"] == pytest.approx(80.0, rel=1e-6)
    assert features["D1D3"] == pytest.approx(160.0, rel=1e-6)
    for name in ("D1U1", "D1U2", "U1D2", "U1U2", "D1U3"):
        assert features[name] == 0.0


def test_emosurv_missing_keyup_and_repeats_do_not_crash() -> None:
    typing_events = [
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp_ms": 100, "text_length": 0},
        {"type": "keydown", "key": "a", "code": "KeyA", "timestamp_ms": 120, "text_length": 1},  # repeat
        {"type": "keyup", "key": "a", "code": "KeyA", "timestamp_ms": 150, "text_length": 1},
        {"type": "keyup", "key": "a", "code": "KeyA", "timestamp_ms": 160, "text_length": 1},  # extra up
        {"type": "keydown", "key": "b", "code": "KeyB", "timestamp_ms": 180, "text_length": 1},
        # missing B keyup
        {"type": "keydown", "key": "c", "code": "KeyC", "timestamp_ms": 260, "text_length": 2},
        {"type": "keyup", "key": "c", "code": "KeyC", "timestamp_ms": 310, "text_length": 3},
    ]
    features = FeatureExtractor().extract_from_session({"typing_events": typing_events, "scroll_events": []})
    for name in ("D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"):
        assert isinstance(features[name], float)

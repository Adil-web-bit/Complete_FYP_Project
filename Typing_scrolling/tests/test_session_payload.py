from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.utils.session_payload import (  # noqa: E402
    calculate_payload_duration_seconds,
    get_payload_event_counts,
    is_stopped_payload,
    normalize_component_payload,
    payload_has_events,
)


def test_helpers_handle_none_payload() -> None:
    assert normalize_component_payload(None) is None
    assert is_stopped_payload(None) is False
    assert get_payload_event_counts(None) == {"typing": 0, "scroll": 0}
    assert payload_has_events(None) is False
    assert calculate_payload_duration_seconds(None) is None


def test_stopped_payload_detection_and_counts() -> None:
    payload = {
        "typed_text": "hi",
        "typing_events": [{"a": 1}],
        "scroll_events": [],
        "started_at_ms": 1000,
        "stopped_at_ms": 3000,
        "monitoring_status": "stopped",
        "component_version": "0.2.0",
        "event_counts": {"typing": 1, "scroll": 0},
    }
    assert is_stopped_payload(payload) is True
    assert get_payload_event_counts(payload) == {"typing": 1, "scroll": 0}
    assert payload_has_events(payload) is True
    assert calculate_payload_duration_seconds(payload) == 2.0


def test_compatibility_with_older_payload_without_event_counts() -> None:
    older = {
        "typed_text": "abc",
        "typing_events": [{}, {}],
        "scroll_events": [{}],
        "started_at_ms": 10,
        "stopped_at_ms": 20,
        "monitoring_status": "stopped",
    }
    normalized = normalize_component_payload(older)
    assert normalized is not None
    assert normalized["event_counts"] == {"typing": 2, "scroll": 1}
    assert payload_has_events(older) is True


def test_duration_none_when_missing_timestamps() -> None:
    payload = {"monitoring_status": "stopped", "typing_events": [], "scroll_events": []}
    assert calculate_payload_duration_seconds(payload) is None


from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_path() -> None:
    """
    Ensure `human_behavior_predictor/` is on sys.path so tests can import `src.*`
    even when pytest is executed from the repository root.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.data_collection.scrolling_collector import ScrollingCollector  # noqa: E402
from src.data_collection.typing_collector import TypingCollector  # noqa: E402
from src.utils.validators import (  # noqa: E402
    validate_non_empty_text,
    validate_non_negative_integer,
    validate_positive_number,
)


def test_validate_non_empty_text() -> None:
    assert validate_non_empty_text("hello") is True
    assert validate_non_empty_text("  hello  ") is True
    assert validate_non_empty_text("") is False
    assert validate_non_empty_text("   ") is False


def test_validate_positive_number() -> None:
    assert validate_positive_number(1.0) is True
    assert validate_positive_number(0.0001) is True
    assert validate_positive_number(0.0) is False
    assert validate_positive_number(-1.0) is False
    assert validate_positive_number("bad") is False  # type: ignore[arg-type]


def test_validate_non_negative_integer() -> None:
    assert validate_non_negative_integer(0) is True
    assert validate_non_negative_integer(10) is True
    assert validate_non_negative_integer(-1) is False
    assert validate_non_negative_integer(True) is False  # bool is not accepted


def test_typing_collector_create_record_calculations() -> None:
    collector = TypingCollector(session_id="test-session")
    record = collector.create_record(
        user_id="u1",
        text="abc",
        duration_seconds=10.0,
        backspace_count=2,
    )
    assert record["text_length"] == 3
    assert record["key_count"] == 5
    assert record["backspace_count"] == 2
    assert record["average_key_interval_ms"] == 2000.0
    assert record["session_id"] == "test-session"
    assert record["user_id"] == "u1"
    assert record["source"] == "streamlit"

    empty = collector.create_record(
        user_id="u1",
        text="",
        duration_seconds=0.0,
        backspace_count=0,
    )
    assert empty["key_count"] == 0
    assert empty["average_key_interval_ms"] is None


def test_scrolling_collector_create_record_calculations() -> None:
    collector = ScrollingCollector(session_id="test-session")
    record = collector.create_record(
        user_id="u1",
        page_name="Home",
        total_scroll_distance=100.0,
        scroll_duration_seconds=20.0,
        scroll_events_count=4,
    )
    assert record["average_scroll_speed"] == 5.0
    assert record["scroll_events_count"] == 4
    assert record["session_id"] == "test-session"
    assert record["user_id"] == "u1"
    assert record["source"] == "streamlit"

    zero_duration = collector.create_record(
        user_id="u1",
        page_name="Home",
        total_scroll_distance=100.0,
        scroll_duration_seconds=0.0,
        scroll_events_count=4,
    )
    assert zero_duration["average_scroll_speed"] is None

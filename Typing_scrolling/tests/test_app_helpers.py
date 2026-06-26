from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.utils.app_helpers import (  # noqa: E402
    safe_percent,
    session_is_usable_for_training,
    summarize_sessions,
)


def test_safe_percent() -> None:
    assert safe_percent(1, 2) == 50.0
    assert safe_percent(0, 0) == 0.0
    assert safe_percent(1, 0) == 0.0


def test_session_is_usable_for_training() -> None:
    assert (
        session_is_usable_for_training(
            {
                "behavior_label": "focused",
                "typing_events": [{"k": 1}],
                "scroll_events": [],
            }
        )
        is True
    )
    assert (
        session_is_usable_for_training(
            {
                "behavior_label": "focused",
                "typing_events": [],
                "scroll_events": [],
            }
        )
        is False
    )
    assert (
        session_is_usable_for_training(
            {
                "behavior_label": "invalid",
                "typing_events": [{"k": 1}],
                "scroll_events": [],
            }
        )
        is False
    )


def test_summarize_sessions() -> None:
    sessions = [
        {"session_id": "s1", "user_id": "u1", "behavior_label": "normal", "typing_events": [{}], "scroll_events": []},
        {"session_id": "s2", "user_id": "u2", "behavior_label": "focused", "typing_events": [], "scroll_events": []},
        {"session_id": "s3", "user_id": "u1", "behavior_label": None, "typing_events": [{}], "scroll_events": []},
    ]
    summary = summarize_sessions(sessions)
    assert summary["total_sessions"] == 3
    assert summary["labeled_sessions"] == 2
    assert summary["usable_for_training"] == 1
    assert summary["unique_users"] == 2
    assert summary["label_counts"]["normal"] == 1


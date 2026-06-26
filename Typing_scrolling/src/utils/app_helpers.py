from __future__ import annotations

from typing import Any

# Keep this module dependency-light to avoid import cycles during Streamlit startup.
ALLOWED_BEHAVIOR_LABELS: tuple[str, ...] = (
    "normal",
    "focused",
    "stressed",
    "rushed",
    "distracted",
)


def format_count(value: int) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def safe_percent(numerator: int, denominator: int) -> float:
    try:
        n = int(numerator)
        d = int(denominator)
    except Exception:
        return 0.0
    if d <= 0:
        return 0.0
    return max(0.0, min(100.0, (n / d) * 100.0))


def session_is_usable_for_training(session: dict[str, Any]) -> bool:
    label = session.get("behavior_label")
    if not isinstance(label, str) or label not in ALLOWED_BEHAVIOR_LABELS:
        return False
    typing_events = session.get("typing_events", [])
    scroll_events = session.get("scroll_events", [])
    typing_count = len(typing_events) if isinstance(typing_events, list) else 0
    scroll_count = len(scroll_events) if isinstance(scroll_events, list) else 0
    return (typing_count + scroll_count) > 0


def summarize_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    total_sessions = len([s for s in sessions if isinstance(s, dict)])

    label_counts: dict[str, int] = {}
    labeled_sessions = 0
    usable_for_training = 0
    unique_users: set[str] = set()

    for s in sessions:
        if not isinstance(s, dict):
            continue
        user_id = s.get("user_id")
        if isinstance(user_id, str) and user_id.strip():
            unique_users.add(user_id.strip())

        label = s.get("behavior_label")
        if isinstance(label, str) and label in ALLOWED_BEHAVIOR_LABELS:
            labeled_sessions += 1
            label_counts[label] = label_counts.get(label, 0) + 1
        if session_is_usable_for_training(s):
            usable_for_training += 1

    return {
        "total_sessions": int(total_sessions),
        "labeled_sessions": int(labeled_sessions),
        "usable_for_training": int(usable_for_training),
        "unique_users": int(len(unique_users)),
        "label_counts": label_counts,
    }

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.config import SESSIONS_FILE
from src.utils.validators import validate_non_empty_text


ALLOWED_BEHAVIOR_LABELS: tuple[str, ...] = (
    "normal",
    "focused",
    "stressed",
    "rushed",
    "distracted",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MonitoringSessionCollector:
    """
    Save full monitoring sessions captured by a Streamlit JS component to JSONL.
    """

    sessions_file: Path = SESSIONS_FILE

    def create_record(
        self,
        user_id: str,
        consent_given: bool,
        behavior_label: Optional[str],
        typed_text: str,
        typing_events: list[dict[str, Any]],
        scroll_events: list[dict[str, Any]],
        started_at_ms: Optional[int],
        stopped_at_ms: Optional[int],
    ) -> dict[str, Any]:
        if not validate_non_empty_text(user_id):
            raise ValueError("user_id must be non-empty.")
        if consent_given is not True:
            raise ValueError("consent_given must be True.")
        if behavior_label is not None and behavior_label not in ALLOWED_BEHAVIOR_LABELS:
            raise ValueError(f"behavior_label must be one of {ALLOWED_BEHAVIOR_LABELS} or None.")
        if not isinstance(typing_events, list):
            raise ValueError("typing_events must be a list.")
        if not isinstance(scroll_events, list):
            raise ValueError("scroll_events must be a list.")

        duration_seconds: Optional[float] = None
        if started_at_ms is not None and stopped_at_ms is not None:
            if not isinstance(started_at_ms, int) or not isinstance(stopped_at_ms, int):
                raise ValueError("started_at_ms and stopped_at_ms must be ints or None.")
            if stopped_at_ms < started_at_ms:
                raise ValueError("stopped_at_ms must be >= started_at_ms.")
            duration_seconds = (stopped_at_ms - started_at_ms) / 1000.0

        return {
            "session_id": uuid4().hex,
            "user_id": user_id.strip(),
            "timestamp_utc": _utc_now_iso(),
            "consent_given": True,
            "behavior_label": behavior_label,
            "typed_text": typed_text,
            "typing_events": typing_events,
            "scroll_events": scroll_events,
            "started_at_ms": started_at_ms,
            "stopped_at_ms": stopped_at_ms,
            "duration_seconds": duration_seconds,
            "source": "streamlit_js_component",
        }

    def save_record(self, record: dict[str, Any]) -> None:
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self.sessions_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def load_records(self) -> list[dict[str, Any]]:
        if not self.sessions_file.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.sessions_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    records.append(parsed)
        return records


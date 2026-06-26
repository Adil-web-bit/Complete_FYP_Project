from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.config import SCROLLING_EVENTS_FILE


@dataclass(slots=True)
class ScrollingCollector:
    """
    Collect (estimated) scrolling behavior and store it as JSON Lines.

    Note: Streamlit cannot easily capture real scroll events without custom JS,
    so this collector estimates scroll-derived features from user-provided values.
    """

    session_id: str = ""
    events_file: Path = SCROLLING_EVENTS_FILE

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = uuid4().hex

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_record(
        self,
        user_id: str,
        page_name: str,
        total_scroll_distance: float,
        scroll_duration_seconds: float,
        scroll_events_count: int,
    ) -> dict[str, Any]:
        average_scroll_speed: Optional[float]
        if scroll_duration_seconds > 0:
            average_scroll_speed = float(total_scroll_distance) / float(scroll_duration_seconds)
        else:
            average_scroll_speed = None

        return {
            "session_id": self.session_id,
            "user_id": user_id,
            "timestamp_utc": self._utc_now_iso(),
            "page_name": page_name,
            "total_scroll_distance": float(total_scroll_distance),
            "scroll_duration_seconds": float(scroll_duration_seconds),
            "scroll_events_count": int(scroll_events_count),
            "average_scroll_speed": average_scroll_speed,
            "source": "streamlit",
        }

    def save_record(self, record: dict[str, Any]) -> None:
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self.events_file.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def load_records(self) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.events_file.open("r", encoding="utf-8") as file:
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


# Backwards-compatible alias (older scaffold name)
ScrollingDataCollector = ScrollingCollector

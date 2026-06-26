from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.config import TYPING_EVENTS_FILE


@dataclass(slots=True)
class TypingCollector:
    """
    Collect (estimated) typing behavior and store it as JSON Lines.

    Note: Streamlit cannot easily capture raw keydown/keyup events without custom JS,
    so this collector estimates typing-derived features from user-provided values.
    """

    session_id: str = ""
    events_file: Path = TYPING_EVENTS_FILE

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = uuid4().hex

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_record(
        self,
        user_id: str,
        text: str,
        duration_seconds: float,
        backspace_count: int,
    ) -> dict[str, Any]:
        text_length = len(text)
        key_count = text_length + backspace_count
        average_key_interval_ms: Optional[float]
        if key_count > 0:
            average_key_interval_ms = (duration_seconds * 1000.0) / float(key_count)
        else:
            average_key_interval_ms = None

        return {
            "session_id": self.session_id,
            "user_id": user_id,
            "timestamp_utc": self._utc_now_iso(),
            "text": text,
            "text_length": text_length,
            "duration_seconds": float(duration_seconds),
            "key_count": int(key_count),
            "backspace_count": int(backspace_count),
            "average_key_interval_ms": average_key_interval_ms,
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
TypingDataCollector = TypingCollector

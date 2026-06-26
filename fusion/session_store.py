from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SESSION_DIR = Path(__file__).resolve().parents[1] / "data" / "multimodal_sessions"


def save_multimodal_session(prediction: dict[str, Any], session_dir: Path = SESSION_DIR) -> Path:
    session_id = str(prediction.get("session_id") or "unknown_session")
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{session_id}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(prediction, file, ensure_ascii=False, indent=2)
    return path


def load_multimodal_history(session_dir: Path = SESSION_DIR, limit: int = 20) -> list[dict[str, Any]]:
    if not session_dir.exists():
        return []
    files = sorted(
        [p for p in session_dir.glob("*.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    history: list[dict[str, Any]] = []
    for path in files[: max(0, int(limit))]:
        try:
            with path.open("r", encoding="utf-8") as file:
                parsed = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            history.append(parsed)
    return history

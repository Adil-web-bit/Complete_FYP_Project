from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ALLOWED_BEHAVIOR_LABELS: tuple[str, ...] = (
    "normal",
    "focused",
    "stressed",
    "rushed",
    "distracted",
)


def load_sessions_from_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    sessions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                sessions.append(parsed)
    return sessions


def save_sessions_to_jsonl(sessions: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for session in sessions:
            file.write(json.dumps(session, ensure_ascii=False) + "\n")


def anonymize_session(session: dict[str, Any]) -> dict[str, Any]:
    """
    Return an anonymized copy of a session.

    - user_id becomes "anonymous"
    - typed_text is removed/blanked
    - session_id is preserved but prefixed with "anon_" if not already
    - adds {"anonymized": True}
    """
    copy: dict[str, Any] = dict(session)
    copy["user_id"] = "anonymous"
    copy["typed_text"] = ""
    session_id = str(copy.get("session_id", ""))
    if session_id and not session_id.startswith("anon_"):
        copy["session_id"] = f"anon_{session_id}"
    copy["anonymized"] = True
    return copy


def export_sessions(
    sessions: list[dict[str, Any]],
    output_path: Path,
    anonymize: bool = True,
    labeled_only: bool = False,
    usable_for_training_only: bool = False,
) -> dict[str, Any]:
    exported = sessions
    if labeled_only:
        exported = [
            s
            for s in exported
            if isinstance(s, dict)
            and isinstance(s.get("behavior_label"), str)
            and s.get("behavior_label") in ALLOWED_BEHAVIOR_LABELS
        ]
    if usable_for_training_only:
        exported = [
            s
            for s in exported
            if isinstance(s, dict)
            and isinstance(s.get("behavior_label"), str)
            and s.get("behavior_label") in ALLOWED_BEHAVIOR_LABELS
            and (
                (len(s.get("typing_events")) if isinstance(s.get("typing_events"), list) else 0)
                + (len(s.get("scroll_events")) if isinstance(s.get("scroll_events"), list) else 0)
            )
            > 0
        ]
    if anonymize:
        exported = [anonymize_session(s) for s in exported if isinstance(s, dict)]

    save_sessions_to_jsonl(exported, output_path)
    return {
        "status": "exported",
        "output_path": str(output_path),
        "session_count": int(len(exported)),
        "anonymized": bool(anonymize),
        "labeled_only": bool(labeled_only),
        "usable_for_training_only": bool(usable_for_training_only),
    }


def import_sessions(
    input_path: Path,
    existing_sessions: list[dict[str, Any]],
    deduplicate: bool = True,
) -> dict[str, Any]:
    incoming = load_sessions_from_jsonl(input_path)
    existing_ids: set[str] = set()
    for s in existing_sessions:
        if not isinstance(s, dict):
            continue
        sid = s.get("session_id")
        if isinstance(sid, str) and sid:
            existing_ids.add(sid)

    imported: list[dict[str, Any]] = []
    skipped_duplicates = 0
    for session in incoming:
        if not isinstance(session, dict):
            continue
        sid = session.get("session_id")
        sid_str = sid if isinstance(sid, str) else ""
        if deduplicate and sid_str and sid_str in existing_ids:
            skipped_duplicates += 1
            continue
        imported.append(session)
        if sid_str:
            existing_ids.add(sid_str)

    return {
        "status": "imported",
        "imported_count": int(len(imported)),
        "skipped_duplicates": int(skipped_duplicates),
        "sessions": imported,
    }

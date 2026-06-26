from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()

from src.utils.session_io import (  # noqa: E402
    anonymize_session,
    export_sessions,
    import_sessions,
    load_sessions_from_jsonl,
)


def test_anonymize_session_removes_typed_text_and_user() -> None:
    session = {
        "session_id": "abc",
        "user_id": "u1",
        "typed_text": "secret",
        "typing_events": [],
        "scroll_events": [],
        "behavior_label": "focused",
    }
    anon = anonymize_session(session)
    assert anon["user_id"] == "anonymous"
    assert anon["typed_text"] == ""
    assert str(anon["session_id"]).startswith("anon_")
    assert anon["anonymized"] is True


def test_export_sessions_writes_anonymized_jsonl(tmp_path: Path) -> None:
    sessions = [
        {"session_id": "s1", "user_id": "u1", "typed_text": "t", "behavior_label": "normal"},
        {"session_id": "s2", "user_id": "u2", "typed_text": "t2", "behavior_label": None},
    ]
    out = tmp_path / "export.jsonl"
    report = export_sessions(sessions, out, anonymize=True, labeled_only=False)
    assert report["status"] == "exported"
    assert out.exists()
    exported = load_sessions_from_jsonl(out)
    assert len(exported) == 2
    assert exported[0]["user_id"] == "anonymous"
    assert exported[0]["typed_text"] == ""


def test_export_sessions_labeled_only_filters(tmp_path: Path) -> None:
    sessions = [
        {"session_id": "s1", "behavior_label": "normal"},
        {"session_id": "s2", "behavior_label": None},
        {"session_id": "s3", "behavior_label": "invalid"},
    ]
    out = tmp_path / "labeled.jsonl"
    report = export_sessions(sessions, out, anonymize=False, labeled_only=True)
    assert report["session_count"] == 1
    exported = load_sessions_from_jsonl(out)
    assert len(exported) == 1
    assert exported[0]["session_id"] == "s1"


def test_import_sessions_skips_duplicates(tmp_path: Path) -> None:
    incoming_path = tmp_path / "incoming.jsonl"
    incoming_path.write_text(
        "\n".join(
            [
                '{"session_id":"s1","user_id":"u"}',
                '{"session_id":"s2","user_id":"u"}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    existing = [{"session_id": "s1"}]
    report = import_sessions(incoming_path, existing_sessions=existing, deduplicate=True)
    assert report["status"] == "imported"
    assert report["imported_count"] == 1
    assert report["skipped_duplicates"] == 1
    assert isinstance(report["sessions"], list)
    assert report["sessions"][0]["session_id"] == "s2"


def test_export_sessions_usable_for_training_only_filters(tmp_path: Path) -> None:
    sessions = [
        {"session_id": "s1", "behavior_label": "normal", "typing_events": [{}], "scroll_events": []},
        {"session_id": "s2", "behavior_label": "normal", "typing_events": [], "scroll_events": []},
        {"session_id": "s3", "behavior_label": None, "typing_events": [{}], "scroll_events": []},
    ]
    out = tmp_path / "usable.jsonl"
    report = export_sessions(sessions, out, anonymize=False, labeled_only=False, usable_for_training_only=True)
    assert report["session_count"] == 1
    exported = load_sessions_from_jsonl(out)
    assert len(exported) == 1
    assert exported[0]["session_id"] == "s1"

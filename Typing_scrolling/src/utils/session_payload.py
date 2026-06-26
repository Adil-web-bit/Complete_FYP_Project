from __future__ import annotations

from typing import Any, Optional


def normalize_component_payload(payload: dict | None) -> Optional[dict[str, Any]]:
    """
    Normalize a monitoring component payload to a predictable dict shape.

    Supports older payloads that may not include `event_counts` or `component_version`.
    """
    if payload is None or not isinstance(payload, dict):
        return None

    typed_text = payload.get("typed_text", "")
    typing_events = payload.get("typing_events", [])
    scroll_events = payload.get("scroll_events", [])

    if not isinstance(typed_text, str):
        typed_text = str(typed_text)
    if not isinstance(typing_events, list):
        typing_events = []
    if not isinstance(scroll_events, list):
        scroll_events = []

    started_at_ms = payload.get("started_at_ms")
    stopped_at_ms = payload.get("stopped_at_ms")
    started_at_ms = started_at_ms if isinstance(started_at_ms, int) else None
    stopped_at_ms = stopped_at_ms if isinstance(stopped_at_ms, int) else None

    monitoring_status = payload.get("monitoring_status")
    if monitoring_status not in ("idle", "monitoring", "stopped"):
        monitoring_status = "idle"

    event_counts = payload.get("event_counts")
    if not isinstance(event_counts, dict):
        event_counts = {
            "typing": len(typing_events),
            "scroll": len(scroll_events),
        }
    else:
        event_counts = {
            "typing": int(event_counts.get("typing", len(typing_events))) if isinstance(event_counts.get("typing", 0), (int, float)) else len(typing_events),
            "scroll": int(event_counts.get("scroll", len(scroll_events))) if isinstance(event_counts.get("scroll", 0), (int, float)) else len(scroll_events),
        }

    component_version = payload.get("component_version")
    if not isinstance(component_version, str):
        component_version = "unknown"

    return {
        "typed_text": typed_text,
        "typing_events": typing_events,
        "scroll_events": scroll_events,
        "started_at_ms": started_at_ms,
        "stopped_at_ms": stopped_at_ms,
        "monitoring_status": monitoring_status,
        "component_version": component_version,
        "event_counts": event_counts,
    }


def is_stopped_payload(payload: dict | None) -> bool:
    normalized = normalize_component_payload(payload)
    return bool(normalized and normalized.get("monitoring_status") == "stopped")


def get_payload_event_counts(payload: dict | None) -> dict[str, int]:
    normalized = normalize_component_payload(payload)
    if not normalized:
        return {"typing": 0, "scroll": 0}
    event_counts = normalized.get("event_counts", {})
    if isinstance(event_counts, dict):
        return {
            "typing": int(event_counts.get("typing", 0)) if isinstance(event_counts.get("typing", 0), (int, float)) else 0,
            "scroll": int(event_counts.get("scroll", 0)) if isinstance(event_counts.get("scroll", 0), (int, float)) else 0,
        }
    return {"typing": 0, "scroll": 0}


def payload_has_events(payload: dict | None) -> bool:
    counts = get_payload_event_counts(payload)
    return (counts.get("typing", 0) + counts.get("scroll", 0)) > 0


def calculate_payload_duration_seconds(payload: dict | None) -> Optional[float]:
    normalized = normalize_component_payload(payload)
    if not normalized:
        return None
    started_at_ms = normalized.get("started_at_ms")
    stopped_at_ms = normalized.get("stopped_at_ms")
    if not isinstance(started_at_ms, int) or not isinstance(stopped_at_ms, int):
        return None
    if stopped_at_ms < started_at_ms:
        return None
    return (stopped_at_ms - started_at_ms) / 1000.0


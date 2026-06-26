from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sorted_timestamps_ms(events: list[dict[str, Any]]) -> list[int]:
    timestamps: list[int] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        ts = event.get("timestamp_ms")
        if isinstance(ts, int):
            timestamps.append(ts)
    timestamps.sort()
    return timestamps


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    try:
        return float(np.mean(np.array(values, dtype=float)))
    except Exception:
        return 0.0


def _build_key_press_sequence(typing_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a key-press sequence from raw typing events.

    Each press is represented as:
      {"key": str, "code": str, "down_ms": int, "up_ms": int | None}

    Pairing strategy (safe, best-effort):
    - On keydown: append a new press (up_ms=None)
    - On keyup: match earliest unmatched press by code, else by key, else earliest unmatched
    """
    events: list[dict[str, Any]] = []
    for e in typing_events:
        if isinstance(e, dict):
            events.append(e)
    events.sort(key=lambda x: x.get("timestamp_ms", 0) if isinstance(x.get("timestamp_ms"), int) else 0)

    presses: list[dict[str, Any]] = []

    def _find_unmatched(match_fn) -> int | None:
        for idx, p in enumerate(presses):
            if p.get("up_ms") is not None:
                continue
            if match_fn(p):
                return idx
        return None

    for e in events:
        etype = e.get("type")
        ts = e.get("timestamp_ms")
        if not isinstance(ts, int):
            continue
        key = e.get("key")
        code = e.get("code")
        key_str = str(key) if key is not None else ""
        code_str = str(code) if code is not None else ""

        if etype == "keydown":
            presses.append({"key": key_str, "code": code_str, "down_ms": ts, "up_ms": None})
            continue

        if etype == "keyup":
            # Match earliest unmatched press with same code
            idx = None
            if code_str:
                idx = _find_unmatched(lambda p: str(p.get("code", "")) == code_str)
            if idx is None and key_str:
                idx = _find_unmatched(lambda p: str(p.get("key", "")) == key_str)
            if idx is None:
                idx = _find_unmatched(lambda _p: True)
            if idx is None:
                continue
            down_ms = presses[idx].get("down_ms")
            if isinstance(down_ms, int) and ts >= down_ms:
                presses[idx]["up_ms"] = ts
            continue

    return presses


def _extract_emosurv_timing_features(typing_events: list[dict[str, Any]]) -> dict[str, float]:
    """
    Extract EmoSurv-style aggregate timing features from a session's typing events.

    Features are means across all consecutive windows:
      - D1U1: U1 - D1
      - D1U2: U2 - D1
      - D1D2: D2 - D1
      - U1D2: D2 - U1
      - U1U2: U2 - U1
      - D1U3: U3 - D1
      - D1D3: D3 - D1

    Backward compatibility:
      - If only keydown exists, D1D2 and D1D3 may still be computed.
      - Features requiring keyup return 0.0 if keyup is unavailable.
    """
    presses = _build_key_press_sequence(typing_events)
    downs: list[int] = []
    ups: list[int | None] = []
    for p in presses:
        d = p.get("down_ms")
        if isinstance(d, int):
            downs.append(d)
            u = p.get("up_ms")
            ups.append(u if isinstance(u, int) else None)

    d1u1_vals: list[float] = []
    for d, u in zip(downs, ups):
        if u is None:
            continue
        delta = float(u - d)
        if delta >= 0:
            d1u1_vals.append(delta)

    d1u2_vals: list[float] = []
    d1d2_vals: list[float] = []
    u1d2_vals: list[float] = []
    u1u2_vals: list[float] = []
    for i in range(len(downs) - 1):
        d1 = downs[i]
        d2 = downs[i + 1]
        u1 = ups[i]
        u2 = ups[i + 1]

        delta_d1d2 = float(d2 - d1)
        if delta_d1d2 >= 0:
            d1d2_vals.append(delta_d1d2)

        if u2 is not None:
            delta_d1u2 = float(u2 - d1)
            if delta_d1u2 >= 0:
                d1u2_vals.append(delta_d1u2)

        if u1 is not None:
            delta_u1d2 = float(d2 - u1)
            if delta_u1d2 >= 0:
                u1d2_vals.append(delta_u1d2)

        if u1 is not None and u2 is not None:
            delta_u1u2 = float(u2 - u1)
            if delta_u1u2 >= 0:
                u1u2_vals.append(delta_u1u2)

    d1u3_vals: list[float] = []
    d1d3_vals: list[float] = []
    for i in range(len(downs) - 2):
        d1 = downs[i]
        d3 = downs[i + 2]
        u3 = ups[i + 2]

        delta_d1d3 = float(d3 - d1)
        if delta_d1d3 >= 0:
            d1d3_vals.append(delta_d1d3)

        if u3 is not None:
            delta_d1u3 = float(u3 - d1)
            if delta_d1u3 >= 0:
                d1u3_vals.append(delta_d1u3)

    return {
        "D1U1": _safe_mean(d1u1_vals),
        "D1U2": _safe_mean(d1u2_vals),
        "D1D2": _safe_mean(d1d2_vals),
        "U1D2": _safe_mean(u1d2_vals),
        "U1U2": _safe_mean(u1u2_vals),
        "D1U3": _safe_mean(d1u3_vals),
        "D1D3": _safe_mean(d1d3_vals),
    }


@dataclass(slots=True)
class FeatureExtractor:
    """
    Extract numeric features from a monitoring session record.
    """

    def extract_from_session(self, session: dict[str, Any]) -> dict[str, float]:
        typing_events_raw = session.get("typing_events", [])
        scroll_events_raw = session.get("scroll_events", [])
        typed_text = session.get("typed_text", "")

        typing_events: list[dict[str, Any]] = (
            typing_events_raw if isinstance(typing_events_raw, list) else []
        )
        scroll_events: list[dict[str, Any]] = (
            scroll_events_raw if isinstance(scroll_events_raw, list) else []
        )

        typed_text_length = float(len(typed_text)) if isinstance(typed_text, str) else 0.0

        # Typing features
        total_key_events = float(len(typing_events))
        backspace_count = float(
            sum(1 for e in typing_events if isinstance(e, dict) and e.get("key") == "Backspace")
        )
        unique_keys_count = float(
            len(
                {
                    str(e.get("key"))
                    for e in typing_events
                    if isinstance(e, dict) and e.get("key") is not None
                }
            )
        )

        typing_timestamps = _sorted_timestamps_ms(typing_events)
        if len(typing_timestamps) >= 2:
            typing_duration_seconds = (typing_timestamps[-1] - typing_timestamps[0]) / 1000.0
            typing_duration_seconds = max(0.0, typing_duration_seconds)
            deltas_ms = np.diff(np.array(typing_timestamps, dtype=np.int64)).astype(float)
            deltas_ms = deltas_ms[deltas_ms >= 0]
            avg_key_interval_ms = float(np.mean(deltas_ms)) if deltas_ms.size else 0.0
            std_key_interval_ms = float(np.std(deltas_ms)) if deltas_ms.size else 0.0
        else:
            typing_duration_seconds = 0.0
            avg_key_interval_ms = 0.0
            std_key_interval_ms = 0.0

        keys_per_second = (
            float(total_key_events) / float(typing_duration_seconds)
            if typing_duration_seconds > 0
            else 0.0
        )

        # Scrolling features
        total_scroll_events = float(len(scroll_events))
        scroll_timestamps = _sorted_timestamps_ms(scroll_events)
        if len(scroll_timestamps) >= 2:
            scroll_duration_seconds = (scroll_timestamps[-1] - scroll_timestamps[0]) / 1000.0
            scroll_duration_seconds = max(0.0, scroll_duration_seconds)
        else:
            scroll_duration_seconds = 0.0

        scroll_tops: list[float] = []
        scroll_times: list[int] = []
        for event in scroll_events:
            if not isinstance(event, dict):
                continue
            ts = event.get("timestamp_ms")
            top = event.get("scroll_top")
            if isinstance(ts, int) and top is not None:
                scroll_times.append(ts)
                scroll_tops.append(_safe_float(top, default=0.0))

        total_scroll_distance = 0.0
        max_scroll_speed = 0.0
        direction_changes = 0.0
        if len(scroll_times) >= 2:
            pairs = sorted(zip(scroll_times, scroll_tops), key=lambda x: x[0])
            times_sorted = [p[0] for p in pairs]
            tops_sorted = [p[1] for p in pairs]
            deltas_top = np.diff(np.array(tops_sorted, dtype=float))
            deltas_time_s = np.diff(np.array(times_sorted, dtype=np.int64)).astype(float) / 1000.0

            total_scroll_distance = float(np.sum(np.abs(deltas_top))) if deltas_top.size else 0.0

            speeds: list[float] = []
            for dt_s, dtop in zip(deltas_time_s, deltas_top):
                if dt_s > 0:
                    speeds.append(abs(float(dtop)) / float(dt_s))
            max_scroll_speed = max(speeds) if speeds else 0.0

            signs = np.sign(deltas_top)
            prev_sign = 0.0
            for sign in signs:
                if sign == 0:
                    continue
                if prev_sign != 0 and sign != prev_sign:
                    direction_changes += 1.0
                prev_sign = float(sign)

        avg_scroll_speed = (
            float(total_scroll_distance) / float(scroll_duration_seconds)
            if scroll_duration_seconds > 0
            else 0.0
        )

        # Session-level duration (prefer recorded duration if valid)
        total_duration_seconds = _safe_float(session.get("duration_seconds"), default=0.0)
        if total_duration_seconds <= 0.0:
            started_at_ms = session.get("started_at_ms")
            stopped_at_ms = session.get("stopped_at_ms")
            if isinstance(started_at_ms, int) and isinstance(stopped_at_ms, int) and stopped_at_ms >= started_at_ms:
                total_duration_seconds = (stopped_at_ms - started_at_ms) / 1000.0
            else:
                total_duration_seconds = max(typing_duration_seconds, scroll_duration_seconds, 0.0)

        emosurv = _extract_emosurv_timing_features(typing_events)

        return {
            "typed_text_length": typed_text_length,
            "total_key_events": float(total_key_events),
            "backspace_count": float(backspace_count),
            "unique_keys_count": float(unique_keys_count),
            "typing_duration_seconds": float(typing_duration_seconds),
            "keys_per_second": float(keys_per_second),
            "avg_key_interval_ms": float(avg_key_interval_ms),
            "std_key_interval_ms": float(std_key_interval_ms),
            "total_scroll_events": float(total_scroll_events),
            "scroll_duration_seconds": float(scroll_duration_seconds),
            "total_scroll_distance": float(total_scroll_distance),
            "avg_scroll_speed": float(avg_scroll_speed),
            "max_scroll_speed": float(max_scroll_speed),
            "scroll_direction_changes": float(direction_changes),
            "total_duration_seconds": float(total_duration_seconds),
            "D1U1": float(emosurv.get("D1U1", 0.0)),
            "D1U2": float(emosurv.get("D1U2", 0.0)),
            "D1D2": float(emosurv.get("D1D2", 0.0)),
            "U1D2": float(emosurv.get("U1D2", 0.0)),
            "U1U2": float(emosurv.get("U1U2", 0.0)),
            "D1U3": float(emosurv.get("D1U3", 0.0)),
            "D1D3": float(emosurv.get("D1D3", 0.0)),
        }

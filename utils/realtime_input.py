from __future__ import annotations

from typing import Any

from PIL import Image


def camera_capture_to_rgb_image(capture: Any) -> Image.Image | None:
    """Convert Streamlit camera_input output into an RGB PIL image."""
    if capture is None:
        return None
    if hasattr(capture, "seek"):
        capture.seek(0)
    return Image.open(capture).convert("RGB")


def prepare_audio_capture(audio_capture: Any) -> Any | None:
    """Reset Streamlit audio_input output so the existing voice wrapper can read it."""
    if audio_capture is None:
        return None
    if hasattr(audio_capture, "seek"):
        audio_capture.seek(0)
    return audio_capture

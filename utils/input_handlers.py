from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def load_image(file: Any) -> Image.Image | None:
    """Load an uploaded or captured Streamlit image as RGB."""
    if file is None:
        return None
    if hasattr(file, "seek"):
        file.seek(0)
    return Image.open(file).convert("RGB")


def capture_image(capture: Any) -> Image.Image | None:
    """Alias for camera captures; keeps page code explicit."""
    return load_image(capture)


def record_audio(audio_capture: Any, seconds: int = 5) -> Any | None:
    """
    Prepare a Streamlit microphone capture for inference.

    Streamlit's browser recorder controls capture start/stop in the widget; the
    seconds argument is kept for UI intent and future recorder backends.
    """
    _ = seconds
    if audio_capture is None:
        return None
    if hasattr(audio_capture, "seek"):
        audio_capture.seek(0)
    return audio_capture


def audio_to_temp_file(audio_file: Any, suffix: str = ".wav") -> str:
    """Persist uploaded/recorded audio to a temporary file path for legacy callers."""
    if audio_file is None:
        raise ValueError("No audio file provided.")
    if hasattr(audio_file, "seek"):
        audio_file.seek(0)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        if hasattr(audio_file, "getbuffer"):
            temp.write(audio_file.getbuffer())
        else:
            temp.write(audio_file.read())
        return temp.name
    finally:
        temp.close()


def suffix_for_upload(file: Any, default: str = ".wav") -> str:
    name = str(getattr(file, "name", "") or "")
    suffix = Path(name).suffix
    return suffix or default

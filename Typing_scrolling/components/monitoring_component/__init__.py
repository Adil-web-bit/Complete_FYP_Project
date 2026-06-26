from pathlib import Path
from typing import Any, Dict, Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "frontend"

_component_func = components.declare_component(
    "monitoring_component",
    path=str(_COMPONENT_DIR),
)


def monitoring_component(key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    value = _component_func(key=key, default=None, height=650)
    if isinstance(value, dict):
        return value
    return None

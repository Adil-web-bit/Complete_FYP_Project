from __future__ import annotations

import sys
from pathlib import Path


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_import_path()


def test_monitoring_component_import_does_not_fail() -> None:
    from components.monitoring_component import monitoring_component  # noqa: F401


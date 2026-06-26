"""
Typing Behaviour Analysis Page
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent.parent
MODULE_DIR = SCRIPT_DIR / "Typing_scrolling"
APP_PATH = MODULE_DIR / "app.py"

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

spec = importlib.util.spec_from_file_location("typing_scrolling_app", APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load typing app from {APP_PATH}")

typing_scrolling_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(typing_scrolling_app)
typing_scrolling_app.main()

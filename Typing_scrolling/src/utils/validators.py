from __future__ import annotations

import math


def validate_non_empty_text(text: str) -> bool:
    """
    Return True when `text` contains at least one non-whitespace character.
    """
    return bool(text.strip())


def validate_positive_number(value: float) -> bool:
    """
    Return True when `value` is a finite positive number (> 0).
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def validate_non_negative_integer(value: int) -> bool:
    """
    Return True when `value` is an integer >= 0.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, int) and value >= 0

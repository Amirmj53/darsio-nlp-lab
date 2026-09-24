"""General utility functions."""
import re
from pathlib import Path


# Persian characters for text quality detection
PERSIAN_CHARS = set("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآأإئءۀ")
# Include Persian digits too
PERSIAN_CHARS |= set("۰۱۲۳۴۵۶۷۸۹")


def persian_ratio(text: str) -> float:
    """
    Ratio of Persian characters to total non-whitespace characters.

    Returns:
        Float between 0 and 1. If >= 0.3, text is likely meaningful Persian.
    """
    if not text:
        return 0.0

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0

    persian_count = sum(1 for c in chars if c in PERSIAN_CHARS)
    return persian_count / len(chars)


def is_gibberish(text: str) -> bool:
    """
    Detect garbled text (e.g., broken PDF font output).
    Criterion: ratio of weird characters (non-ASCII, non-Persian).
    """
    if not text or len(text) < 20:
        return True

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True

    ascii_count = sum(1 for c in chars if ord(c) < 128)
    persian_count = sum(1 for c in chars if c in PERSIAN_CHARS)
    weird_count = len(chars) - ascii_count - persian_count

    weird_ratio = weird_count / len(chars)
    return weird_ratio > 0.3


def safe_filename(name: str) -> str:
    """Sanitize filename for storage."""
    name = re.sub(r"[^\w\s\-\.]", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")[:100]
"""Text normalization and cleaning for Persian."""
import re
import unicodedata
from collections import Counter
from hazm import Normalizer


# ============================================================
# Global constants
# ============================================================

ARABIC_TO_PERSIAN = {
    "ي": "ی",
    "ك": "ک",
    "ة": "ه",
    "ۀ": "ه",
    "أ": "ا",
    "إ": "ا",
    "ؤ": "و",
    "ﻻ": "لا",
}

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

MULTI_SPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([،؛:!؟\.\»\)\]])")
SPACE_AFTER_OPEN = re.compile(r"([\(\[\«])\s+")
BROKEN_HYPHEN = re.compile(r"(\S+)-\n(\S+)")
TRIPLE_DOT = re.compile(r"\.{3,}")

# Persian letters pattern (for noise detection)
PERSIAN_LETTER = re.compile(r"[ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآأإئءۀ]")


# ============================================================
# Helper functions (called from clean_persian_text)
# ============================================================

def remove_noise_lines(text: str, min_chars: int = 3) -> str:
    """
    Remove noise lines from OCR output.

    A noise line is one that:
    - Has fewer than min_chars Persian letters
    - Has low Persian-to-total ratio
    - Has too many spaces
    - Is just repeated symbols
    - Has too many dots/dashes
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Keep empty lines (paragraph breaks)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Keep page separators
        if stripped.startswith("=====") and stripped.endswith("====="):
            cleaned_lines.append(stripped)
            continue

        total_chars = len(stripped)
        persian_count = len(PERSIAN_LETTER.findall(stripped))

        # Rule 1: too few Persian letters
        if persian_count < min_chars:
            continue

        # Rule 2: low Persian-to-total ratio
        # (line has many chars but few Persian → noise)
        if persian_count / max(total_chars, 1) < 0.5:
            continue

        # Rule 3: too many spaces
        space_ratio = stripped.count(" ") / max(total_chars, 1)
        if space_ratio > 0.3:
            continue

        # Rule 4: repeated symbols only
        if len(set(stripped.replace(" ", ""))) <= 2:
            continue

        # Rule 5: too many dots/dashes/underscores
        dot_ratio = (
            stripped.count(".") + stripped.count("-") + stripped.count("_")
        ) / max(total_chars, 1)
        if dot_ratio > 0.3:
            continue

        cleaned_lines.append(stripped)

    # Collapse multiple empty lines into one
    result = []
    prev_empty = False
    for line in cleaned_lines:
        if not line:
            if not prev_empty:
                result.append("")
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False

    return "\n".join(result)


def remove_repeated_lines(text: str, min_occurrences: int = 2) -> str:
    """
    Remove lines that repeat across multiple pages (headers/footers).
    """
    lines = text.split("\n")
    stripped_lines = [l.strip() for l in lines]

    # Count non-empty, non-separator lines
    counts = Counter(
        l for l in stripped_lines
        if l and not (l.startswith("=====") and l.endswith("====="))
    )

    # Identify repeated lines (appear >= min_occurrences times, length > 10)
    repeated = {
        l for l, c in counts.items()
        if c >= min_occurrences and len(l) > 10
    }

    # Filter out
    result = [l for l in lines if l.strip() not in repeated]
    return "\n".join(result)


def clean_ocr_text(text: str, min_persian_per_line: int = 5) -> str:
    """
    Aggressive cleaning specifically for OCR output.

    More strict than remove_noise_lines. Use this when
    OCR quality is poor (e.g., decorative backgrounds).

    Args:
        text: OCR output
        min_persian_per_line: Minimum Persian letters per line to keep

    Returns:
        Cleaned text
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Keep empty lines (paragraph breaks)
        if not stripped:
            cleaned.append("")
            continue

        # Keep page separators
        if stripped.startswith("=====") and stripped.endswith("====="):
            cleaned.append(stripped)
            continue

        total_chars = len(stripped)
        persian_count = len(PERSIAN_LETTER.findall(stripped))

        # Rule 1: minimum Persian letters
        if persian_count < min_persian_per_line:
            continue

        # Rule 2: at least 50% Persian
        if persian_count / max(total_chars, 1) < 0.5:
            continue

        # Rule 3: not too many spaces
        if stripped.count(" ") / max(total_chars, 1) > 0.25:
            continue

        # Rule 4: not just repeated symbols
        non_space = stripped.replace(" ", "")
        if len(set(non_space)) <= 3:
            continue

        cleaned.append(stripped)

    # Collapse empty lines
    result = []
    prev_empty = False
    for line in cleaned:
        if not line:
            if not prev_empty:
                result.append("")
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False

    return "\n".join(result)

# ============================================================
# Main cleaning function
# ============================================================

def clean_persian_text(
    text: str,
    aggressive: bool = False,
    digits_to_ascii: bool = True,
    remove_noise: bool = True,
    aggressive_ocr_clean: bool = False, 
) -> str:
    """
    Clean and normalize Persian text.

    Args:
        text: Raw input text
        aggressive: If True, also removes diacritics
        digits_to_ascii: Convert Persian/Arabic digits to ASCII
        remove_noise: Remove OCR noise lines and repeated headers

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove invisible control characters (keep ZWNJ)
    text = re.sub(r"[\u200b\u200e\u200f\u202a-\u202e\u2060\ufeff]", "", text)

    # 3. Arabic characters → Persian
    for ar, fa in ARABIC_TO_PERSIAN.items():
        text = text.replace(ar, fa)

    # 4. Convert digits to ASCII
    if digits_to_ascii:
        text = text.translate(ARABIC_DIGITS)
        text = text.translate(PERSIAN_DIGITS)

    # 5. Fix broken hyphenated words
    text = BROKEN_HYPHEN.sub(r"\1\2", text)

    # 6. Normalize ellipsis
    text = TRIPLE_DOT.sub("…", text)

    # 7. Remove extra spaces around punctuation
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = SPACE_AFTER_OPEN.sub(r"\1", text)
    text = MULTI_SPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)

    # 8. Strip whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 9. hazm Normalizer
    normalizer = Normalizer(correct_spacing=True)
    text = normalizer.normalize(text)
    if aggressive:
        text = normalizer.remove_diacritics(text)

    # 10. Final basic cleanup
    text = MULTI_SPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

     # 11. Remove noise lines (helpful for OCR output)
    if remove_noise:
        text = remove_noise_lines(text, min_chars=3)

        # 12. Remove repeated headers/footers
        text = remove_repeated_lines(text, min_occurrences=2)

        # 13. Aggressive OCR cleaning if needed
        if aggressive_ocr_clean:
            text = clean_ocr_text(text, min_persian_per_line=4)

    return text.strip()


# ============================================================
# Utility functions
# ============================================================

def digits_to_persian(text: str) -> str:
    """
    Convert ASCII digits back to Persian digits.
    Use when presenting output to Persian users.
    """
    return text.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def compare_before_after(raw: str, cleaned: str) -> dict:
    """Generate before/after stats for analysis."""
    return {
        "chars_before": len(raw),
        "chars_after": len(cleaned),
        "reduction_pct": round(
            (1 - len(cleaned) / max(len(raw), 1)) * 100, 2
        ),
        "arabic_yeh": raw.count("ي"),
        "arabic_kaf": raw.count("ك"),
        "arabic_digits": sum(raw.count(c) for c in "٠١٢٣٤٥٦٧٨٩"),
        "persian_digits": sum(raw.count(c) for c in "۰۱۲۳۴۵۶۷۸۹"),
        "extra_spaces": len(MULTI_SPACE.findall(raw)),
    }
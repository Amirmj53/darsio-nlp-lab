"""Darsio NLP - Persian academic notes processing engine."""

__version__ = "0.1.0"

from .extractor import extract_from_pdf, ExtractionResult, save_result
from .cleaner import clean_persian_text, digits_to_persian

__all__ = [
    "extract_from_pdf",
    "ExtractionResult",
    "save_result",
    "clean_persian_text",
    "digits_to_persian",
]
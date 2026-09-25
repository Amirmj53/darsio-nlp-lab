"""Extract text from Persian PDFs with automatic method detection."""
try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:
    import fitz  # older versions

from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

from .utils import persian_ratio, is_gibberish
from .cleaner import clean_persian_text, compare_before_after


ExtractionMethod = Literal["fitz", "ocr", "failed"]


@dataclass
class ExtractionResult:
    """Result of extracting text from a PDF."""
    pdf_path: str
    method: ExtractionMethod
    raw_text: str
    cleaned_text: str
    page_count: int
    persian_ratio: float
    quality_score: float  # 0-1
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ============================================================
# Method 1: fitz (text-based PDFs)
# ============================================================

def _extract_with_fitz(pdf_path: str) -> tuple[str, int, list[str], list[str]]:
    """
    Extract text using PyMuPDF.

    Returns:
        (text, page_count, fonts_list, warnings)
    """
    warnings = []
    all_fonts = set()
    pages_text = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    page_count = len(doc)

    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
            pages_text.append(f"\n===== Page {page_num} =====\n\n{text}")

            fonts = page.get_fonts()
            if fonts:
                all_fonts.update(f[3] for f in fonts)
        except Exception as e:
            warnings.append(f"Error extracting page {page_num}: {e}")

    doc.close()

    full_text = "".join(pages_text)
    return full_text, page_count, sorted(all_fonts), warnings


# ============================================================
# Image preprocessing (for better OCR)
# ============================================================

def _preprocess_image(image):
    """
    Preprocess image for better OCR results.

    Steps:
    1. Convert to grayscale
    2. Apply adaptive thresholding (binarization)
    3. Denoise
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    # Denoise
    denoised = cv2.medianBlur(binary, 3)

    return Image.fromarray(denoised)


# ============================================================
# Method 2: OCR (image-based PDFs)
# ============================================================


def _is_page_valid(text: str) -> tuple[bool, str]:
    """
    Check if OCR page has meaningful Persian content.

    Returns:
        (is_valid, reason)
    """
    import re

    if not text or len(text.strip()) < 50:
        return False, "too little text"

    # Count meaningful Persian words (3+ chars)
    persian_words = re.findall(
        r"[ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآأإئءۀ]{3,}",
        text,
    )

    if len(persian_words) < 10:
        return False, f"only {len(persian_words)} meaningful words"

    # Ratio of meaningful words to total tokens
    total_tokens = text.split()
    if len(total_tokens) > 0:
        word_ratio = len(persian_words) / len(total_tokens)
        if word_ratio < 0.3:
            return False, f"low word ratio ({word_ratio:.1%})"

    return True, "valid"


def _detect_ocr_language(text: str) -> tuple[str, str]:
    """
    Detect the best OCR language based on text content.

    Analyzes the ratio of Persian vs Latin characters to decide:
    - 'fas' if text is predominantly Persian
    - 'fas+eng' if text is mixed
    - 'eng' if text is predominantly English

    Args:
        text: Sample text (usually from fitz)

    Returns:
        (lang, reason) - language code and explanation
    """
    import re

    if not text or len(text.strip()) < 20:
        return "fas+eng", "not enough text to analyze, defaulting to fas+eng"

    # Count Persian letters
    persian_chars = len(re.findall(r"[\u0600-\u06FF]", text))

    # Count Latin letters
    latin_chars = len(re.findall(r"[A-Za-z]", text))

    total_letters = persian_chars + latin_chars
    if total_letters == 0:
        return "fas+eng", "no letters found, defaulting to fas+eng"

    persian_ratio = persian_chars / total_letters
    latin_ratio = latin_chars / total_letters

    # Count Latin words (3+ letters, real words)
    latin_words = re.findall(r"[A-Za-z]{3,}", text)
    latin_word_count = len(latin_words)

    # Decision logic
    if persian_ratio > 0.85:
        return "fas", f"predominantly Persian ({persian_ratio:.1%})"
    elif latin_ratio > 0.5:
        return "eng", f"predominantly English ({latin_ratio:.1%})"
    elif persian_ratio > 0.5:
        return "fas+eng", (
            f"mixed content (Persian: {persian_ratio:.1%}, "
            f"Latin: {latin_ratio:.1%}, Latin words: {latin_word_count})"
        )
    else:
        return "fas+eng", f"mixed content (Persian: {persian_ratio:.1%})"


def _detect_language_from_pdf(pdf_path: str, sample_page: int = 1) -> tuple[str, str]:
    """
    Detect OCR language by sampling a page from the PDF.

    More reliable than analyzing garbled fitz output.

    Args:
        pdf_path: Path to PDF
        sample_page: Page number to sample (1-indexed)

    Returns:
        (lang, reason)
    """
    import re

    try:
        from pdf2image import convert_from_path
        import pytesseract

        # Convert just one page
        images = convert_from_path(
            pdf_path,
            dpi=200,
            first_page=sample_page,
            last_page=sample_page,
        )

        if not images:
            return "fas+eng", "could not extract sample page"

        # Quick OCR with both languages to see which fits better
        sample_text = pytesseract.image_to_string(
            images[0],
            lang="fas+eng",
            config="--oem 3 --psm 4",
        )

        if not sample_text or len(sample_text.strip()) < 20:
            return "fas+eng", "sample page has too little text"

        # Analyze
        persian_chars = len(re.findall(r"[\u0600-\u06FF]", sample_text))
        latin_chars = len(re.findall(r"[A-Za-z]", sample_text))
        total = persian_chars + latin_chars

        if total == 0:
            return "fas+eng", "no letters detected"

        persian_ratio = persian_chars / total
        latin_ratio = latin_chars / total

        if persian_ratio > 0.7:
            return "fas", f"sample page mostly Persian ({persian_ratio:.1%})"
        elif latin_ratio > 0.7:
            return "eng", f"sample page mostly English ({latin_ratio:.1%})"
        else:
            return "fas+eng", (
                f"sample page mixed "
                f"(Persian: {persian_ratio:.1%}, Latin: {latin_ratio:.1%})"
            )
    except Exception as e:
        return "fas+eng", f"detection failed: {e}"


def _trim_ocr_result(text: str, lang: str) -> str:
    """
    Post-process OCR result based on detected language.

    If lang='fas', removes isolated Latin characters (OCR noise).
    If lang='fas+eng', keeps everything.
    """
    import re

    if lang == "fas":
        # Remove isolated Latin characters (2 chars or less)
        text = re.sub(r"(?<!\w)[A-Za-z]{1,2}(?!\w)", "", text)
        # Clean up double spaces
        text = re.sub(r"\s+", " ", text)

    return text

def _extract_with_ocr(
    pdf_path: str,
    dpi: int = 300,
    lang: str = "fas",
    max_pages: int | None = None,
    start_page: int = 1,
    preprocess: bool = True,
    skip_pages: list[int] | None = None,
) -> tuple[str, int, list[str]]:
    """
    Extract text using OCR (Tesseract).

    Args:
        pdf_path: Path to PDF
        dpi: Image resolution
        lang: Tesseract language code
        max_pages: Limit number of pages (None = all)
        start_page: First page to process (1-indexed)
        preprocess: Apply image preprocessing before OCR
        skip_pages: List of page numbers to skip

    Returns:
        (text, pages_processed, warnings)
    """
    from pdf2image import convert_from_path
    import pytesseract

    warnings = []
    pages_text = []
    skip_pages = skip_pages or []

    # Step 1: Convert PDF pages to images
    try:
        kwargs = {"dpi": dpi}
        if max_pages is not None:
            kwargs["first_page"] = start_page
            kwargs["last_page"] = start_page + max_pages - 1

        images = convert_from_path(pdf_path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}")

    pages_processed = len(images)

    # Tesseract config
    custom_config = r"--oem 3 --psm 4"

    # Step 2: OCR each image
    for idx, image in enumerate(images, start=start_page):
        # Skip manually specified pages
        if idx in skip_pages:
            warnings.append(f"Skipped page {idx}: user-specified")
            continue

        try:
            # Preprocess if requested
            if preprocess:
                processed = _preprocess_image(image)
            else:
                processed = image

            text = pytesseract.image_to_string(
                processed,
                lang=lang,
                config=custom_config,
            )
            text = _trim_ocr_result(text, lang)

            # Validate page content
            is_valid, reason = _is_page_valid(text)
            if not is_valid:
                warnings.append(f"Skipped page {idx}: {reason}")
                continue

            pages_text.append(f"\n===== Page {idx} =====\n\n{text}")
        except Exception as e:
            warnings.append(f"OCR error on page {idx}: {e}")

    full_text = "".join(pages_text)
    return full_text, pages_processed, warnings


# ============================================================
# Decision logic: which method to use?
# ============================================================

def _should_use_ocr(
    raw_text: str,
    persian_ratio_val: float,
    page_count: int,
) -> tuple[bool, str]:
    """
    Decide whether OCR is needed.

    Returns:
        (needs_ocr, reason)
    """
    # Case 1: Empty or nearly empty text
    if len(raw_text.strip()) < 50:
        return True, "Extracted text is nearly empty"

    # Case 2: Garbled text
    if is_gibberish(raw_text):
        return True, "Extracted text is garbled (gibberish)"

    # Case 3: Very low Persian ratio
    if persian_ratio_val < 0.1:
        return True, f"Persian character ratio is too low ({persian_ratio_val:.2%})"

    # Case 4: Very low chars-per-page average
    chars_per_page = len(raw_text.strip()) / max(page_count, 1)
    if chars_per_page < 100:
        return True, f"Average chars per page is too low ({chars_per_page:.0f})"

    return False, "Text is extractable with fitz"


# ============================================================
# Main entry point
# ============================================================

def extract_from_pdf(
    pdf_path: str,
    use_ocr_if_needed: bool = True,
    clean: bool = True,
    aggressive_clean: bool = False,
    digits_to_ascii: bool = True,
    ocr_dpi: int = 300,
    ocr_max_pages: int | None = None,
    ocr_preprocess: bool = True,
    aggressive_ocr_clean: bool = False, 
    skip_pages: list[int] | None = None,
) -> ExtractionResult:
    """
    Extract text from a PDF with automatic method detection.

    Args:
        pdf_path: Path to PDF file
        use_ocr_if_needed: Run OCR if fitz fails
        clean: Apply text cleaning
        aggressive_clean: Aggressive cleaning (removes diacritics)
        digits_to_ascii: Convert digits to ASCII (recommended for LLM)
        ocr_dpi: OCR image resolution
        ocr_max_pages: Max pages to OCR (None = all)
        ocr_preprocess: Apply preprocessing before OCR

    Returns:
        ExtractionResult
    """
    pdf_path = str(Path(pdf_path).resolve())

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Step 1: Try fitz
    fitz_raw_text, page_count, fonts, warnings = _extract_with_fitz(pdf_path)
    fitz_p_ratio = persian_ratio(fitz_raw_text)

    # Keep initial fitz stats for reporting
    fitz_chars = len(fitz_raw_text)

    # Step 2: Decide method
    needs_ocr, reason = _should_use_ocr(fitz_raw_text, fitz_p_ratio, page_count)

    raw_text = fitz_raw_text
    p_ratio = fitz_p_ratio
    method: ExtractionMethod = "fitz"
    ocr_chars = 0
    ocr_pages_processed = 0
    ocr_lang = None          # ← جدید
    lang_reason = None 

    if needs_ocr:
        warnings.append(f"OCR suggested: {reason}")

        if use_ocr_if_needed:
            # Choose detection method based on fitz quality
            if fitz_p_ratio > 0.1:
                # fitz has some Persian → fast method
                ocr_lang, lang_reason = _detect_ocr_language(fitz_raw_text)
            else:
                # fitz is garbage → reliable image-based method
                ocr_lang, lang_reason = _detect_language_from_pdf(pdf_path)
            
            warnings.append(f"OCR language: '{ocr_lang}' ({lang_reason})")

            try:
                ocr_text, ocr_pages_processed, ocr_warnings = _extract_with_ocr(
                    pdf_path,
                    dpi=ocr_dpi,
                    lang=ocr_lang,   # ← اینجا از lang تشخیص‌داده‌شده استفاده کن
                    max_pages=ocr_max_pages,
                    preprocess=ocr_preprocess,
                    skip_pages=skip_pages,
                )
                warnings.extend(ocr_warnings)

                ocr_chars = len(ocr_text)
                ocr_p_ratio = persian_ratio(ocr_text)

                # Use OCR result if it's better
                if ocr_p_ratio > p_ratio and len(ocr_text.strip()) > 100:
                    raw_text = ocr_text
                    p_ratio = ocr_p_ratio
                    method = "ocr"
                    warnings.append(
                        f"OCR successful: {ocr_pages_processed} pages processed, "
                        f"Persian ratio {ocr_p_ratio:.2%}"
                    )
                else:
                    warnings.append(
                        f"OCR result not better than fitz "
                        f"(OCR: {ocr_p_ratio:.2%} vs fitz: {fitz_p_ratio:.2%}). "
                        f"Using fitz output."
                    )
            except Exception as e:
                warnings.append(f"OCR failed: {e}. Using fitz output.")

    # Step 3: Clean
    # Auto-enable aggressive OCR clean if method was OCR
    use_aggressive_ocr = aggressive_ocr_clean or (method == "ocr")

    cleaned = (
        clean_persian_text(
            raw_text,
            aggressive=aggressive_clean,
            digits_to_ascii=digits_to_ascii,
            aggressive_ocr_clean=use_aggressive_ocr,
        )
        if clean
        else raw_text
    )

    # Step 4: Compute quality score
    quality = min(1.0, p_ratio * 1.5)
    if len(raw_text.strip()) < 500:
        quality *= 0.5
    if is_gibberish(raw_text):
        quality *= 0.2

    # Step 5: Stats
    stats = compare_before_after(raw_text, cleaned) if clean else {}
    stats["fonts"] = fonts
    stats["needs_ocr"] = needs_ocr
    stats["ocr_reason"] = reason
    stats["method_used"] = method
    stats["raw_source"] = method
    # Transparency: show fitz vs ocr sizes
    stats["fitz_chars"] = fitz_chars
    stats["ocr_chars"] = ocr_chars if method == "ocr" else 0
    stats["final_raw_chars"] = len(raw_text)
    if method == "ocr":
        stats["ocr_pages_processed"] = ocr_pages_processed

    if ocr_lang:
        stats["ocr_lang"] = ocr_lang
        stats["ocr_lang_reason"] = lang_reason

    return ExtractionResult(
        pdf_path=pdf_path,
        method=method,
        raw_text=raw_text,
        cleaned_text=cleaned,
        page_count=page_count,
        persian_ratio=p_ratio,
        quality_score=round(quality, 3),
        warnings=warnings,
        stats=stats,
    )


def save_result(result: ExtractionResult, output_dir: str) -> dict:
    """Save extraction result to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(result.pdf_path).stem

    raw_path = output_dir / f"{base_name}.raw.txt"
    cleaned_path = output_dir / f"{base_name}.cleaned.txt"

    raw_path.write_text(result.raw_text, encoding="utf-8")
    cleaned_path.write_text(result.cleaned_text, encoding="utf-8")

    return {
        "raw": str(raw_path),
        "cleaned": str(cleaned_path),
    }
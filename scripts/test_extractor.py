"""Test the extraction module on PDF files."""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from darsio_nlp import extract_from_pdf, save_result
from darsio_nlp.config import EXTRACTED_DIR


def print_result(result, elapsed: float):
    """Print extraction report."""
    print("=" * 70)
    print(f"📄 File: {Path(result.pdf_path).name}")
    print(f"📊 Pages: {result.page_count}")
    print(f"🔧 Method: {result.method}")
    print(f"🇮🇷 Persian ratio: {result.persian_ratio:.2%}")
    print(f"⭐ Quality: {result.quality_score}")
    print(f"⏱️  Elapsed: {elapsed:.2f}s")
    print("=" * 70)

    if result.warnings:
        print("\n⚠️  Warnings:")
        for w in result.warnings:
            print(f"   • {w}")

    print("\n📊 Stats:")
    for k, v in result.stats.items():
        if k == "fonts":
            print(f"   • Fonts: {v}")
        else:
            print(f"   • {k}: {v}")

    print("\n📖 Sample text (raw):")
    print("-" * 70)
    print(repr(result.raw_text[:300]))
    print("-" * 70)

    print("\n✨ Sample text (cleaned):")
    print("-" * 70)
    print(result.cleaned_text[:500])
    print("-" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_extractor.py <path_to_pdf> [options]")
        print("\nOptions:")
        print("  --ocr-max-pages N   Limit OCR to N pages (for testing)")
        print("  --ocr-dpi N         OCR DPI (default: 300)")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Parse options
    ocr_max_pages = None
    ocr_dpi = 300

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--ocr-max-pages" and i + 1 < len(args):
            ocr_max_pages = int(args[i + 1])
            i += 2
        elif args[i] == "--ocr-dpi" and i + 1 < len(args):
            ocr_dpi = int(args[i + 1])
            i += 2
        else:
            i += 1

    print(f"🚀 Starting extraction: {pdf_path}")
    if ocr_max_pages:
        print(f"   OCR limited to {ocr_max_pages} pages")
    print()

    start = time.time()
    result = extract_from_pdf(
        pdf_path,
        clean=True,
        ocr_dpi=ocr_dpi,
        ocr_max_pages=ocr_max_pages,
    )
    elapsed = time.time() - start

    print_result(result, elapsed)

    paths = save_result(result, str(EXTRACTED_DIR))
    print(f"\n💾 Saved:")
    print(f"   • Raw: {paths['raw']}")
    print(f"   • Cleaned: {paths['cleaned']}")


if __name__ == "__main__":
    main()
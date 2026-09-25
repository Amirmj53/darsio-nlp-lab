"""
Direct DeepSeek API test for PDF summarization.
Usage:
    python scripts/direct_api_test.py data/input_pdfs/your_file.pdf
"""
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from darsio_nlp import extract_from_pdf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# DeepSeek Pricing (USD per 1M tokens) - Off-Peak
# ============================================================
PRICING = {
    "input_cache_miss": 0.14,
    "input_cache_hit": 0.0028,
    "output": 0.28,
}
USD_TO_TOMAN = 235_000


def estimate_tokens(text: str) -> int:
    """Rough token estimation for Persian text (1 token ≈ 2 chars)."""
    return len(text) // 2


def calculate_cost(usage) -> dict:
    """Calculate cost from API usage object."""
    input_miss = getattr(usage, "prompt_cache_miss_tokens", 0)
    input_hit = getattr(usage, "prompt_cache_hit_tokens", 0)
    output = getattr(usage, "completion_tokens", 0)

    cost_miss = (input_miss / 1_000_000) * PRICING["input_cache_miss"]
    cost_hit = (input_hit / 1_000_000) * PRICING["input_cache_hit"]
    cost_output = (output / 1_000_000) * PRICING["output"]
    total_usd = cost_miss + cost_hit + cost_output

    return {
        "input_cache_miss_tokens": input_miss,
        "input_cache_hit_tokens": input_hit,
        "output_tokens": output,
        "cost_cache_miss_usd": round(cost_miss, 6),
        "cost_cache_hit_usd": round(cost_hit, 6),
        "cost_output_usd": round(cost_output, 6),
        "total_usd": round(total_usd, 6),
        "total_toman": round(total_usd * USD_TO_TOMAN, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()

    # Step 1: Extract text from PDF using our NLP pipeline
    print("=" * 60)
    print(f"📄 Extracting: {args.pdf_path}")
    result = extract_from_pdf(args.pdf_path, use_ocr_if_needed=True)
    print(f"   Method: {result.method}")
    print(f"   Pages: {result.page_count}")
    print(f"   Cleaned chars: {len(result.cleaned_text):,}")
    print(f"   Estimated tokens: {estimate_tokens(result.cleaned_text):,}")

    # Step 2: Prepare prompt for DeepSeek
    # Put the PDF text as a fixed SYSTEM message so it can be cached
    system_prompt = f"""شما یک دستیار مطالعه هستید. متن جزوه زیر را به دقت بخوانید و بر اساس آن به سوالات پاسخ دهید.

<جزوه>
{result.cleaned_text}
</جزوه>
"""

    user_prompt = "این جزوه را در ۳ پاراگراف خلاصه کن."

    # Step 3: Call DeepSeek API
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

    print("\n🚀 Sending request to DeepSeek...")
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )

    # Step 4: Print summary and usage
    print("\n" + "=" * 60)
    print("📝 SUMMARY:")
    print("=" * 60)
    print(response.choices[0].message.content[:500] + "...")

    print("\n" + "=" * 60)
    print("💰 USAGE & COST (DeepSeek Flash, Off-Peak)")
    print("=" * 60)

    usage = response.usage
    cost = calculate_cost(usage)

    print(f"   Input (Cache Miss): {cost['input_cache_miss_tokens']:,} tokens")
    print(f"   Input (Cache Hit):  {cost['input_cache_hit_tokens']:,} tokens")
    print(f"   Output:             {cost['output_tokens']:,} tokens")
    print("-" * 60)
    print(f"   Cost (Cache Miss):  ${cost['cost_cache_miss_usd']:.6f}")
    print(f"   Cost (Cache Hit):   ${cost['cost_cache_hit_usd']:.6f}")
    print(f"   Cost (Output):      ${cost['cost_output_usd']:.6f}")
    print("-" * 60)
    print(f"   TOTAL:              ${cost['total_usd']:.6f}  (~{cost['total_toman']:,.0f} Toman)")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
Generate exam questions from PDF using DeepSeek API.
Usage:
    python scripts/generate_questions.py data/input_pdfs/your_file.pdf --per-topic 5
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from darsio_nlp import extract_from_pdf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PRICING = {
    "input_cache_miss": 0.14,
    "input_cache_hit": 0.0028,
    "output": 0.28,
}
USD_TO_TOMAN = 235_000


def calculate_cost(usage) -> dict:
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
        "total_usd": round(total_usd, 6),
        "total_toman": round(total_usd * USD_TO_TOMAN, 2),
    }


# ============================================================
# Prompt Engineering
# ============================================================

def build_system_prompt(notes_text: str) -> str:
    """
    Build the system prompt with the notes as fixed context.
    This is cached by DeepSeek for subsequent requests.
    """
    return f"""شما یک استاد دانشگاه با ۲۰ سال تجربه در طراحی سوالات امتحانی هستید.
تخصص شما طراحی سوالات استاندارد، دقیق و مفهومی از جزوه‌های دانشگاهی است.

متن جزوه مورد نظر در ادامه آمده است. این متن را به دقت مطالعه کنید:

<جزوه>
{notes_text}
</جزوه>

قوانین مهم:
- فقط از محتوای همین جزوه سوال طراحی کنید.
- از اطلاعات خارج از جزوه استفاده نکنید.
- به سطح دشواری و نوع سوالات توجه کنید.
- سوالات باید دقیق، بدون ابهام و قابل پاسخ باشند.
"""


def build_user_prompt(
    topic: str,
    num_questions: int = 5,
    difficulty: str = "medium",
    question_types: list[str] = None,
) -> str:
    """
    Build the user prompt for generating questions from a specific topic.

    Args:
        topic: The topic/section to generate questions from
        num_questions: Number of questions
        difficulty: "easy" | "medium" | "hard" | "mixed"
        question_types: List of question types to include
    """
    if question_types is None:
        question_types = ["تعریف", "توضیح", "مقایسه", "کاربرد", "تحلیل"]

    difficulty_map = {
        "easy": "آسان (سطح یادآوری و درک)",
        "medium": "متوسط (سطح کاربرد و تحلیل)",
        "hard": "دشوار (سطح ترکیب و ارزیابی)",
        "mixed": "ترکیبی از سطوح آسان، متوسط و دشوار",
    }

    types_str = "، ".join(question_types)

    return f"""از مبحث «{topic}» در جزوه، دقیقاً {num_questions} سوال امتحانی طراحی کنید.

مشخصات سوالات:
- سطح دشواری: {difficulty_map.get(difficulty, difficulty_map["medium"])}
- انواع سوالات: {types_str}
- همه سوالات باید تشریحی باشند (نه چند گزینه‌ای)
- برای هر سوال، پاسخ کامل و دقیق هم بنویسید
- سوالات نباید تکراری باشند
- سوالات باید مستقیماً از متن جزوه استخراج شده باشند

خروجی را دقیقاً به این فرمت JSON بدهید:

{{
  "topic": "{topic}",
  "questions": [
    {{
      "id": 1,
      "type": "نوع سوال (تعریف/توضیح/مقایسه/کاربرد/تحلیل)",
      "difficulty": "آسان/متوسط/دشوار",
      "question": "متن سوال",
      "answer": "پاسخ کامل و دقیق",
      "source_hint": "اشاره به بخشی از جزوه که سوال از آن آمده"
    }}
  ]
}}

فقط JSON برگردانید، هیچ متن اضافه‌ای قبل یا بعد از آن ننویسید.
"""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--topic", default="کل جزوه",
                        help="Topic to generate questions from")
    parser.add_argument("--per-topic", type=int, default=5,
                        help="Number of questions per topic")
    parser.add_argument("--difficulty", default="mixed",
                        choices=["easy", "medium", "hard", "mixed"])
    parser.add_argument("--output", default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    # Step 1: Extract text
    print("=" * 60)
    print(f"📄 Extracting: {args.pdf_path}")
    result = extract_from_pdf(args.pdf_path, use_ocr_if_needed=True)
    print(f"   Method: {result.method}")
    print(f"   Pages: {result.page_count}")
    print(f"   Cleaned chars: {len(result.cleaned_text):,}")

    # Step 2: Build prompts
    system_prompt = build_system_prompt(result.cleaned_text)
    user_prompt = build_user_prompt(
        topic=args.topic,
        num_questions=args.per_topic,
        difficulty=args.difficulty,
    )

    # Step 3: Call DeepSeek
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

    print(f"\n🚀 Generating {args.per_topic} questions on '{args.topic}'...")
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    # Step 4: Parse and display
    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse JSON. Raw output:")
        print(content)
        return

    print("\n" + "=" * 60)
    print(f"📝 QUESTIONS ON: {data.get('topic', args.topic)}")
    print("=" * 60)

    for q in data.get("questions", []):
        print(f"\n🔹 سوال {q['id']} [{q['type']} - {q['difficulty']}]")
        print(f"   {q['question']}")
        print(f"\n   ✅ پاسخ:")
        print(f"   {q['answer']}")
        print(f"\n   📖 منبع: {q.get('source_hint', 'نامشخص')}")
        print("-" * 60)

    # Step 5: Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/outputs") / f"{Path(args.pdf_path).stem}_questions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n💾 Saved: {output_path}")

    # Step 6: Report usage & cost
    usage = response.usage
    cost = calculate_cost(usage)

    print("\n" + "=" * 60)
    print("💰 USAGE & COST")
    print("=" * 60)
    print(f"   Input (Cache Miss): {cost['input_cache_miss_tokens']:,}")
    print(f"   Input (Cache Hit):  {cost['input_cache_hit_tokens']:,}")
    print(f"   Output:             {cost['output_tokens']:,}")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL: ${cost['total_usd']:.6f}  (~{cost['total_toman']:,.0f} Toman)")
    print("=" * 60)


if __name__ == "__main__":
    main()
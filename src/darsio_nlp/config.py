import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

INPUT_PDF_DIR = DATA_DIR / "input_pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted_text"
CLEANED_DIR = DATA_DIR / "cleaned_text"
CHUNKS_DIR = DATA_DIR / "chunks"
COMPRESSED_DIR = DATA_DIR / "compressed"
OUTPUTS_DIR = DATA_DIR / "outputs"


for d in [INPUT_PDF_DIR, EXTRACTED_DIR, CLEANED_DIR, 
          CHUNKS_DIR, COMPRESSED_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"


MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "500"))
COMPRESSION_RATIO = float(os.getenv("COMPRESSION_RATIO", "0.3"))
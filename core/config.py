from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_FILE = DATA_DIR / "documents.db"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
JSON_OUTPUT = DATA_DIR / "documents.json"

EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
EMBEDDING_DIM = 384

LLM_MODEL = "models/gemini-3-flash-preview"

CHUNK_SIZE = 850
CHUNK_OVERLAP = 140

SEED_URLS = [
    "https://cloud.ru/docs",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Referer": "https://cloud.ru/",
}

from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_FILE = DATA_DIR / "documents.db"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
JSON_OUTPUT = DATA_DIR / "cloud_ru_docs.json"

EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
EMBEDDING_DIM = 384

LLM_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SEED_URLS = [
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__compute",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__network",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__containers",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__storage",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__brokers",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__database",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__dataplatform",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__ai-factory",
    "https://cloud.ru/docs/tutorials-evolution/list/topics/index__monitoring-management",
    "https://cloud.ru/docs/tutorials-evolution/list/index?source-platform=Evolution",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Referer": "https://cloud.ru/",
}

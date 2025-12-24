from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_FILE = DATA_DIR / "documents.db"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
JSON_OUTPUT = DATA_DIR / "documents.json"

EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
EMBEDDING_DIM = 384

# LLM providers
LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_QWEN = "qwen"

# models
LLM_MODEL_GEMINI = "models/gemini-3-flash-preview"
LLM_MODEL_QWEN = "qwen2.5:7b"
LLM_RERANK_MODEL = "models/gemma-3-27b-it"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# defaults
LLM_DEFAULT_PROVIDER = LLM_PROVIDER_GEMINI
LLM_DEFAULT_MODEL = LLM_MODEL_GEMINI
LLM_MODEL = LLM_DEFAULT_MODEL

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

SEED_URLS = [
    "https://cloud.ru/docs",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Referer": "https://cloud.ru/",
}

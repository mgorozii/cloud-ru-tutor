# Cloud.ru Tutor

RAG-система для обучения по документации Cloud.ru

📄 [Отчёт о проекте](REPORT.md)

## Структура

```
cloud-ru-tutor/
├── core/
│   ├── config.py      # все настройки и пути
│   ├── logger.py      # structlog
│   └── models.py      # dataclasses
│
├── services/
│   ├── crawler.py     # парсинг сайта
│   ├── storage.py     # работа с БД
│   ├── embeddings_service.py  # векторизация
│   └── rag.py         # генерация ответов
│
├── data/              # все БД и файлы
│   ├── documents.db
│   ├── documents.json
│   └── chroma_db/
│
├── app.py             # UI (Streamlit)
├── pipeline_full.py   # CLI парсинг
└── embeddings.py      # CLI индексация
```

## Быстрый старт

Для работы понадобится Python 3.11+, [uv](https://docs.astral.sh/uv/) и [GEMINI_API_KEY](https://aistudio.google.com/app/api-keys).

```bash
# Установка
uv sync

# Парсинг документации (опционально)
uv run python crawl.py

# Создание эмбедингов (создает SQLite + ChromaDB)
uv run python embeddings.py

# Запуск интерфейса
# Добавьте ваш GEMINI_API_KEY в .streamlit/secrets.toml
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
uv run streamlit run app.py
```

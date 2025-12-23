# Cloud.ru Tutor

RAG-система для обучения по документации Cloud.ru

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

Для работы понадобится Python 3.11+ и [uv](https://docs.astral.sh/uv/) и и [GEMINI_API_KEY](https://aistudio.google.com/app/api-keys).

```bash
# Установка
uv sync

# Парсинг документации (опционально)
uv run python crawl.py

# Создание эмбедингов (создает SQLite + ChromaDB)
uv run python embeddings.py

### Настройка API-ключа Gemini
Добавьте ваш `GEMINI_API_KEY` в `.streamlit/secrets.toml`:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Откройте .streamlit/secrets.toml и вставьте ваш ключ

# Запуск интерфейса
uv run streamlit run app.py
```
По умолчанию используется **Gemini** (работает через API, не требует локальной установки).

---
## Использование локальной модели Qwen (опционально)

Если хотите использовать **Qwen** (локальная модель без API-ключей и интернета):

### 1. Установите Ollama
- **Windows/Mac/Linux**: [ollama.com](https://ollama.com/download)

### 2. Скачайте модель Qwen
ollama pull qwen2.5:7b

Проверьте, что модель скачалась:
ollama list

### 3. Запустите Ollama (если не запустилась автоматически)
ollama serve

Оставьте это окно открытым или запустите Ollama как службу.

### 4. Выбор модели в UI
После запуска Streamlit (`uv run streamlit run app.py`):
- В боковой панели найдите **"⚙️ Выбор LLM"**
- Переключите провайдер с `gemini` на `qwen`
- Выберите нужный тег модели (например, `qwen2.5:7b`)

Теперь ответы генерируются локально через Qwen без отправки данных в облако.
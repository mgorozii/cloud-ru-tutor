# Cloud.ru AI-Репетитор — Отчёт

## Команда

- Алексей
- Дмитрий
- Максим
- Никита

## Ход работы

| Дата | Этап |
|------|------|
| 4 дек | Выбор кейса AI Репетитор Cloud.ru |
| 7 дек | Первая встреча: обсуждение RAG-архитектуры, стека, сбор материалов в Miro |
| 9 дек | Первая попытка краулинга документации cloud.ru |
| 10 дек | Доработка краулера, первый рабочий прототип |
| 13 дек | Рефакторинг, настройка инфраструктуры, публикация на GitHub |
| 13 дек | Встреча с ментором: обратная связь по качеству и подходам |
| 14 дек | Фиксы, режим тестирования, базовая фильтрация контента |
| 16 дек | Встреча с заказчиком: положительный фидбек, получены примеры вопросов |
| 21 дек | Финальная доработка, проверка по критериям, сдача |
| 23 дек | Интеграция Ollama (Qwen), выбор LLM в UI |

## Артефакты

- **Репозиторий**: https://github.com/mgorozii/cloud-ru-tutor
- **Отчёт**: https://github.com/mgorozii/cloud-ru-tutor/blob/master/REPORT.md

## Архитектура

### Компоненты системы

\`\`\`plantuml
@startuml architecture
skinparam backgroundColor white
skinparam componentStyle rectangle

package "Frontend" {
  [Streamlit UI] as UI
}

package "Backend" {
  [RAG Service] as RAG
  [Embeddings Service] as EMB
  [Content Moderator] as MOD
  [Storage] as STORE
}

package "External" {
  [Gemini API] as GEMINI
  [Ollama (Qwen)] as QWEN
}

database "ChromaDB" as CHROMA
database "SQLite" as SQL

UI --> MOD : фильтрация
UI --> EMB : поиск
UI --> RAG : генерация
RAG --> GEMINI
RAG --> QWEN
RAG --> CHROMA
EMB --> CHROMA
STORE --> SQL
@enduml
\`\`\`

### RAG Pipeline

\`\`\`plantuml
@startuml pipeline
skinparam backgroundColor white

start
:Вопрос пользователя;
:Фильтрация входа (Moderator);

if (Разрешено?) then (да)
  :Векторизация запроса (rubert-tiny2);
  :Поиск в ChromaDB (top_k * 3);
  :LLM Rerank (Gemma 27B);
  :Расширение контекста (соседние чанки);
  :Генерация ответа (Gemini / Qwen);
  :Фильтрация выхода (Moderator);
  :Ответ + источники;
else (нет)
  :Предупреждение;
endif

stop
@enduml
\`\`\`

### Структура данных

\`\`\`plantuml
@startuml data
skinparam backgroundColor white

entity Document {
  * url : string
  --
  title : string
  text : string
}

entity Chunk {
  * id : string
  --
  source_url : string
  source_title : string
  chunk_index : int
  text : string
}

entity ChromaVector {
  * id : string
  --
  embedding : float[384]
  metadata : json
}

Document ||--|{ Chunk : chunking
Chunk ||--|| ChromaVector : indexing
@enduml
\`\`\`

## Технологии

| Компонент | Технология | Описание |
|-----------|------------|----------|
| UI | Streamlit | Веб-интерфейс с чатом и тестами |
| Embeddings | rubert-tiny2 | Векторизация (384-мерные векторы) |
| Vector DB | ChromaDB | Семантический поиск (HNSW, cosine) |
| LLM (Cloud) | Gemini API | Генерация ответов |
| LLM (Local) | Ollama (Qwen) | Локальная генерация |
| Rerank | Gemma 27B | Переранжирование результатов |
| Crawler | Scrapy + Trafilatura | Парсинг документации |
| Storage | SQLite | Хранение чанков |
| Logger | structlog | Структурированное логирование |

## Инструкция по созданию AI-репетитора

### 1. Сбор данных
- Определить источники (документация, учебники)
- Реализовать краулер с фильтрацией по домену
- Извлечь текст (Trafilatura)
- Сохранить в JSON

### 2. Обработка и индексация
- Разбить документы на чанки (512 символов, overlap 100)
- Сохранить чанки в SQLite
- Создать эмбеддинги (rubert-tiny2)
- Индексировать в ChromaDB

### 3. RAG Pipeline
- Поиск: векторный поиск по косинусной близости
- Rerank: LLM-переранжирование
- Расширение контекста: соседние чанки
- Генерация: промпт с контекстом → LLM → ответ

### 4. Модерация
- Фильтрация входа: блокировка нецензурного контента
- Маскировка PII (email, телефоны, карты)
- Фильтрация выхода: удаление опасного кода

### 5. UI
- Чат-интерфейс с историей
- Отображение источников
- Генерация тестов для самопроверки
- Выбор LLM провайдера (Gemini / Qwen)

## Выбор LLM провайдера

### Gemini (по умолчанию)
- Требования: API ключ (GEMINI_API_KEY)
- Плюсы: быстрый, высокое качество
- Минусы: отправляет данные в облако

### Qwen (опционально)
- Требования: Ollama установлен локально
- Плюсы: приватность, без API-лимитов
- Минусы: требует ~8 ГБ RAM для 7B модели

## Масштабируемость

- **Новые предметы**: добавить URL в \`SEED_URLS\`, перезапустить краулер
- **Другие базы знаний**: реализовать новый loader в \`Storage\`
- **Другие LLM**: изменить провайдер в конфиге или добавить новый

## Этика и безопасность

- Фильтрация нецензурного контента (500+ слов)
- Маскировка персональных данных (PII)
- Защита от prompt injection
- API-ключи в secrets (не коммитятся в Git)

## Запуск

См. [README.md](README.md)

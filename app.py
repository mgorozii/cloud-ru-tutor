#!/usr/bin/env python3
import os
import streamlit as st
from core import DATABASE_FILE
from core.config import (
    LLM_PROVIDER_GEMINI,
    LLM_PROVIDER_QWEN,
    LLM_MODEL_GEMINI,
    LLM_MODEL_QWEN,
)
from services import EmbeddingsService, RAG, Storage
from services.moderation import ContentModerator

st.set_page_config(page_title="Cloud.ru Tutor", layout="wide")


@st.cache_resource
def init_services():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY не найден.")
        st.stop()

    try:
        embeddings = EmbeddingsService()
        collection = embeddings.load_chroma()

        if collection.count() == 0:
            with st.spinner("Индексируем документы..."):
                storage = Storage(DATABASE_FILE)
                chunks = storage.load_chunks()
                if chunks:
                    embeddings.index_chunks(chunks, collection)
                    st.success(f"Проиндексировано {len(chunks)} чанков")
                else:
                    st.error("Нет данных для индексации")
                    st.stop()

        rag = RAG(api_key)
        storage = Storage(DATABASE_FILE)
        moderator = ContentModerator()

        return embeddings, collection, rag, storage, moderator

    except Exception as e:
        st.error(f"Ошибка инициализации: {str(e)}")
        st.stop()


def generate_quiz(rag, topic: str, context_docs: list):
    if not topic:
        return "Введите тему для создания вопросов"

    context = "\n\n".join([doc["text"] for doc in context_docs[:3]])

    prompt = f"""Создай 3 вопроса с вариантами ответов по теме: {topic}

Контекст:
```
{context}
```

Формат каждого вопроса:
**N. Текст вопроса**<br>
a) вариант<br><br>b) вариант<br><br>c) вариант<br><br>d) вариант

<details><summary>Ответ</summary>

**Ответ:** буква<br><br>
**Почему:** объяснение

</details>
<br><br>
---
"""

    try:
        return rag.generate(
            prompt,
            [],
            model_name=st.session_state.llm_model,
            provider=st.session_state.llm_provider,
        )
    except Exception as e:
        return f"Ошибка: {str(e)}"


def expand_context(storage, docs: list, neighbor_count: int = 1) -> list:
    expanded = []
    seen = set()

    for doc in docs:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        chunks = storage.get_chunk_with_neighbors(
            doc_id, neighbor_count=neighbor_count, trim_overlap=True
        )
        for chunk in chunks:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            expanded.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": {"url": chunk.source_url, "title": chunk.source_title},
                }
            )

    return expanded or docs


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Cloud.ru AI-Репетитор</h1>",
        unsafe_allow_html=True,
    )

    embeddings, collection, rag, storage, moderator = init_services()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = LLM_PROVIDER_GEMINI
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = LLM_MODEL_GEMINI

    with st.sidebar:
        st.header("Настройки")
        top_k = st.slider("Количество источников", 3, 15, 5)
        show_sources = st.checkbox("Показывать источники", value=True)

        st.divider()
        st.subheader("Статистика")
        try:
            count = collection.count()
            st.metric("Чанков в БД", count)
        except Exception:
            st.metric("Чанков в БД", "N/A")

        st.metric("Размер вектора", embeddings.embedding_dim)

        # === Выбор провайдера и модели ===
        st.divider()
        st.subheader("⚙️ Выбор LLM")

        provider = st.selectbox(
            "Провайдер",
            [LLM_PROVIDER_GEMINI, LLM_PROVIDER_QWEN],
            index=0 if st.session_state.llm_provider == LLM_PROVIDER_GEMINI else 1,
        )

        if provider == LLM_PROVIDER_GEMINI:
            model_options = [
                LLM_MODEL_GEMINI,
                "models/gemini-1.5-flash",
                "models/gemma-3-27b-it",
            ]
        else:  # Qwen
            model_options = [LLM_MODEL_QWEN, "qwen2.5:3b", "qwen2.5:1.5b"]

        selected_model = st.selectbox("Модель", model_options, index=0)

        # Сохраняем выбор
        st.session_state.llm_provider = provider
        st.session_state.llm_model = selected_model

    tab1, tab2 = st.tabs(["Чат", "Самопроверка"])

    with tab1:
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False
        if "pending_question" not in st.session_state:
            st.session_state.pending_question = None

        messages_container = st.container()
        with messages_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if show_sources and "sources" in msg and msg["sources"]:
                        with st.expander("Источники"):
                            for src in msg["sources"]:
                                if src.get("url") and src.get("title"):
                                    st.markdown(f"[{src['title']}]({src['url']})")

        # bottom input: disabled during generation, active otherwise
        input_container = st.container()
        with input_container:
            user_input = None

            if st.session_state.is_generating:
                st.chat_input(
                    "Генерирую ответ...",
                    key="chat_input_disabled",
                    disabled=True,
                )
            else:
                user_input = st.chat_input(
                    "Задайте вопрос о Cloud.ru...",
                    key="chat_input",
                )

        # on submit: show user bubble immediately, set generating state, rerun to disable input
        if user_input:
            allowed, filtered = moderator.filter_input(user_input)
            if not allowed:
                st.warning(filtered)
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.last_user_question = user_input
                st.session_state.pending_question = user_input
                st.session_state.is_generating = True
                st.rerun()

        # generation phase: run when is_generating and pending_question set
        if st.session_state.is_generating and st.session_state.pending_question:
            question = st.session_state.pending_question

            with st.spinner("Поиск информации..."):
                search_results = embeddings.search(question, collection, top_k * 3)

            if not search_results:
                st.warning("По вашему запросу ничего не найдено.")
                response = "Не нашел информации по вашему вопросу. Попробуйте переформулировать."
                top_docs = []
            else:
                with st.spinner("Генерация ответа..."):
                    top_docs = rag.rerank(question, search_results, top_k)
                    context_docs = expand_context(storage, top_docs, neighbor_count=1)
                    response = moderator.filter_response(
                        rag.generate(
                            question,
                            context_docs,
                            model_name=st.session_state.llm_model,
                            provider=st.session_state.llm_provider,
                        )
                    )

            sources = []
            for doc in top_docs:
                src_info = storage.get_source_info(doc["id"])
                if src_info and src_info.get("url"):
                    sources.append(
                        {
                            "title": src_info.get("title", "Без названия"),
                            "url": src_info["url"],
                        }
                    )
                elif doc.get("metadata") and doc["metadata"].get("url"):
                    sources.append(
                        {
                            "title": doc["metadata"].get("title", "Без названия"),
                            "url": doc["metadata"]["url"],
                        }
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources}
            )

            # reset generating state and clear pending
            st.session_state.pending_question = None
            st.session_state.is_generating = False
            # clear input value if used
            if "chat_input" in st.session_state:
                st.session_state.chat_input = ""
            st.rerun()

    with tab2:
        st.subheader("Создание вопросов для самопроверки")

        last_question = st.session_state.get("last_user_question", "")

        if last_question:
            st.write(f"Последний вопрос в чате: {last_question}")
            topic = st.text_input(
                "Тема для вопросов (или оставьте предыдущий):", value=last_question
            )
        else:
            topic = st.text_input("Введите тему для создания вопросов:")

        if st.button("Создать вопросы для самопроверки"):
            if not topic:
                st.error("Введите тему для создания вопросов")
            else:
                with st.spinner("Создаю вопросы..."):
                    search_results = embeddings.search(topic, collection, 5)
                    if search_results:
                        top_docs = rag.rerank(topic, search_results, 3)
                        quiz = generate_quiz(rag, topic, top_docs)
                        st.markdown(quiz, unsafe_allow_html=True)
                    else:
                        st.warning(
                            "Не найдено информации по теме. Попробуйте другую тему."
                        )


if __name__ == "__main__":
    main()

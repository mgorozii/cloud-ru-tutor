#!/usr/bin/env python3
import os
import streamlit as st
from core import DATABASE_FILE
from core.config import LLM_MODEL
from services import EmbeddingsService, RAG, Storage

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

        return embeddings, collection, rag, storage

    except Exception as e:
        st.error(f"Ошибка инициализации: {str(e)}")
        st.stop()


def generate_quiz(rag, topic: str, context_docs: list):
    if not topic:
        return "Введите тему для создания вопросов"

    context = "\n\n".join([doc["text"] for doc in context_docs[:3]])

    prompt = f"""
    Создай 3 вопроса для самопроверки по теме: {topic}
    
    Контекст:
    {context}
    
    Формат ответа:
    1. [Вопрос]
       a) Вариант 1
       b) Вариант 2
       c) Вариант 3
       d) Вариант 4
       Ответ: [буква правильного ответа]
       Объяснение: [краткое объяснение]
    
    2. [Вопрос]
       a) Вариант 1
       b) Вариант 2
       c) Вариант 3
       d) Вариант 4
       Ответ: [буква правильного ответа]
       Объяснение: [краткое объяснение]
    
    3. [Вопрос]
       a) Вариант 1
       b) Вариант 2
       c) Вариант 3
       d) Вариант 4
       Ответ: [буква правильного ответа]
       Объяснение: [краткое объяснение]
    """

    try:
        response = rag.generate(prompt, [])
        return response
    except Exception as e:
        return f"Ошибка генерации вопросов: {str(e)}"


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Cloud.ru AI-Репетитор</h1>",
        unsafe_allow_html=True,
    )

    embeddings, collection, rag, storage = init_services()

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
        st.metric("LLM модель", LLM_MODEL)

    tab1, tab2 = st.tabs(["Чат", "Самопроверка"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("Источники"):
                        for src in msg["sources"]:
                            if src.get("url") and src.get("title"):
                                st.markdown(f"[{src['title']}]({src['url']})")

        if user_input := st.chat_input("Задайте вопрос о Cloud.ru..."):
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Поиск информации..."):
                    search_results = embeddings.search(user_input, collection, top_k)

                if not search_results:
                    st.warning("По вашему запросу ничего не найдено.")
                    response = "Не нашел информации по вашему вопросу. Попробуйте переформулировать."
                    top_docs = []
                else:
                    with st.spinner("Генерация ответа..."):
                        top_docs = rag.rerank(search_results, 3)
                        response = rag.generate(user_input, top_docs)

                    st.markdown(response)

                    if show_sources and top_docs:
                        with st.expander("Источники информации"):
                            for i, doc in enumerate(top_docs, 1):
                                src_info = storage.get_source_info(doc["id"])
                                if src_info and src_info.get("url"):
                                    st.markdown(
                                        f"{i}. [{src_info.get('title', 'Без названия')}]({src_info['url']})"
                                    )
                                elif doc.get("metadata") and doc["metadata"].get("url"):
                                    st.markdown(
                                        f"{i}. [{doc['metadata'].get('title', 'Без названия')}]({doc['metadata']['url']})"
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

                st.session_state.last_user_question = user_input

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
                        top_docs = rag.rerank(search_results, 3)
                        quiz = generate_quiz(rag, topic, top_docs)
                        st.markdown(quiz)
                    else:
                        st.warning(
                            "Не найдено информации по теме. Попробуйте другую тему."
                        )


if __name__ == "__main__":
    main()

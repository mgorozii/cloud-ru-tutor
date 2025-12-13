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
        st.error("GEMINI_API_KEY not found")
        st.stop()

    embeddings = EmbeddingsService()
    collection = embeddings.load_chroma()
    rag = RAG(api_key)
    storage = Storage(DATABASE_FILE)

    return embeddings, collection, rag, storage


def main():
    st.markdown(
        "<h1 style='text-align: center;'>Cloud.ru Tutor</h1>", unsafe_allow_html=True
    )

    embeddings, collection, rag, storage = init_services()

    with st.sidebar:
        st.header("Настройки")
        top_k = st.slider("Документов для поиска", 5, 20, 10)

        st.divider()
        st.subheader("Статистика")
        st.metric("Чанков", collection.count())
        st.metric(
            "Размер вектора",
            getattr(embeddings, "embedding_dim", None)
            or embeddings.model.get_sentence_embedding_dimension(),
        )
        st.metric("LLM", LLM_MODEL)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Источники"):
                    for src in msg["sources"]:
                        st.markdown(f"[{src['title']}]({src['url']})")

    if user_input := st.chat_input("Вопрос о Cloud.ru..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Поиск..."):
                search_results = embeddings.search(user_input, collection, top_k)

            if not search_results:
                st.warning("Ничего не найдено")
            else:
                top_docs = rag.rerank(search_results, 3)

                with st.spinner("Генерация..."):
                    answer = rag.generate(user_input, top_docs)

                st.markdown(answer)

                sources = []
                seen = set()
                for doc in top_docs:
                    if src := storage.get_source_info(doc["id"]):
                        if src["url"] not in seen:
                            sources.append(src)
                            seen.add(src["url"])

                if sources:
                    with st.expander("Источники"):
                        for src in sources:
                            st.markdown(f"[{src['title']}]({src['url']})")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )


if __name__ == "__main__":
    main()

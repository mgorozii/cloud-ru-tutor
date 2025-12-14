#!/usr/bin/env python3
import os
import streamlit as st
from pathlib import Path
from core import DATABASE_FILE, JSON_OUTPUT
from core.config import LLM_MODEL, SEED_URLS
from services import EmbeddingsService, RAG, Storage, Crawler
from services.moderation import ContentModerator  # Новый модуль для модерации

st.set_page_config(page_title="Cloud.ru Tutor", layout="wide")

def check_and_create_data():
    """Проверка и создание данных при необходимости"""
    if not DATABASE_FILE.exists():
        st.warning("База данных не найдена. Создаем...")
        
        with st.spinner("Собираем данные с Cloud.ru..."):
            crawler = Crawler()
            documents = crawler.crawl(SEED_URLS, max_pages=100)
            
            storage = Storage(DATABASE_FILE)
            storage.save_json(documents, JSON_OUTPUT)
            storage.save_chunks(documents)
        
        st.success(f"Собрано {len(documents)} документов")
        return True
    
    if DATABASE_FILE.stat().st_size == 0:
        st.error("База данных пуста. Удалите файл и перезапустите приложение.")
        return False
    
    return True

@st.cache_resource
def init_services():
    """Инициализация сервисов с проверкой"""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY не найден. Установите в секретах Streamlit или переменных окружения.")
        st.stop()
    
    # Проверяем и создаем данные
    if not check_and_create_data():
        st.stop()
    
    # Инициализируем сервисы
    try:
        embeddings = EmbeddingsService()
        
        collection = embeddings.load_chroma(force_recreate=False)
        
        # Проверяем, есть ли данные в коллекции
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
        
        return {
            "embeddings": embeddings,
            "collection": collection,
            "rag": rag,
            "storage": storage,
            "moderator": moderator
        }
        
    except Exception as e:
        st.error(f"Ошибка инициализации: {str(e)}")
        st.stop()

def generate_quiz(rag, topic: str, context_docs: list):
    """Генерация вопросов для самопроверки"""
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
    
    Сделай вопросы разного уровня сложности.
    """
    
    try:
        response = rag.generate(prompt, [])
        return response
    except Exception as e:
        return f"Ошибка генерации вопросов: {str(e)}"

def main():
    st.markdown("""
        <h1 style='text-align: center; color: #1E3A8A;'>
            🤖 Cloud.ru AI-Репетитор
        </h1>
        <p style='text-align: center; color: #6B7280;'>
            Помощник по техническим дисциплинам и облачным технологиям
        </p>
    """, unsafe_allow_html=True)
    
    # Инициализация сервисов
    services = init_services()
    embeddings = services["embeddings"]
    collection = services["collection"]
    rag = services["rag"]
    storage = services["storage"]
    moderator = services["moderator"]
    
    # Сайдбар
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Выбор режима
        mode = st.radio(
            "Режим работы:",
            ["📚 Обучение", "🔍 Поиск материалов"]
        )
        
        # Параметры поиска
        top_k = st.slider("Количество источников", 3, 15, 5)
        show_sources = st.checkbox("Показывать источники", value=True)
        
        st.divider()
        
        # Информация о системе
        st.subheader("📊 Статистика")
        col1, col2 = st.columns(2)
        
        try:
            count = collection.count()
            col1.metric("Чанков в БД", count)
        except:
            col1.metric("Чанков в БД", "N/A")
        
        col2.metric("Размер вектора", embeddings.embedding_dim)
        st.metric("LLM модель", LLM_MODEL)
        
        # Кнопка переиндексации
        st.divider()
        if st.button("🔄 Переиндексировать базу", type="secondary"):
            with st.spinner("Переиндексация..."):
                try:
                    storage = Storage(DATABASE_FILE)
                    chunks = storage.load_chunks()
                    collection = embeddings.init_chroma()
                    embeddings.index_chunks(chunks, collection)
                    st.success("База переиндексирована")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
    
    # Основная область
    if mode == "📚 Обучение":
        tab1, tab2, tab3 = st.tabs(["💬 Чат", "📖 Материалы", "🎯 Вопросы"])
        
        with tab1:
            # История сообщений
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Отображение истории
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg:
                        with st.expander("📚 Источники"):
                            for src in msg["sources"]:
                                st.markdown(f"**{src['title']}**")
                                st.caption(f"📎 {src['url']}")
            
            # Ввод пользователя
            if prompt := st.chat_input("Задайте вопрос о Cloud.ru..."):
                # Проверка контента
                is_valid, message = moderator.filter_input(prompt)
                if not is_valid:
                    st.warning(message)
                    st.stop()
                
                # Добавление в историю
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Поиск информации..."):
                        search_results = embeddings.search(prompt, collection, top_k)
                    
                    if not search_results:
                        st.warning("По вашему запросу ничего не найдено.")
                        response = "Извините, не нашел информации по вашему вопросу. Попробуйте переформулировать."
                    else:
                        with st.spinner("💭 Генерация ответа..."):
                            top_docs = sorted(search_results, key=lambda x: x["similarity"], reverse=True)[:3]
                            response = rag.generate(prompt, top_docs)
                        
                        # Отображение ответа
                        st.markdown(response)
                        
                        # Источники
                        if show_sources and search_results:
                            with st.expander("📚 Источники информации"):
                                for i, doc in enumerate(top_docs, 1):
                                    st.markdown(f"**{i}. {doc['metadata'].get('title', 'Без названия')}**")
                                    st.caption(doc['metadata'].get('url', ''))
                                    st.caption(f"Релевантность: {doc['similarity']:.2%}")
                    
                    # Добавление в историю
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": [
                            storage.get_source_info(doc["id"]) 
                            for doc in top_docs[:3] 
                            if storage.get_source_info(doc["id"])
                        ] if search_results else []
                    })
        
        with tab2:
            st.subheader("Рекомендуемые материалы")
            # Здесь можно добавить рекомендательную систему
        
        with tab3:
            st.subheader("Вопросы для самопроверки")
            topic = st.text_input("Введите тему для вопросов:")
            if st.button("Сгенерировать вопросы") and topic:
                with st.spinner("Создаю вопросы..."):
                    docs = embeddings.search(topic, collection, top_k=5)
                    quiz = generate_quiz(rag, topic, docs)
                    st.markdown(quiz)
        
    elif mode == "🔍 Поиск материалов":
        st.subheader("Поиск по материалам")
        search_query = st.text_input("Что ищем?")
        if search_query:
            results = embeddings.search(search_query, collection, 10)
            for i, doc in enumerate(results, 1):
                with st.expander(f"Результат {i}: {doc['metadata'].get('title', 'Без названия')}"):
                    st.markdown(doc["text"][:500] + "...")
                    st.caption(f"📊 Релевантность: {doc['similarity']:.2%}")
                    if st.button("📋 Копировать", key=f"copy_{i}"):
                        st.code(doc["text"])

if __name__ == "__main__":
    main()
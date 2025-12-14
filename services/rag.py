import os

import google.generativeai as genai

from core import get_logger
from core.config import LLM_MODEL

log = get_logger(__name__)


class RAG:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        self.model_name = model_name or LLM_MODEL
        genai.configure(api_key=self.api_key)

    def rerank(self, documents: list[dict], top_k: int = 3) -> list[dict]:
        sorted_docs = sorted(documents, key=lambda x: x["similarity"], reverse=True)
        return sorted_docs[:top_k]

    def generate(
        self, query: str, documents: list[dict], model_name: str | None = None
    ) -> str:
        context = "\n\n".join(
            [
                f"Источник: {doc['metadata'].get('title', 'Без названия')}\n{doc['text']}"
                for doc in documents
            ]
        )
        context = context[:4000]
        model_to_use = model_name or self.model_name

        system_prompt = """Ты репетитор по Cloud.ru. Помогаешь разбираться в облачных сервисах.

Правила:
1. Отвечай кратко и ясно
2. Используй контекст из документации
3. Если контекста недостаточно, честно скажи
4. Максимум 300 слов"""

        user_message = f"""Контекст:\n{context}\n\nВопрос: {query}"""

        try:
            model = genai.GenerativeModel(model_to_use)
            response = model.generate_content(
                [system_prompt, user_message],
                generation_config={"max_output_tokens": 1000, "temperature": 0.7},
            )
            return response.text
        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "quota" in error_str.lower():
                if "limit: 0" in error_str:
                    return f"Модель {model_to_use} недоступна. Попробуй gemini-1.5-flash."
                return "Лимит API исчерпан. Подожди немного."

            log.error("generation_failed", model=model_to_use, error=error_str[:200])
            return f"Ошибка генерации ({model_to_use}): {error_str[:200]}"

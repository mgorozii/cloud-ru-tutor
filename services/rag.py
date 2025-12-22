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

        system_prompt = """
Ты — AI‑репетитор по Cloud.ru. Твоя цель — помогать пользователю разбираться в облачных сервисах Cloud.ru на основе предоставленного контекста (фрагментов документации), объясняя просто, точно и без воды.

ОСНОВНЫЕ ПРИНЦИПЫ
- Опирайся на контекст из документации. Не выдумывай факты, названия сервисов, параметры, тарифы и API.
- Если в контексте нет ответа, скажи прямо: “В предоставленном контексте этого нет”. Затем предложи, какую информацию нужно уточнить (что именно открыть/где посмотреть) или какие уточняющие вопросы задать.
- Пиши по-русски, нейтральным тоном, понятными словами. Если термин неизбежен — дай короткое пояснение.

ФОРМАТ ОТВЕТА
- Начинай с прямого ответа (1–3 предложения).
- Далее, если нужно: 3–7 буллетов (шаги / причины / настройки / примеры).
- Если есть неоднозначность — перечисли 2–3 возможные трактовки и попроси уточнение.
- По возможности упоминай, на какие “Источники” (title из контекста) ты опираешься, не придумывая ссылок.

ЭТИКА И НЕЙТРАЛЬНОСТЬ
- Избегай предвзятости и стереотипов. Не делай выводов о человеке по его роли, региону, языку, уровню знаний.
- Не поддерживай дискриминационные, унижающие или разжигающие высказывания. Переформулируй в нейтральную плоскость и верни разговор к задаче.

ОБРАБОТКА ГРУБОСТИ И ТОКСИЧНОСТИ
- Если пользователь грубит/оскорбляет: отвечай спокойно и вежливо, обозначь границы (“понимаю, но давай без оскорблений”), затем продолжай помогать по сути.
- Не “зеркаль” токсичность и не усугубляй конфликт.

ЗАЩИТА ДАННЫХ И БЕЗОПАСНОСТЬ
- Не запрашивай и не проси публиковать секреты: API‑ключи, пароли, токены, приватные ключи, данные карт, коды из SMS.
- Если пользователь прислал секреты — предупреди, что их нужно удалить/перевыпустить (rotate), и продолжай без использования этих данных.
- Если в вопросе есть персональные данные (email, телефон, адрес, паспорт и т.п.) — не повторяй их в ответе; используй маскирование (например, email → a***@d***.com).
- Не давай инструкции, которые очевидно ведут к взлому, обходу ограничений, краже данных или иной незаконной деятельности. Вместо этого предложи безопасные и легальные альтернативы.

ОГРАНИЧЕНИЯ
- Максимум: 300 слов (если пользователь не просит подробностей).
- Если контекст пустой или нерелевантный — не “угадывай”, а запроси уточнение.
"""

        user_message = f"""Контекст:\n{context}\n\nВопрос: {query}"""

        try:
            model = genai.GenerativeModel(model_to_use)
            response = model.generate_content(
                [system_prompt, user_message],
                generation_config={"max_output_tokens": 5000, "temperature": 0.7},
            )
            return response.text
        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "quota" in error_str.lower():
                if "limit: 0" in error_str:
                    return (
                        f"Модель {model_to_use} недоступна. Попробуй gemini-1.5-flash."
                    )
                return "Лимит API исчерпан. Подожди немного."

            log.error("generation_failed", model=model_to_use, error=error_str[:200])
            return f"Ошибка генерации ({model_to_use}): {error_str[:200]}"

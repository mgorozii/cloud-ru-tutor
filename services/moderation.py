import re
from typing import Tuple, List
from core import get_logger

log = get_logger(__name__)

class ContentModerator:
    def __init__(self):
        self.blocked_words = [
            # Ругательства
            "мат", "брань", "оскорбление",
            # И т.д.
        ]
        
        self.sensitive_patterns = [
            r"(?i)(пароль|кредит[а-я]*\s*карт|pin.*код)",
            r"(?i)(личн[а-я]*\s*данн)",
        ]
        
        self.learning_keywords = [
            "cloud", "облако", "docker", "kubernetes", "сервер",
            "база данных", "контейнер", "виртуализация", "сеть"
        ]

    def filter_input(self, text: str) -> Tuple[bool, str]:
        """Фильтрация входящих сообщений"""
        text_lower = text.lower()
        
        # Проверка на запрещенные слова
        for word in self.blocked_words:
            if re.search(rf'\b{word}\b', text_lower):
                log.warning("blocked_content", word=word)
                return False, "Сообщение содержит недопустимые выражения"
        
        # Проверка на чувствительную информацию
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text_lower):
                log.warning("sensitive_content", pattern=pattern)
                return False, "Запрос содержит потенциально чувствительную информацию"
        
        # Проверка релевантности (опционально)
        has_keywords = any(keyword in text_lower for keyword in self.learning_keywords)
        if not has_keywords and len(text.split()) > 5:
            log.info("off_topic_query", query=text[:50])
            return True, text  # Все равно разрешаем, но логируем
        
        return True, text

    def filter_response(self, response: str) -> str:
        """Фильтрация исходящих ответов"""
        # Проверка на наличие вредоносного кода
        code_patterns = [
            r"<script>.*?</script>",
            r"javascript:",
            r"onclick=",
            r"onload="
        ]
        
        for pattern in code_patterns:
            response = re.sub(pattern, "[скрипт удален]", response, flags=re.IGNORECASE)
        
        return response
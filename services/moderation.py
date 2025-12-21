import re
from typing import Tuple

from badwords import ProfanityFilter

from core import get_logger

log = get_logger(__name__)

_filter = ProfanityFilter()
_filter.init(["ru", "en"])


class PIIMasker:
    patterns = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        (
            r"\b(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b",
            "[PHONE]",
        ),
        (r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[CARD]"),
        (r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{2}\b", "[SNILS]"),
        (r"\b\d{2}[\s]?\d{2}[\s]?\d{6}\b", "[PASSPORT]"),
        (r"\b(?:пароль|password)[\s:=]+\S+", "[PASSWORD]"),
        (r"\b(?:api[_\-]?key|token|secret)[\s:=]+\S+", "[SECRET]", re.I),
    ]

    @classmethod
    def mask(cls, text: str) -> str:
        for item in cls.patterns:
            pattern, repl = item[0], item[1]
            flags = item[2] if len(item) > 2 else 0
            text = re.sub(pattern, repl, text, flags=flags)
        return text


class ContentModerator:
    def filter_input(self, text: str) -> Tuple[bool, str]:
        if _filter.filter_text(text, match_threshold=1.0):
            log.warning("blocked_content")
            return False, "Сообщение содержит недопустимые выражения"

        masked = PIIMasker.mask(text)
        if masked != text:
            log.info("pii_detected_input")

        return True, masked

    def filter_response(self, response: str) -> str:
        code_patterns = [
            r"<script>.*?</script>",
            r"javascript:",
            r"on(?:click|load|error|mouse\w+)=",
        ]
        for pattern in code_patterns:
            response = re.sub(pattern, "[removed]", response, flags=re.I | re.S)

        return PIIMasker.mask(response)

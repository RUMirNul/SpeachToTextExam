import re
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detects language and assesses recognition quality."""

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language based on alphabet (Cyrillic/Latin)."""
        cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))

        if cyrillic_count == 0 and latin_count == 0:
            return "unknown"

        total = cyrillic_count + latin_count
        cyrillic_ratio = cyrillic_count / total if total > 0 else 0

        if cyrillic_ratio > 0.6:
            return "ru"
        elif cyrillic_ratio < 0.4:
            return "en"
        else:
            return "mixed"

    @staticmethod
    def assess_quality(text: str) -> float:
        """Assess recognition quality (0.0 - 1.0)."""
        if not text.strip():
            return 0.0

        words = text.split()
        word_count = len(words)
        letter_count = len([c for c in text if c.isalpha()])

        # Word count score (0-1)
        word_score = min(word_count / 10, 1.0) * 0.2

        # Letter count score (0-1)
        letter_score = min(letter_count / 50, 1.0) * 0.2

        # Alphabet homogeneity score
        cyrillic_ratio = len(re.findall(r'[а-яёА-ЯЁ]', text)) / max(letter_count, 1)
        latin_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(letter_count, 1)
        alphabet_score = max(cyrillic_ratio, latin_ratio) * 0.2

        # Text length score
        text_len = len(text)
        length_score = 0.2
        if 10 < text_len < 200:
            length_score = 0.2
        elif text_len < 5 or text_len > 300:
            length_score = 0.05

        # Transliteration penalty
        transliteration_penalty = 1.0 if not LanguageDetector.is_transliteration(text) else 0.3

        quality = (word_score + letter_score + alphabet_score + length_score + 0.2) * transliteration_penalty
        return min(quality, 1.0)

    @staticmethod
    def is_transliteration(text: str) -> bool:
        """Check if text is transliteration of English words in Cyrillic."""
        words = text.split()
        if len(words) < 3:
            return False

        short_words = sum(1 for w in words if 1 <= len(w) <= 3)
        long_words = sum(1 for w in words if len(w) >= 7)

        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0

        criteria = 0
        if short_words > len(words) * 0.55:
            criteria += 1
        if avg_word_len < 5.2:
            criteria += 1
        if long_words < len(words) * 0.25:
            criteria += 1

        return criteria >= 2
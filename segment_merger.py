import re
from typing import List, Tuple, Optional
import logging
from transcription_segment import TranscriptionSegment
from language_detector import LanguageDetector

logger = logging.getLogger(__name__)


class SegmentMerger:
    """
    Объединяет сегменты из моделей RU и EN с помощью интеллектуальной дедупликации
    используя функцию _find_corresponding_segment() для интеллектуального сопоставления.
    """

    OVERLAP_THRESHOLD = 0.5
    MERGE_GAP_THRESHOLD = 0.3
    TIME_TOLERANCE = 0.1
    MIN_QUALITY_THRESHOLD = 0.50  # Минимальное качество для доверия RU

    @staticmethod
    def merge_segments(
            segments_ru: List[TranscriptionSegment],
            segments_en: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Объедините сегменты из двух моделей, выбрав наилучшие для перекрывающихся интервалов.
        """
        if not segments_ru and not segments_en:
            return []

        logger.debug(f"Объединение {len(segments_ru)} RU + {len(segments_en)} EN сегментов")

        merged = SegmentMerger.merge_overlapping_pairs(segments_ru, segments_en)
        logger.debug(f"После слияния наложений: {len(merged)} сегментов")

        merged = SegmentMerger.merge_adjacent_identical(merged)
        logger.debug(f"После слияния соседних идентичных: {len(merged)} сегментов")

        merged = SegmentMerger.remove_complete_duplicates(merged)
        logger.debug(f"После удаления дубликатов: {len(merged)} сегментов")

        merged.sort(key=lambda s: s.start_time)

        return merged

    @staticmethod
    def _find_corresponding_segment(
            ru_segment: TranscriptionSegment,
            en_segments: List[TranscriptionSegment],
            overlap_threshold: float = OVERLAP_THRESHOLD
    ) -> Optional[TranscriptionSegment]:
        """
        Поиск наиболее подходящего английского сегмента для русского сегмента.

        Стратегия:
        1. Вычислите временное перекрытие для всех сегментов EN
        2. Найдите сегмент(ы) с наибольшим перекрытием
        3. Верните сегмент с наилучшей оценкой качества, если перекрытие > порогового значения
        4. Верните None, если не найдено подходящего совпадения

        Agrs:
            ru_segment: русский сегмент для поиска
            en_segments: Список английских сегментов для поиска
            overlap_threshold: Минимальный коэффициент перекрытия (0,0-1,0)

        return:
            Наилучшее соответствие фрагмента транскрипции или его отсутствие
        """
        if not en_segments:
            logger.debug(
                f"Нет EN сегментов для RU: '{ru_segment.text[:40]}...'"
            )
            return None

        best_en_segment = None
        best_overlap = 0.0
        best_quality = 0.0
        candidates = []

        for idx, en_segment in enumerate(en_segments):
            overlap = SegmentMerger.calculate_overlap(ru_segment, en_segment)

            if overlap > overlap_threshold:
                en_quality = LanguageDetector.assess_quality(en_segment.text)
                combined_score = overlap * 0.6 + en_quality * 0.4

                candidates.append({
                    'segment': en_segment,
                    'overlap': overlap,
                    'quality': en_quality,
                    'score': combined_score,
                    'index': idx
                })

                if combined_score > (best_overlap * 0.6 + best_quality * 0.4):
                    best_en_segment = en_segment
                    best_overlap = overlap
                    best_quality = en_quality

                    logger.debug(
                        f"  Кандидат #{idx + 1}: перекрытие={overlap:.2f}, "
                        f"качество={en_quality:.2f}, очки={combined_score:.2f} "
                        f"'{en_segment.text[:30]}...'"
                    )

        if best_en_segment:
            logger.debug(
                f"  Найден соответствующий сегмент EN:\n"
                f"  RU: {ru_segment.start_time:.2f}-{ru_segment.end_time:.2f} "
                f"'{ru_segment.text[:40]}...'\n"
                f"  EN: {best_en_segment.start_time:.2f}-{best_en_segment.end_time:.2f} "
                f"'{best_en_segment.text[:40]}...'\n"
                f"  перекрытие: {best_overlap:.2f}, качество: {best_quality:.2f}"
            )
        else:
            if candidates:
                logger.debug(
                    f"  Нет совпадений выше порога качества для RU: "
                    f"'{ru_segment.text[:40]}...' "
                    f"({len(candidates)} кандидатов найдено)"
                )
            else:
                logger.debug(
                    f" Не найдено EN сегментов перекрывающих RU: "
                    f"'{ru_segment.text[:40]}...'"
                )

        return best_en_segment

    @staticmethod
    def merge_overlapping_pairs(
            segments_ru: List[TranscriptionSegment],
            segments_en: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
       Для каждого RU сегмента находит перекрывающий EN сегмент и выберает наилучший.
        """
        result = []
        used_en_indices = set()

        for ru_seg in segments_ru:
            en_seg = SegmentMerger._find_corresponding_segment(
                ru_seg, segments_en,
                overlap_threshold=SegmentMerger.OVERLAP_THRESHOLD
            )

            if en_seg:
                best_seg = SegmentMerger.select_best_segment(ru_seg, en_seg)
                logger.debug(
                    f"Выбран: {best_seg.start_time:.2f}-{best_seg.end_time:.2f} "
                    f"'{best_seg.text[:30]}...'"
                )
                result.append(best_seg)

                en_idx = segments_en.index(en_seg)
                used_en_indices.add(en_idx)
            else:
                result.append(ru_seg)

        for en_idx, en_seg in enumerate(segments_en):
            if en_idx not in used_en_indices:
                lang_en = LanguageDetector.detect_language(en_seg.text)
                quality_en = LanguageDetector.assess_quality(en_seg.text)
                is_transliteration_en = LanguageDetector.is_transliteration(en_seg.text)

                if (lang_en == "en" and not is_transliteration_en and
                        quality_en > 0.5):
                    result.append(en_seg)
                    logger.debug(
                        f"Добавлен не перекрытый EN сегмент: {en_seg.start_time:.2f}-"
                        f"{en_seg.end_time:.2f} '{en_seg.text[:30]}...'"
                    )

        return result

    @staticmethod
    def calculate_overlap(
            seg1: TranscriptionSegment,
            seg2: TranscriptionSegment
    ) -> float:
        """
        Вычисление коэффициента перекрытия между двумя сегментами.

        Возвращает значение 0,0-1,0:
        - 0,0: перекрытия нет
        - 1,0: полное ограничение
        """
        intersection_start = max(seg1.start_time, seg2.start_time)
        intersection_end = min(seg1.end_time, seg2.end_time)

        if intersection_end <= intersection_start:
            return 0.0

        intersection = intersection_end - intersection_start
        min_duration = min(
            seg1.end_time - seg1.start_time,
            seg2.end_time - seg2.start_time
        )

        if min_duration <= 0:
            return 0.0

        return intersection / min_duration

    @staticmethod
    def select_best_segment(
            seg_ru: TranscriptionSegment,
            seg_en: TranscriptionSegment
    ) -> TranscriptionSegment:
        """
        Выбирает лучший сегмент в зависимости от языка, транслитерации и качества.

        Добавлены проверки на пригодность, позволяющие избежать выбора поврежденных сегментов.
        """
        lang_ru = LanguageDetector.detect_language(seg_ru.text)
        lang_en = LanguageDetector.detect_language(seg_en.text)

        quality_ru = LanguageDetector.assess_quality(seg_ru.text)
        quality_en = LanguageDetector.assess_quality(seg_en.text)

        is_transliteration_ru = LanguageDetector.is_transliteration(seg_ru.text)
        is_transliteration_en = LanguageDetector.is_transliteration(seg_en.text)

        ru_avg_len = SegmentMerger._get_avg_word_length(seg_ru.text)
        en_avg_len = SegmentMerger._get_avg_word_length(seg_en.text)

        ru_short_ratio = SegmentMerger._get_short_word_ratio(seg_ru.text)
        en_short_ratio = SegmentMerger._get_short_word_ratio(seg_en.text)

        logger.debug(
            f"Сравнение: RU(lang={lang_ru}, q={quality_ru:.2f}, "
            f"транслит={is_transliteration_ru}, сред. длина={ru_avg_len:.2f}) "
            f"vs EN(lang={lang_en}, q={quality_en:.2f}, "
            f"транслит={is_transliteration_en}, сред. длина={en_avg_len:.2f})"
        )

        #1: RU качество слишком низкое = явный не подходит
        if quality_ru < SegmentMerger.MIN_QUALITY_THRESHOLD and quality_en >= quality_ru:
            logger.debug(
                f" Качество RU сегмента критически мало  ({quality_ru:.2f} < "
                f"{SegmentMerger.MIN_QUALITY_THRESHOLD}), "
                f"likely corrupted/garbage → SELECT EN"
            )
            return seg_en

        #2: EN качество намного лучше RU
        if quality_en > quality_ru + 0.25:
            logger.debug(
                f" Качество EN значительно выше "
                f"(EN={quality_en:.2f} vs RU={quality_ru:.2f}, diff=+{quality_en - quality_ru:.2f}) "
                f" Выбран EN"
            )
            return seg_en

        # Правило 1: Найдет чистый русский
        if lang_ru == "ru" and not is_transliteration_ru:
            if ru_avg_len > 4.5 and quality_ru >= SegmentMerger.MIN_QUALITY_THRESHOLD:
                logger.debug(" Чисты русский с нормальным качеством и средней длиной слов → выбран RU")
                return seg_ru
            elif (ru_avg_len < 3.5 and en_avg_len > 4.0 and
                  ru_short_ratio > 0.7):
                logger.debug(
                    f" Скрытая транслитерация найдена: "
                    f"RU avg={ru_avg_len:.2f} vs EN avg={en_avg_len:.2f} → выбран EN"
                )
                return seg_en

        # Правило 2: RU - транслит
        if lang_ru == "ru" and is_transliteration_ru:
            logger.debug(f" RU - это транслит → выбран EN")
            return seg_en

        # Правило 3: Оба сегмента - русский язык
        if lang_ru == "ru" and lang_en == "ru":
            if is_transliteration_en and not is_transliteration_ru:
                logger.debug(" Оба сегмента - русский язык, но EN - трансирован → выбран RU")
                return seg_ru
            logger.debug(
                " Оба сегмента - русский язык → выбран RU"
            )
            return seg_ru

        # Правило 4: Чистый EN
        if lang_en == "en" and not is_transliteration_en:
            if lang_ru == "en" or lang_ru == "mixed":
                logger.debug(" EN - чистый английский, RU - смешанный или английский → выбран EN")
                return seg_en

        # Правило 5: Проверка по качеству
        if is_transliteration_ru and not is_transliteration_en:
            logger.debug(" RU - транслирован, EN чистый → выбран EN")
            return seg_en

        # Rule 6: Default choice
        choice = seg_ru if quality_ru >= quality_en else seg_en
        logger.debug(
            f"→ Стандартный выбор по качеству: RU={quality_ru:.2f} "
            f"vs EN={quality_en:.2f} → выбран {'RU' if choice == seg_ru else 'EN'}"
        )
        return choice

    @staticmethod
    def _get_avg_word_length(text: str) -> float:
        """Вычисление средней длины слов."""
        words = re.findall(r'[а-яА-ЯёЁa-zA-Z]+', text.lower())

        if not words:
            return 0.0

        if re.search(r'[а-яА-ЯёЁ]', text):
            filtered_words = [
                w for w in words
                if w not in LanguageDetector.COMMON_SHORT_RUSSIAN_WORDS
            ]

            if not filtered_words:
                return sum(len(w) for w in words) / len(words)

            if all(len(w) <= 4 for w in filtered_words):
                return sum(len(w) for w in filtered_words) / len(filtered_words)

            return sum(len(w) for w in filtered_words) / len(filtered_words)

        return sum(len(w) for w in words) / len(words)

    @staticmethod
    def _get_short_word_ratio(text: str) -> float:
        """Среднее количество слов меньше длинны = 3."""
        words = re.findall(r'[а-яА-ЯёЁa-zA-Z]+', text.lower())

        if not words:
            return 0.0

        if re.search(r'[а-яА-ЯёЁ]', text):
            filtered_words = [
                w for w in words
                if w not in LanguageDetector.COMMON_SHORT_RUSSIAN_WORDS
            ]

            if not filtered_words:
                return 1.0

            short_count = sum(1 for w in filtered_words if len(w) <= 3)
            return short_count / len(filtered_words)

        short_count = sum(1 for w in words if len(w) <= 3)
        return short_count / len(words)

    @staticmethod
    def merge_adjacent_identical(
            segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """Объединение соседних сегментов с идентичным текстом."""
        if not segments:
            return []

        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]

            if (seg.text == last.text and
                    0 <= seg.start_time - last.end_time <=
                    SegmentMerger.MERGE_GAP_THRESHOLD):

                last.end_time = seg.end_time
                logger.debug(
                    f"Объединение соседних сегментов: '{seg.text[:30]}...'"
                )
            else:
                merged.append(seg)

        return merged

    @staticmethod
    def remove_complete_duplicates(
            segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """Полностью удалить сегменты, содержащиеся в другом."""
        if len(segments) <= 1:
            return segments

        result = []
        used_indices = set()

        for i, seg_i in enumerate(segments):
            if i in used_indices:
                continue

            is_duplicate = False

            for j, seg_j in enumerate(segments):
                if i == j or j in used_indices:
                    continue

                if (seg_j.start_time <= seg_i.start_time and
                        seg_i.end_time <= seg_j.end_time and
                        SegmentMerger.text_similarity(seg_i.text, seg_j.text) > 0.8):
                    is_duplicate = True
                    logger.debug(
                        f"Удаление дубликатов: '{seg_i.text[:30]}...'"
                    )
                    used_indices.add(i)
                    break

            if not is_duplicate:
                result.append(seg_i)

        return result

    @staticmethod
    def text_similarity(text1: str, text2: str) -> float:
        """Простое сходство текста, основанное на совпадающих словах."""
        if text1 == text2:
            return 1.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 1.0 if text1 == text2 else 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

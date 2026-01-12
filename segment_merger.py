import re
from typing import List, Tuple

import logging

from transcription_segment import TranscriptionSegment

from language_detector import LanguageDetector

logger = logging.getLogger(__name__)

# Helper functions
def get_avg_word_length(text: str) -> float:
    """
    Calculate average word length for RU text, EXCLUDING common Russian service words.
    This helps detect transliteration like "кам кейн" which only has short non-service words.

    For EN text, returns normal average (no exclusions).
    """
    words = re.findall(r'[а-яА-ЯёЁa-zA-Z]+', text.lower())
    if not words:
        return 0.0

    # Filter out common Russian words (only for Russian context)
    filtered_words = [w for w in words if w not in LanguageDetector.COMMON_SHORT_RUSSIAN_WORDS]

    if not filtered_words:
        # All words are common - shouldn't happen, but return original average
        len_more_4 = [w for w in words if len(w) > 4]
        if not len_more_4:
            return 10.0
        return sum(len(w) for w in len_more_4) / len(len_more_4)

    return sum(len(w) for w in filtered_words) / len(filtered_words)


def get_short_word_ratio(text: str) -> float:
    """
    Get ratio of words with <=3 chars, EXCLUDING common Russian service words.
    """
    words = re.findall(r'[а-яА-ЯёЁa-zA-Z]+', text.lower())
    if not words:
        return 0.0

    # Filter out common Russian words
    filtered_words = [w for w in words if w not in LanguageDetector.COMMON_SHORT_RUSSIAN_WORDS]

    if not filtered_words:
        # All words are common - they're all short, so ratio = 1.0
        return 1.0

    short = sum(1 for w in filtered_words if len(w) <= 3)
    return short / len(filtered_words)

class SegmentMerger:

    """Merges segments from RU and EN models with intelligent deduplication."""

    OVERLAP_THRESHOLD = 0.5  # 50% overlap = same segment
    MERGE_GAP_THRESHOLD = 0.3  # Gap < 0.3s between segments = merge

    @staticmethod
    def merge_segments(segments_ru: List[TranscriptionSegment],
                       segments_en: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Merge segments from two models, selecting best for overlapping intervals.

        Algorithm:
        1. Find overlapping pairs (RU segment vs EN segments)
        2. For each pair with overlap > 50%, select best
        3. Keep non-overlapping segments from both models
        4. Deduplicate and merge adjacent identical segments
        """
        if not segments_ru and not segments_en:
            return []

        logger.debug(f"Merging {len(segments_ru)} RU + {len(segments_en)} EN segments")

        # Step 1: Merge overlapping pairs
        merged = SegmentMerger._merge_overlapping_pairs(segments_ru, segments_en)
        logger.debug(f"After merging overlaps: {len(merged)} segments")

        # Step 2: Merge adjacent identical segments
        merged = SegmentMerger._merge_adjacent_identical(merged)
        logger.debug(f"After merging adjacent identical: {len(merged)} segments")

        # Step 3: Remove complete duplicates (one segment completely contained in another)
        merged = SegmentMerger._remove_complete_duplicates(merged)
        logger.debug(f"After removing duplicates: {len(merged)} segments")

        # Step 4: Sort by start time
        merged.sort(key=lambda s: s.start_time)

        return merged

    @staticmethod
    def _merge_overlapping_pairs(segments_ru: List[TranscriptionSegment],
                                 segments_en: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        For each RU segment, find overlapping EN segment and select best.
        Add non-overlapping EN segments.
        """
        result = []
        used_en_indices = set()

        # Process each RU segment
        for ru_seg in segments_ru:
            # Find best overlapping EN segment
            best_en_idx = -1
            best_overlap = 0

            for en_idx, en_seg in enumerate(segments_en):
                overlap = SegmentMerger._calculate_overlap(ru_seg, en_seg)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_en_idx = en_idx

            # If good overlap found, select best; otherwise keep RU
            if best_overlap > SegmentMerger.OVERLAP_THRESHOLD:
                en_seg = segments_en[best_en_idx]
                best = SegmentMerger._select_best_segment(ru_seg, en_seg)
                logger.debug(f"Best segment: [{best.start_time:.2f} - {best.end_time:.2f}] {best.text}")
                result.append(best)
                used_en_indices.add(best_en_idx)
            else:
                result.append(ru_seg)

        # Add non-overlapping EN segments
        for en_idx, en_seg in enumerate(segments_en):
            if en_idx not in used_en_indices:
                lang_en = LanguageDetector.detect_language(en_seg.text)
                quality_en = LanguageDetector.assess_quality(en_seg.text)
                is_transliteration_en = LanguageDetector.is_transliteration(en_seg.text)

                if lang_en == "en" and not is_transliteration_en and quality_en >= 0.5:
                    result.append(en_seg)  # ✓ Только хорошие EN

        return result

    @staticmethod
    def _calculate_overlap(seg1: TranscriptionSegment,
                          seg2: TranscriptionSegment) -> float:
        """
        Calculate overlap ratio: intersection / min(duration1, duration2)
        Returns 0.0 - 1.0 (0 = no overlap, 1 = complete containment)
        """
        # Calculate intersection
        intersection_start = max(seg1.start_time, seg2.start_time)
        intersection_end = min(seg1.end_time, seg2.end_time)

        if intersection_end <= intersection_start:
            return 0.0  # No overlap

        intersection = intersection_end - intersection_start

        # Calculate overlap ratio relative to shorter segment
        min_duration = min(
            seg1.end_time - seg1.start_time,
            seg2.end_time - seg2.start_time
        )

        if min_duration == 0:
            return 0.0

        return intersection / min_duration

    @staticmethod
    def _select_best_segment(seg_ru: TranscriptionSegment,
                            seg_en: TranscriptionSegment) -> TranscriptionSegment:
        """
        Select best segment based on:
        1. Language match (Cyrillic for RU, Latin for EN)
        2. Transliteration detection (penalty)
        3. Quality assessment with STRONG RU BIAS for pure Russian audio

        КРИТИЧЕСКИ ВАЖНО: Если это русское аудио (lang_ru == 'ru' и транслит = False),
        то ВСЕГДА выбираем RU, даже если EN имеет немного лучшее качество.
        Потому что EN мусор все равно имеет высокий quality score благодаря количеству слов.
        """
        lang_ru = LanguageDetector.detect_language(seg_ru.text)
        lang_en = LanguageDetector.detect_language(seg_en.text)
        quality_ru = LanguageDetector.assess_quality(seg_ru.text)
        quality_en = LanguageDetector.assess_quality(seg_en.text)
        is_transliteration_ru = LanguageDetector.is_transliteration(seg_ru.text)
        is_transliteration_en = LanguageDetector.is_transliteration(seg_en.text)


        # Calculate word length metrics
        ru_avg_len = get_avg_word_length(seg_ru.text)
        en_avg_len = get_avg_word_length(seg_en.text)
        ru_short_ratio = get_short_word_ratio(seg_ru.text)
        en_short_ratio = get_short_word_ratio(seg_en.text)

        logger.debug(
            f"Comparing: RU(lang={lang_ru}, q={quality_ru:.2f}, trans={is_transliteration_ru}, text={seg_ru.text}) "
            f"vs EN(lang={lang_en}, q={quality_en:.2f}, trans={is_transliteration_en}, text={seg_en.text})"
        )

        # ========================================================================
        # CASE 1: RU has Cyrillic (real Russian) and is NOT transliteration
        # ========================================================================
        # КРИТИЧЕСКИ ВАЖНО: Это чистый русский текст!
        # Выбираем RU ВСЕГДА, даже если EN имеет чуть лучше качество.
        # Потому что EN мусор получает высокий quality благодаря количеству слов.
        if lang_ru == "ru" and not is_transliteration_ru:
            # EXTRA CHECK: Is it hidden transliteration?
            # All words very short (avg < 3.5) + many short words (>50%)
            # AND EN has much longer words (avg > 4.5)
            if ru_avg_len < 4.5:
                logger.debug(
                    f"→ HIDDEN TRANSLITERATION DETECTED "
                    f"(RU: avg={ru_avg_len:.2f}, short={ru_short_ratio * 100:.0f}% vs "
                    f"EN: avg={en_avg_len:.2f}, short={en_short_ratio * 100:.0f}%) "
                    f"→ selecting EN"
                )
                return seg_en

            logger.debug(f"→ PURE RUSSIAN (lang_ru='ru', no_trans) → selecting RU")
            return seg_ru

        # ========================================================================
        # CASE 2: RU has Cyrillic, EN has Latin (ideal case for mixed audio)
        # ========================================================================
        if lang_ru == "ru" and lang_en == "en":
            if is_transliteration_ru:
                # RU is transliteration, prefer EN
                logger.debug(f"→ RU is transliteration → selecting EN")
                return seg_en

            # Both clean, prefer RU with strong bias
            # For mixed audio: we still prefer RU because it matched Russian model
            logger.debug(f"→ Both clean (RU={lang_ru}, EN={lang_en}) → RU bias → selecting RU")
            return seg_ru

        # ========================================================================
        # CASE 3: RU has Cyrillic but EN is mixed/unclear
        # ========================================================================
        elif lang_ru == "ru":
            if is_transliteration_ru:
                # RU is transliteration, prefer EN
                logger.debug(f"→ RU is transliteration, EN mixed → selecting EN")
                return seg_en

            # RU is clean Russian, prefer RU
            logger.debug(f"→ RU is clean Russian, EN mixed → selecting RU")
            return seg_ru

        # ========================================================================
        # CASE 4: EN has pure Latin, RU has no Cyrillic
        # ========================================================================
        elif lang_en == "en":
            # EN is clean English, prefer EN
            logger.debug(f"→ EN is clean English, RU no cyrillic → selecting EN")
            return seg_en

        # ========================================================================
        # CASE 5: Both mixed or unknown - use quality with penalty for transliteration
        # ========================================================================
        else:
            if is_transliteration_ru and not is_transliteration_en:
                logger.debug(f"→ RU is transliteration, EN clean → selecting EN")
                return seg_en

            # Choose by quality
            choice = seg_ru if quality_ru >= quality_en else seg_en
            logger.debug(f"→ Both mixed/unknown → choose by quality → selecting {'RU' if choice == seg_ru else 'EN'}")
            return choice

    @staticmethod
    def _merge_adjacent_identical(segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Merge adjacent segments with identical text if gap < 0.3s
        """
        if not segments:
            return []

        merged = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]

            # Check if text is identical and gap is small
            if (seg.text == last.text and
                0 <= seg.start_time - last.end_time < SegmentMerger.MERGE_GAP_THRESHOLD):
                # Merge by extending end time
                last.end_time = seg.end_time
                logger.debug(f"Merged adjacent identical: '{seg.text}'")
            else:
                merged.append(seg)

        return merged

    @staticmethod
    def _remove_complete_duplicates(segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Remove segments that are completely contained in another segment
        with identical or very similar text (fuzzy match).

        Example:
        - Segment A: 0.0-5.0 "hello world"
        - Segment B: 0.5-4.5 "hello world" (contained in A)
        → Remove B (keep A)
        """
        if len(segments) <= 1:
            return segments

        result = []
        used_indices = set()

        for i, seg_i in enumerate(segments):
            if i in used_indices:
                continue

            is_duplicate = False

            # Check if seg_i is contained in any other segment
            for j, seg_j in enumerate(segments):
                if i == j or j in used_indices:
                    continue

                # Check if seg_i is contained in seg_j with similar text
                if (seg_j.start_time <= seg_i.start_time and
                    seg_i.end_time <= seg_j.end_time and
                    SegmentMerger._text_similarity(seg_i.text, seg_j.text) > 0.8):
                    # seg_i is contained in seg_j, keep j (larger one)
                    is_duplicate = True
                    logger.debug(f"Removing duplicate (contained): '{seg_i.text}'")
                    break

            if not is_duplicate:
                result.append(seg_i)
            else:
                used_indices.add(i)

        return result

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        Simple text similarity: matching words / total words
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 1.0 if text1 == text2 else 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

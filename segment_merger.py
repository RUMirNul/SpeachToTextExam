from typing import List, Tuple
import logging
from transcription_segment import TranscriptionSegment
from language_detector import LanguageDetector

logger = logging.getLogger(__name__)


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
                result.append(best)
                used_en_indices.add(best_en_idx)
            else:
                result.append(ru_seg)

        # Add non-overlapping EN segments
        for en_idx, en_seg in enumerate(segments_en):
            if en_idx not in used_en_indices:
                result.append(en_seg)

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
        3. Quality assessment
        """
        lang_ru = LanguageDetector.detect_language(seg_ru.text)
        lang_en = LanguageDetector.detect_language(seg_en.text)
        quality_ru = LanguageDetector.assess_quality(seg_ru.text)
        quality_en = LanguageDetector.assess_quality(seg_en.text)
        is_transliteration_ru = LanguageDetector.is_transliteration(seg_ru.text)
        is_transliteration_en = LanguageDetector.is_transliteration(seg_en.text)

        logger.debug(
            f"Comparing: RU(lang={lang_ru}, q={quality_ru:.2f}, trans={is_transliteration_ru}) "
            f"vs EN(lang={lang_en}, q={quality_en:.2f}, trans={is_transliteration_en})"
        )

        # Case 1: RU has Cyrillic, EN has Latin (ideal case)
        if lang_ru == "ru" and lang_en == "en":
            if is_transliteration_ru:
                # RU is transliteration, prefer EN
                return seg_en
            # Both clean, prefer by quality with RU bias (+5%)
            return seg_ru if quality_ru * 1.05 >= quality_en else seg_en

        # Case 2: RU has Cyrillic, EN mixed or no Latin
        elif lang_ru == "ru":
            if is_transliteration_ru:
                return seg_en
            # RU is clean Russian, prefer RU
            return seg_ru

        # Case 3: EN has pure Latin, RU has no Cyrillic
        elif lang_en == "en":
            # EN is clean English, prefer EN
            return seg_en

        # Case 4: Both mixed or unknown - use quality
        else:
            if is_transliteration_ru and not is_transliteration_en:
                return seg_en
            # Choose by quality
            return seg_ru if quality_ru >= quality_en else seg_en

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

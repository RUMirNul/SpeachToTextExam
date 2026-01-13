#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

from audio_extractor import AudioExtractor
from speech_recognition_service import SpeechRecognitionService
from segment_merger import SegmentMerger
from subtitle_generator import SubtitleGenerator

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# БОЛЬШИЕ МОДЕЛИ (рекомендуется для лучшего качества)
# Скачать с: https://alphacephei.com/vosk/models
#MODEL_RU = "models/vosk-model-ru-0.42"  # Большая русская (1.8Gb, WER 12%)
#MODEL_EN = "models/vosk-model-en-us-0.22"  # Большая английская (1.5Gb, WER 11%)

MODEL_RU = "models/vosk-model-small-ru-0.22"
MODEL_EN = "models/vosk-model-small-en-us-0.15"


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        logger.info("Не задан файл")
        raise Exception("Не задан файл")

    if not input_file.exists():
        logger.error(f"Файл не найден: {input_file}")
        sys.exit(1)

    logger.info(f"Начался процесс с файлом: {input_file}")

    # Extract audio
    extractor = AudioExtractor()
    audio_file = extractor.extract_audio(input_file)

    try:
        # Recognize with both models
        logger.info("Распознавание с моделью Русского языка...")
        service_ru = SpeechRecognitionService(MODEL_RU)
        segments_ru = service_ru.recognize_speech(audio_file)
        service_ru.close()
        logger.info(f"Русский: {len(segments_ru)} сегмент")

        logger.info("Распознавание с моделью Английского языка...")
        service_en = SpeechRecognitionService(MODEL_EN)
        segments_en = service_en.recognize_speech(audio_file)
        service_en.close()
        logger.info(f"Английский: {len(segments_en)} сегмент")

        # Merge and generate
        merger = SegmentMerger()
        final_segments = merger.merge_segments(segments_ru, segments_en)
        logger.info(f"Объединённые сегменты: {len(final_segments)} segments")

        # Output
        output_path = Path("output") / f"{input_file.stem}.srt"
        generator = SubtitleGenerator()
        generator.generate_srt(final_segments, output_path)

        logger.info(f"✓ Субтитры сохранены в: {output_path}")

    finally:
        # Cleanup
        try:
            audio_file.unlink()
            logger.info("Очистка временных файлов")
        except:
            pass


if __name__ == "__main__":
    main()
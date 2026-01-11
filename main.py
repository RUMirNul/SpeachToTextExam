#!/usr/bin/env python3
import sys
import logging
from pathlib import Path
import tempfile

from audio_extractor import AudioExtractor
from speech_recognition_service import SpeechRecognitionService
from language_detector import LanguageDetector
from segment_merger import SegmentMerger
from subtitle_generator import SubtitleGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_RU = "models/vosk-model-small-ru-0.22"
MODEL_EN = "models/vosk-model-small-en-us-0.15"


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        logger.info("No input file specified, using default")
        input_file = Path("test_video.mp4")

    if not input_file.exists():
        logger.error(f"File not found: {input_file}")
        sys.exit(1)

    logger.info(f"Processing: {input_file}")

    # Extract audio
    extractor = AudioExtractor()
    audio_file = extractor.extract_audio(input_file)

    try:
        # Recognize with both models
        logger.info("Recognizing with Russian model...")
        service_ru = SpeechRecognitionService(MODEL_RU)
        segments_ru = service_ru.recognize_speech(audio_file)
        service_ru.close()
        logger.info(f"Russian: {len(segments_ru)} segments")

        logger.info("Recognizing with English model...")
        service_en = SpeechRecognitionService(MODEL_EN)
        segments_en = service_en.recognize_speech(audio_file)
        service_en.close()
        logger.info(f"English: {len(segments_en)} segments")

        # Merge and generate
        merger = SegmentMerger()
        final_segments = merger.merge_segments(segments_ru, segments_en)
        logger.info(f"Merged: {len(final_segments)} segments")

        # Output
        output_path = Path("output") / f"{input_file.stem}.srt"
        generator = SubtitleGenerator()
        generator.generate_srt(final_segments, output_path)

        logger.info(f"✓ Subtitles saved to: {output_path}")

    finally:
        # Cleanup
        try:
            audio_file.unlink()
            logger.info("Cleaned up temporary audio file")
        except:
            pass


if __name__ == "__main__":
    main()
import json
import wave
from pathlib import Path
from typing import List
import vosk
import logging

from transcription_segment import TranscriptionSegment

logger = logging.getLogger(__name__)


class SpeechRecognitionService:
    """Распознавание голоса используя Vosk и создание сегментов по паузам."""

    SAMPLE_RATE = 16000
    BUFFER_SIZE = 4000
    PAUSE_THRESHOLD = 0.5
    MAX_SEGMENT_LENGTH = 10.0

    def __init__(self, model_path: str):
        """Загрузка Vosk модели."""
        logger.info(f"Загрузка моделей из {model_path}")
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(model_path)

    def recognize_speech(self, audio_file: Path) -> List[TranscriptionSegment]:
        """Распознание речи и деление на сегменты по паузам."""
        logger.info(f"Распознание речи из{audio_file}")

        recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
        recognizer.SetWords(True)

        words_info = []

        with wave.open(str(audio_file), "rb") as wav_file:
            while True:
                data = wav_file.readframes(self.BUFFER_SIZE)
                if len(data) == 0:
                    break

                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    words_info.extend(self._extract_words(result))


        final_result = recognizer.FinalResult()
        words_info.extend(self._extract_words(final_result))


        segments = self._create_segments_by_pauses(words_info)
        logger.info(f"Создано {len(segments)} сегментов")

        return segments

    def _extract_words(self, json_result: str) -> list:
        """Получение информации о словах из JSON."""
        try:
            result = json.loads(json_result)
            words = result.get("result", [])
            return words
        except:
            return []

    def _create_segments_by_pauses(self, words: list) -> List[TranscriptionSegment]:
        """Группировка слов из сегментов с паузой больше 0.5."""
        if not words:
            return []

        segments = []
        current_segment = []

        for word in words:
            word_text = word.get("word", "")
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)

            if current_segment:
                pause = word_start - current_segment[-1].get("end", 0)
                if pause > self.PAUSE_THRESHOLD:
                    # Create segment from accumulated words
                    segment_text = " ".join([w.get("word", "") for w in current_segment])
                    segment_start = current_segment[0].get("start", 0)
                    segment_end = current_segment[-1].get("end", 0)

                    if segment_text.strip():
                        segments.append(TranscriptionSegment(segment_start, segment_end, segment_text))

                    current_segment = []

            current_segment.append(word)


        if current_segment:
            segment_text = " ".join([w.get("word", "") for w in current_segment])
            segment_start = current_segment[0].get("start", 0)
            segment_end = current_segment[-1].get("end", 0)

            if segment_text.strip():
                segments.append(TranscriptionSegment(segment_start, segment_end, segment_text))

        return segments

    def close(self):
        """Free resources."""
        pass
import json
import wave
from pathlib import Path
from typing import List
import vosk
import logging

from transcription_segment import TranscriptionSegment

logger = logging.getLogger(__name__)


class SpeechRecognitionService:
    """Recognizes speech using Vosk and creates segments by pauses."""

    SAMPLE_RATE = 16000
    BUFFER_SIZE = 4000
    PAUSE_THRESHOLD = 0.5
    MAX_SEGMENT_LENGTH = 10.0

    def __init__(self, model_path: str):
        """Load Vosk model from path."""
        logger.info(f"Loading model from {model_path}")
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(model_path)

    def recognize_speech(self, audio_file: Path) -> List[TranscriptionSegment]:
        """Recognize speech and create segments by pauses."""
        logger.info(f"Recognizing speech from {audio_file}")

        recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
        recognizer.SetWords(True)

        words_info = []

        # Read WAV file and process chunks
        with wave.open(str(audio_file), "rb") as wav_file:
            while True:
                data = wav_file.readframes(self.BUFFER_SIZE)
                if len(data) == 0:
                    break

                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    words_info.extend(self._extract_words(result))

        # Get final result
        final_result = recognizer.FinalResult()
        words_info.extend(self._extract_words(final_result))

        # Create segments based on pauses
        segments = self._create_segments_by_pauses(words_info)
        logger.info(f"Created {len(segments)} segments")

        return segments

    def _extract_words(self, json_result: str) -> list:
        """Extract word info from JSON result."""
        try:
            result = json.loads(json_result)
            words = result.get("result", [])
            return words
        except:
            return []

    def _create_segments_by_pauses(self, words: list) -> List[TranscriptionSegment]:
        """Group words into segments by pauses > 0.5 seconds."""
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

        # Add final segment
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
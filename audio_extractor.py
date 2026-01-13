import tempfile
from pathlib import Path
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Получение аудио дорожки и конвертация в WAV MONO 16kHz."""

    SAMPLE_RATE = 16000
    CHANNELS = 1

    @staticmethod
    def extract_audio(input_file: Path) -> Path:
        """Получение аудио дорожки и конвертация в WAV MONO 16kHz."""
        logger.info(f"Получение аудио дорожки из: {input_file}")

        # Create temporary file
        temp_wav = Path(tempfile.gettempdir()) / "audio_temp.wav"

        try:
            # Load audio from any format
            audio = AudioSegment.from_file(str(input_file))

            # Convert to mono and set frame rate
            audio = audio.set_channels(AudioExtractor.CHANNELS)
            audio = audio.set_frame_rate(AudioExtractor.SAMPLE_RATE)

            # Export to WAV
            audio.export(str(temp_wav), format="wav")
            logger.info(f"Аудио дорожка получена в {temp_wav}")

            return temp_wav
        except Exception as e:
            logger.error(f"Ошибка при получении аудио дорожки: {e}")
            raise
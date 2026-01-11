import tempfile
from pathlib import Path
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extracts and converts audio to WAV 16kHz mono."""

    SAMPLE_RATE = 16000
    CHANNELS = 1

    @staticmethod
    def extract_audio(input_file: Path) -> Path:
        """Extract audio from video/audio and convert to WAV 16kHz mono."""
        logger.info(f"Extracting audio from {input_file}")

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
            logger.info(f"Audio extracted to {temp_wav}")

            return temp_wav
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
from pathlib import Path
from typing import List
import logging
from transcription_segment import TranscriptionSegment

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """Generates SRT subtitle files."""

    @staticmethod
    def generate_srt(segments: List[TranscriptionSegment], output_path: Path):
        """Generate SRT file from segments."""
        logger.info(f"Generating SRT: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for index, seg in enumerate(segments, 1):
                start_time = SubtitleGenerator._format_time(seg.start_time)
                end_time = SubtitleGenerator._format_time(seg.end_time)

                f.write(f"{index}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{seg.text}\n")
                f.write("\n")

        logger.info(f"SRT file created: {output_path}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Convert seconds to SRT time format HH:MM:SS,mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
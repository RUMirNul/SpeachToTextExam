class TranscriptionSegment:
    """Выделение одного сегмента распознанной речи с помощью временных кодов."""

    def __init__(self, start_time: float, end_time: float, text: str):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

    def __str__(self) -> str:
        return f"[{self.start_time:.2f} - {self.end_time:.2f}] {self.text}"

    def __repr__(self) -> str:
        return f"TranscriptionSegment({self.start_time}, {self.end_time}, '{self.text}')"
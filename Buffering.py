"""
Buffering.py
------------
Thread-safe transcript accumulator.
Segments are stored with timestamps for RAG and memory use.
"""

import re
import time
import threading

from transcriber import transcribe_audio
from utils import logger


class TranscriptionBuffer:

    def __init__(self):
        self._segments: list[dict] = []
        self._lock = threading.Lock()

    # ── Public ────────────────────────────────────────────────────

    def add_chunk(self, wav_path: str) -> None:
        """Transcribe WAV and store segment. Called from recorder thread."""
        text = transcribe_audio(wav_path)
        if not text:
            return

        cleaned = self._clean(text)
        segment = {"text": cleaned, "timestamp": time.time()}

        with self._lock:
            self._segments.append(segment)
            logger.info("Buffer", f"Segment #{len(self._segments)} | {len(cleaned)} chars")

    def get_full_transcript(self) -> str:
        with self._lock:
            return " ".join(s["text"] for s in self._segments)

    def get_structured_transcript(self) -> list[dict]:
        with self._lock:
            return list(self._segments)

    def get_segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[.*?\]", "", text)
        return text.strip()
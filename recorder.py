"""
recorder.py
-----------
Real-time microphone recorder using sounddevice.
Emits fixed-length WAV chunks to a transcription callback.
"""

import tempfile
import wave
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from config import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_RECORD_SECONDS
from utils import logger


class AudioRecorder:

    def __init__(self, on_chunk_ready: Callable[[str], None]):
        self.on_chunk_ready = on_chunk_ready
        self._is_running    = False

    def start(self) -> None:
        self._is_running = True
        logger.success("Recorder", "Microphone started.")
        threading.Thread(target=self._record_loop, daemon=True).start()

    def stop(self) -> None:
        self._is_running = False
        logger.info("Recorder", "Recorder stopped.")

    def _record_loop(self) -> None:
        chunk_samples = AUDIO_SAMPLE_RATE * AUDIO_RECORD_SECONDS
        try:
            with sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype="int16",
            ) as stream:
                logger.info("Recorder", "Audio stream opened.")
                while self._is_running:
                    audio_chunk, _ = stream.read(chunk_samples)
                    audio_chunk    = np.array(audio_chunk, dtype=np.int16)
                    wav_path       = self._save_wav(audio_chunk)
                    if wav_path:
                        self.on_chunk_ready(wav_path)
        except Exception as e:
            logger.error("Recorder", f"Stream error: {e}")

    def _save_wav(self, audio: np.ndarray) -> str:
        try:
            tmp  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            path = tmp.name
            tmp.close()
            with wave.open(path, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(audio.tobytes())
            return path
        except Exception as e:
            logger.error("Recorder", f"WAV save failed: {e}")
            return ""
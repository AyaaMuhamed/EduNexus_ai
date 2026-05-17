"""
transcriber.py
--------------
Batch transcription using OpenAI Whisper (whisper-1).
Handles temp file cleanup and API retry.
"""

import os
import time
from openai import OpenAI
from config import OPENAI_API_KEY, WHISPER_MODEL
from utils import logger

_client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio(wav_path: str) -> str:
    if not wav_path or not os.path.exists(wav_path):
        logger.warning("Transcriber", f"File not found: {wav_path}")
        return ""

    try:
        for attempt in range(3):
            try:
                logger.info("Transcriber", f"Transcribing (attempt {attempt+1})...")
                with open(wav_path, "rb") as f:
                    response = _client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=f,
                        response_format="text",
                    )
                text = (response if isinstance(response, str)
                        else getattr(response, "text", "")).strip()

                if text:
                    logger.success("Transcriber", f"{len(text)} chars transcribed.")
                else:
                    logger.warning("Transcriber", "Empty transcription.")
                return text

            except Exception as e:
                logger.error("Transcriber", f"Attempt {attempt+1} failed: {e}")
                time.sleep(2)

        return ""

    finally:
        _delete(wav_path)


def _delete(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass
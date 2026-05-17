"""
core/topic_tracker.py
---------------------
Detects topic transitions in an ongoing lecture using embedding drift.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  New transcript chunk arrives                                │
  │       │                                                      │
  │  embed chunk  ──► compare to last topic centroid             │
  │       │                                                      │
  │  similarity < TOPIC_SIMILARITY_THRESHOLD  ──► new topic      │
  │       │                                                      │
  │  Update timeline:                                            │
  │    {"time": "08:32", "topic": "...", "chunks": [...]}        │
  └──────────────────────────────────────────────────────────────┘

Features:
  ✔ Embedding-drift topic detection (Feature 4)
  ✔ Structured timeline data with summaries (Feature 5)
  ✔ Configurable similarity threshold (Feature 8)
  ✔ Detailed logging (Feature 9)
"""

import re
import time
import threading
from datetime import datetime

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY, GPT_MODEL, TOPIC_SIMILARITY_THRESHOLD
from utils import logger
from utils.RAG_Matcher import embed

_client = OpenAI(api_key=OPENAI_API_KEY)


class TopicTracker:
    """
    Tracks lecture topic transitions using cosine similarity drift
    between consecutive chunk embeddings.

    Timeline structure:
        [
          {
            "index":   0,
            "time":    "09:01",
            "topic":   "Introduction to Neural Networks",
            "chunks":  ["...", "..."],
            "summary": "...",
            "score":   0.82,
          },
          ...
        ]
    """

    def __init__(self):
        self._timeline:       list[dict]   = []
        self._current_chunks: list[str]    = []
        self._last_embedding: np.ndarray | None = None
        self._session_start:  float        = time.time()
        self._lock            = threading.Lock()

    # ── Public ────────────────────────────────────────────────────

    def ingest(self, chunk: str) -> bool:
        """
        Process a new transcript chunk.

        Returns:
            True if a topic transition was detected, False otherwise.
        """
        if not chunk.strip():
            return False

        try:
            chunk_emb = embed([chunk])
        except Exception as e:
            logger.error("TopicTracker", f"Embedding failed: {e}")
            return False

        with self._lock:
            if self._last_embedding is None:
                # First chunk — start first topic
                self._start_new_topic(chunk, chunk_emb)
                return False

            sim = float(cosine_similarity(chunk_emb, self._last_embedding)[0][0])

            if sim < TOPIC_SIMILARITY_THRESHOLD:
                logger.info(
                    "TopicTracker",
                    f"Topic shift detected (sim={sim:.2f} < {TOPIC_SIMILARITY_THRESHOLD})"
                )
                self._finalize_current_topic()
                self._start_new_topic(chunk, chunk_emb)
                return True

            # Same topic — accumulate
            self._current_chunks.append(chunk)
            self._last_embedding = chunk_emb
            return False

    def get_timeline(self) -> list[dict]:
        """Return a copy of the full lecture timeline."""
        with self._lock:
            return list(self._timeline)

    def get_current_topic(self) -> str:
        """Return the inferred label of the current topic segment."""
        with self._lock:
            if self._timeline:
                return self._timeline[-1].get("topic", "Unknown")
            return "Unknown"

    def get_timeline_summary(self) -> str:
        """Human-readable timeline string for logging / MCQ context."""
        with self._lock:
            if not self._timeline:
                return "No topic transitions detected yet."
            lines = [
                f"  [{t['time']}] Topic {t['index']+1}: {t['topic']}"
                for t in self._timeline
            ]
            return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────

    def _elapsed_time(self) -> str:
        """Return HH:MM timestamp relative to session start."""
        elapsed = int(time.time() - self._session_start)
        h, m    = divmod(elapsed // 60, 60)
        return f"{h:02d}:{m:02d}"

    def _start_new_topic(self, chunk: str, embedding: np.ndarray) -> None:
        """Initialize a new topic segment."""
        self._current_chunks  = [chunk]
        self._last_embedding  = embedding

        topic_entry = {
            "index":   len(self._timeline),
            "time":    self._elapsed_time(),
            "topic":   self._infer_topic_label(chunk),
            "chunks":  self._current_chunks,
            "summary": "",
            "score":   0.0,
        }
        self._timeline.append(topic_entry)

        logger.success(
            "TopicTracker",
            f"New topic #{topic_entry['index']+1} @ {topic_entry['time']}: "
            f"{topic_entry['topic']}"
        )

    def _finalize_current_topic(self) -> None:
        """
        Summarize the current topic's chunks and store final data
        back into the last timeline entry.
        """
        if not self._timeline or not self._current_chunks:
            return

        last = self._timeline[-1]
        last["chunks"]  = list(self._current_chunks)
        last["summary"] = self._summarize_topic(self._current_chunks)

        logger.info(
            "TopicTracker",
            f"Finalized topic #{last['index']+1}: {len(last['chunks'])} chunks"
        )

    def _infer_topic_label(self, chunk: str) -> str:
        """
        Ask GPT to produce a short topic label (3–6 words) for a chunk.
        Falls back to first sentence on failure.
        """
        try:
            response = _client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a lecture topic labeler. "
                            "Given a lecture excerpt, respond with ONLY a short topic label "
                            "(3-6 words, no punctuation)."
                        ),
                    },
                    {"role": "user", "content": chunk[:400]},
                ],
                temperature=0.2,
                max_tokens=20,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # Fallback: first sentence
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            return sentences[0][:60] if sentences else "Unknown Topic"

    def _summarize_topic(self, chunks: list[str]) -> str:
        """Produce a 1-2 sentence summary of a completed topic segment."""
        text = " ".join(chunks)[:1200]
        try:
            response = _client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this lecture segment in 1-2 sentences.",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=80,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return text[:200]
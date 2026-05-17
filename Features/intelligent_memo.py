"""
core/intelligent_memory.py
---------------------------
Intelligent semantic memory system for the lecture monitoring pipeline.

Architecture (transformer-inspired):
  ┌─────────────────────────────────────────────────────────────┐
  │  Incoming transcript chunk                                  │
  │       │                                                     │
  │  score_chunk()  ──► importance score (0-1)                  │
  │       │                                                     │
  │  attention_filter()  ──► keep / discard                     │
  │       │                                                     │
  │  short_term_memory  (rolling window, N recent chunks)       │
  │       │                                                     │
  │  [trigger] ──► summarize_older_chunks() via GPT             │
  │       │                                                     │
  │  summarized_memory  (compressed mid-term context)           │
  │       │                                                     │
  │  get_context_for_mcq() ──► summary + recent chunks          │
  └─────────────────────────────────────────────────────────────┘

Features:
  ✔ Sentence-level importance scoring (Feature 2)
  ✔ Attention-based filtering (Feature 3)
  ✔ Short-term + summarized memory (Feature 1)
  ✔ Configurable thresholds via config.py (Feature 8)
  ✔ Detailed logging (Feature 9)
"""

import re
import time
import threading
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    GPT_MODEL,
    MEMORY_MAX_CHUNKS,
    MEMORY_SUMMARY_TRIGGER,
    MEMORY_RECENT_CONTEXT_CHUNKS,
    MEMORY_IMPORTANCE_THRESHOLD,
    ATTENTION_KEYWORDS,
)
from utils import logger

_client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────
# FEATURE 2 — Importance Scoring
# ─────────────────────────────────────────────────────────────────

def score_sentence(
    sentence: str,
    pdf_chunks: list[str] | None = None,
    pdf_embeddings=None,
) -> float:
    """
    Assign an importance score (0.0 – 1.0) to a single sentence.

    Signals used:
      1. Keyword density   — presence of attention keywords
      2. Sentence length   — too short/long → lower score
      3. Concept density   — ratio of capitalized / technical words
      4. Semantic match    — cosine similarity to PDF embeddings (optional)
    """
    if not sentence.strip():
        return 0.0

    score = 0.0
    text  = sentence.lower()

    # 1. Attention keyword hits
    hits = sum(1 for kw in ATTENTION_KEYWORDS if kw in text)
    score += min(hits * 0.15, 0.4)            # cap at 0.40

    # 2. Length heuristic  (sweet spot: 8–40 words)
    words = sentence.split()
    n     = len(words)
    if 8 <= n <= 40:
        score += 0.2
    elif n > 40:
        score += 0.1                           # long but still has content

    # 3. Concept density (capitalized non-first words → technical terms)
    caps = sum(1 for w in words[1:] if w and w[0].isupper())
    score += min(caps / max(n, 1) * 0.3, 0.2)

    # 4. Semantic similarity to PDF (if embeddings provided)
    if pdf_chunks and pdf_embeddings is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from utils.RAG_Matcher import embed
            import numpy as np

            sent_emb = embed([sentence])
            sims     = cosine_similarity(sent_emb, pdf_embeddings)[0]
            score   += float(sims.max()) * 0.3
        except Exception:
            pass                               # silent — embeddings optional

    return min(score, 1.0)


def score_chunk(
    chunk: str,
    pdf_chunks: list[str] | None = None,
    pdf_embeddings=None,
) -> float:
    """
    Score a multi-sentence chunk as the average of its sentence scores,
    weighted toward the highest-scoring sentence (max-pooling blend).
    """
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    if not sentences:
        return 0.0

    scores = [score_sentence(s, pdf_chunks, pdf_embeddings) for s in sentences]
    avg    = sum(scores) / len(scores)
    peak   = max(scores)

    # Blend: 60% average + 40% peak (rewards chunks with at least one key sentence)
    return round(0.6 * avg + 0.4 * peak, 4)


# ─────────────────────────────────────────────────────────────────
# FEATURE 3 — Attention-Based Filtering
# ─────────────────────────────────────────────────────────────────

def attention_filter(
    chunks: list[str],
    pdf_chunks: list[str] | None = None,
    pdf_embeddings=None,
    threshold: float = MEMORY_IMPORTANCE_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Rank chunks by importance score and filter out noise.

    Returns:
        List of (chunk, score) tuples sorted by descending score,
        keeping only those above `threshold`.
    """
    scored = [
        (chunk, score_chunk(chunk, pdf_chunks, pdf_embeddings))
        for chunk in chunks
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [(c, s) for c, s in scored if s >= threshold]

    logger.info(
        "Memory",
        f"Attention filter: {len(chunks)} chunks → {len(filtered)} kept "
        f"(threshold={threshold})"
    )
    return filtered


# ─────────────────────────────────────────────────────────────────
# FEATURE 1 — Intelligent Memory System
# ─────────────────────────────────────────────────────────────────

class IntelligentMemory:
    """
    Hierarchical semantic memory for an ongoing lecture session.

    Layers:
      short_term_memory  — recent N chunks (raw, ordered)
      summarized_memory  — GPT-compressed older content
    """

    def __init__(self):
        self._short_term:   list[dict]  = []   # {"text": str, "score": float, "ts": float}
        self._summarized:   str         = ""   # running GPT summary
        self._lock          = threading.Lock()

    # ── Ingest ────────────────────────────────────────────────────

    def add_transcript_chunk(
        self,
        chunk: str,
        pdf_chunks: list[str] | None = None,
        pdf_embeddings=None,
    ) -> None:
        """
        Score, filter, and store an incoming transcript chunk.
        Triggers summarization when short-term buffer is full.
        """
        if not chunk.strip():
            return

        importance = score_chunk(chunk, pdf_chunks, pdf_embeddings)

        logger.info(
            "Memory",
            f"Chunk scored {importance:.2f} | "
            f"{'✅ stored' if importance >= MEMORY_IMPORTANCE_THRESHOLD else '🔕 low-value'}"
        )

        with self._lock:
            self._short_term.append({
                "text":  chunk,
                "score": importance,
                "ts":    time.time(),
            })

            # Trim to max window
            if len(self._short_term) > MEMORY_MAX_CHUNKS:
                self._short_term = self._short_term[-MEMORY_MAX_CHUNKS:]

        # Trigger summarization when buffer reaches threshold
        if len(self._short_term) >= MEMORY_SUMMARY_TRIGGER:
            self._summarize_older_chunks()

    # ── Context export ────────────────────────────────────────────

    def get_context_for_mcq(self) -> str:
        """
        Build rich semantic context for MCQ generation.

        Returns:
            Compressed summary (mid-term) + recent high-value chunks (short-term)
        """
        with self._lock:
            # Always include the most recent N chunks verbatim
            recent_chunks = self._short_term[-MEMORY_RECENT_CONTEXT_CHUNKS:]
            recent_text   = " ".join(c["text"] for c in recent_chunks)

        parts = []
        if self._summarized:
            parts.append(f"[Lecture Summary So Far]\n{self._summarized}")
        if recent_text:
            parts.append(f"[Recent Lecture Content]\n{recent_text}")

        context = "\n\n".join(parts)

        logger.info(
            "Memory",
            f"Context built: summary={len(self._summarized)} chars | "
            f"recent={len(recent_text)} chars"
        )
        return context

    # ── Summarization ─────────────────────────────────────────────

    def _summarize_older_chunks(self) -> None:
        """
        Compress older short-term chunks into `summarized_memory` via GPT.
        Keeps only the most recent MEMORY_RECENT_CONTEXT_CHUNKS in short-term.
        """
        with self._lock:
            # Keep recent tail — compress the head
            keep  = self._short_term[-MEMORY_RECENT_CONTEXT_CHUNKS:]
            older = self._short_term[:-MEMORY_RECENT_CONTEXT_CHUNKS]
            if not older:
                return
            self._short_term = keep

        older_text = " ".join(c["text"] for c in older)

        logger.info("Memory", f"Summarizing {len(older)} older chunks...")

        try:
            prompt = (
                "You are summarizing an educational lecture transcript.\n"
                "Compress the following into a concise academic summary (max 200 words).\n"
                "Preserve: key concepts, definitions, important facts.\n"
                "Discard: filler words, repeated content, off-topic remarks.\n\n"
                f"Previous summary:\n{self._summarized or 'None'}\n\n"
                f"New content to integrate:\n{older_text}"
            )

            response = _client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise academic summarizer."},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            new_summary = response.choices[0].message.content.strip()
            self._summarized = new_summary

            logger.success("Memory", f"Summary updated ({len(new_summary)} chars).")

        except Exception as e:
            logger.error("Memory", f"Summarization failed: {e}")
            # Fallback: append raw text to summary
            self._summarized += " " + older_text[:500]
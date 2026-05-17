"""
utils/RAG_Matcher.py
--------------------
Semantic RAG pipeline for real-time lecture grounding.

Features:
  ✔ Precompute PDF embeddings once at session start
  ✔ LRU query embedding cache (max 100 entries)
  ✔ Paragraph chunking + sentence-window fallback
  ✔ Score normalization + low-similarity fallback
  ✔ Embedding API retry (3 attempts)
"""

import re
import time
import hashlib
from collections import OrderedDict

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY
from utils import logger

client = OpenAI(api_key=OPENAI_API_KEY)

# ── LRU query cache ───────────────────────────────────────────────
_query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
MAX_CACHE_SIZE = 100


# ── Embedding API ─────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts with retry logic."""
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return np.array([r.embedding for r in response.data])
        except Exception as e:
            logger.warning("RAG", f"Embed attempt {attempt+1} failed: {e}")
            time.sleep(1)
    raise RuntimeError("Embedding API failed after 3 retries")


def _embed_query_cached(query: str) -> np.ndarray:
    """Embed a single query string with LRU caching."""
    key = hashlib.md5(query.encode()).hexdigest()
    if key in _query_cache:
        _query_cache.move_to_end(key)
        return _query_cache[key]
    if len(_query_cache) >= MAX_CACHE_SIZE:
        _query_cache.popitem(last=False)
    result = embed([query])
    _query_cache[key] = result
    return result


# ── Chunking helpers ──────────────────────────────────────────────

def _merge_short_chunks(chunks: list[str], min_len: int = 80) -> list[str]:
    merged, buffer = [], ""
    for chunk in chunks:
        buffer = (buffer + " " + chunk).strip()
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def _chunk_sentences_with_overlap(
    sentences: list[str],
    window: int = 4,
    overlap: int = 1,
) -> list[str]:
    chunks = []
    step   = max(1, window - overlap)
    for i in range(0, len(sentences), step):
        text = " ".join(sentences[i : i + window]).strip()
        if len(text) > 80:
            chunks.append(text)
    return chunks


def chunk_pdf(pdf_text: str) -> list[str]:
    """Convert raw PDF text into semantic chunks."""
    raw    = re.split(r"\n{2,}", pdf_text)
    chunks = _merge_short_chunks([c.strip() for c in raw if c.strip()])

    if len(chunks) < 3:
        sentences = re.split(r"(?<=[.!?])\s+", pdf_text)
        chunks    = _chunk_sentences_with_overlap(sentences)

    if not chunks:
        chunks = [pdf_text[:1500]]

    return chunks


# ── Step 1: Prepare (run once) ────────────────────────────────────

def prepare_pdf_embeddings(pdf_text: str) -> tuple[list[str], np.ndarray]:
    """
    Chunk PDF and embed all chunks.
    Call once at session start; results are passed everywhere.
    """
    chunks     = chunk_pdf(pdf_text)
    embeddings = embed(chunks)
    logger.success("RAG", f"Indexed {len(chunks)} PDF chunks.")
    return chunks, embeddings


# ── Step 2: Retrieve (real-time) ──────────────────────────────────

def find_relevant_pdf_chunks(
    transcript: str,
    pdf_chunks: list[str],
    pdf_embeddings: np.ndarray,
    top_k: int = 3,
) -> str:
    """
    Find the most semantically relevant PDF chunks for the current
    lecture transcript window.
    """
    query = transcript[-800:].strip()

    if not query or len(query.split()) < 5:
        logger.warning("RAG", "Query too short — using first chunk as fallback.")
        return pdf_chunks[0]

    query_emb = _embed_query_cached(query)
    scores    = cosine_similarity(query_emb, pdf_embeddings)[0]

    # Normalize
    if scores.max() != scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # Low-similarity fallback
    if scores.max() < 0.2:
        logger.warning("RAG", "Low similarity — using distributed fallback chunks.")
        n       = len(pdf_chunks)
        indices = sorted({0, n // 2, n - 1})
        return "\n\n".join(pdf_chunks[i] for i in indices)

    top_k    = min(top_k, len(pdf_chunks))
    top_idx  = np.argsort(scores)[-top_k:][::-1]
    result   = "\n\n".join(pdf_chunks[i] for i in top_idx)

    logger.info("RAG", f"Retrieved {top_k} chunks (top score={scores.max():.2f})")
    return result
"""
utils/embedding_cache.py
------------------------
Disk-based embedding cache using MD5 hash of PDF content.
Avoids re-embedding the same PDF on every session start.
"""

import os
import hashlib
import numpy as np
from utils import logger

CACHE_DIR = "./cache"


def load_or_create_embeddings(
    pdf_text: str,
    embed_fn,
    chunk_fn,
) -> tuple[list[str], np.ndarray]:
    """
    Load cached embeddings if available, otherwise compute and cache them.

    Args:
        pdf_text:  Raw text extracted from PDF.
        embed_fn:  Function(list[str]) -> np.ndarray
        chunk_fn:  Function(str) -> list[str]

    Returns:
        (chunks, embeddings)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    pdf_hash     = hashlib.md5(pdf_text[:5000].encode()).hexdigest()
    chunks_path  = os.path.join(CACHE_DIR, f"{pdf_hash}_chunks.npy")
    embed_path   = os.path.join(CACHE_DIR, f"{pdf_hash}_embeddings.npy")

    if os.path.exists(chunks_path) and os.path.exists(embed_path):
        logger.success("EmbedCache", "Cache hit — loading from disk.")
        chunks     = np.load(chunks_path, allow_pickle=True).tolist()
        embeddings = np.load(embed_path)
        return chunks, embeddings

    logger.info("EmbedCache", "Cache miss — computing embeddings...")
    chunks     = chunk_fn(pdf_text)
    embeddings = embed_fn(chunks)

    np.save(chunks_path, np.array(chunks, dtype=object))
    np.save(embed_path,  embeddings)

    logger.success("EmbedCache", f"Cached {len(chunks)} chunks to disk.")
    return chunks, embeddings
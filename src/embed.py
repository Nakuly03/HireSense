"""
Embedding helpers using sentence-transformers (local, no API cost).
Model: all-MiniLM-L6-v2  →  384-dimensional cosine embeddings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the model once and reuse across the process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str) -> List[float]:
    """Return a single embedding vector for *text*."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embedding vectors for a list of texts (batched)."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]

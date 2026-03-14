"""
Pinecone client helpers.
Index dimension is read from config (384 for all-MiniLM-L6-v2).
"""
from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from src.config import (
    EMBEDDING_DIM,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX,
    PINECONE_REGION,
)

_pc: Pinecone | None = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def ensure_index(
    name: str = PINECONE_INDEX,
    dimension: int = EMBEDDING_DIM,
    metric: str = "cosine",
) -> None:
    """Create the Pinecone index if it does not already exist."""
    pc = _get_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print(f"Created Pinecone index '{name}' (dim={dimension}, metric={metric})")
    else:
        print(f"Pinecone index '{name}' already exists.")


def get_index(name: str = PINECONE_INDEX):
    """Return a handle to the named Pinecone index."""
    return _get_client().Index(name)

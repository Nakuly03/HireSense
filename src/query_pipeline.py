"""
Full RAG pipeline with LangSmith observability:
  1. Classify query → category  (Llama-3.1-8B-Instruct)
  2. Embed query               (sentence-transformers, local)
  3. Metadata-filter + search  (Pinecone)
  4. Generate answer           (Llama-3.1-8B-Instruct)
"""
from __future__ import annotations

import time
from typing import Any

from src.config import PINECONE_INDEX
from src.embed import embed_text
from src.llm import generate_answer_with_trace
from src.pinecone_client import get_index
from src.tools import classify_category


def retrieve(user_query: str, top_k: int = 5) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []

    t0 = time.time()
    category = classify_category(user_query)
    trace.append({
        "step":       "category_classification",
        "model":      "meta-llama/Llama-3.1-8B-Instruct (HuggingFace)",
        "result":     category,
        "duration_s": round(time.time() - t0, 2),
    })


    t0 = time.time()
    vector = embed_text(user_query)
    trace.append({
        "step":       "query_embedding",
        "model":      "sentence-transformers/all-MiniLM-L6-v2 (local)",
        "dim":        len(vector),
        "duration_s": round(time.time() - t0, 2),
    })


    t0 = time.time()
    index = get_index(PINECONE_INDEX)
    results = index.query(
        vector=vector,
        top_k=top_k,
        filter={"category": category},
        include_metadata=True,
    )
    docs = [
        {
            "id":    m["id"],
            "score": round(m["score"], 4),
            "text":  m.get("metadata", {}).get("text", ""),
        }
        for m in results.get("matches", [])
    ]
    trace.append({
        "step":       "pinecone_search",
        "index":      PINECONE_INDEX,
        "filter":     {"category": category},
        "top_k":      top_k,
        "returned":   len(docs),
        "doc_ids":    [d["id"] for d in docs],
        "duration_s": round(time.time() - t0, 2),
    })

    return {"category": category, "docs": docs, "trace": trace}


def answer(user_query: str, retrieved: dict[str, Any]) -> dict[str, Any]:
    result = generate_answer_with_trace(user_query, retrieved["docs"])
    return {"answer": result["answer"], "trace": result["trace"]}

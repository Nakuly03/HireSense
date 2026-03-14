"""
LLM answer generation using HuggingFace Inference API (Llama-3.1-8B-Instruct).
All calls are wrapped with langsmith.traceable so every token, latency,
and prompt is logged to LangSmith when LANGSMITH_API_KEY is set in .env.
"""
from __future__ import annotations

import time
from typing import Any

from huggingface_hub import InferenceClient

from src.config import HF_API_TOKEN, HF_LLM_MODEL


try:
    from langsmith import traceable
    _HAS_LANGSMITH = True
except ImportError:

    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    _HAS_LANGSMITH = False

_client = InferenceClient(token=HF_API_TOKEN)

_SYSTEM_PROMPT = """You are a helpful HR assistant that answers questions about resumes.
You are given a user question and the top matching resume excerpts retrieved from a database.
Provide a concise, accurate, and helpful answer based solely on the provided documents.
Reference specific resume IDs (e.g. row_12) when relevant.
Do not invent information not present in the documents."""


def _build_context(docs: list[dict[str, Any]]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        doc_id = doc.get("id", f"doc_{i}")
        score  = doc.get("score", 0.0)
        text   = doc.get("text", "")[:800]
        parts.append(f"[{doc_id}] (score={score:.3f})\n{text}")
    return "\n\n---\n\n".join(parts)


@traceable(run_type="llm", name="hiresense-answer-generation")
def generate_answer(user_query: str, docs: list[dict[str, Any]]) -> str:
    """Generate a readable answer from docs for user_query.
    Decorated with @traceable so LangSmith logs prompt, response, and latency.
    """
    context = _build_context(docs)
    user_message = (
        f"Question: {user_query}\n\n"
        f"Top matching resumes:\n\n{context}\n\n"
        "Answer:"
    )
    response = _client.chat_completion(
        model=HF_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_answer_with_trace(
    user_query: str,
    docs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Same as generate_answer but also returns a trace dict for the Streamlit UI."""
    t0 = time.time()
    ans = generate_answer(user_query, docs)
    elapsed = round(time.time() - t0, 2)
    return {
        "answer": ans,
        "trace": {
            "step":       "answer_generation",
            "model":      HF_LLM_MODEL,
            "provider":   "HuggingFace Inference API",
            "langsmith":  _HAS_LANGSMITH,
            "doc_ids":    [d.get("id") for d in docs],
            "duration_s": elapsed,
        },
    }

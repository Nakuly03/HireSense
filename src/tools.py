"""
Category classifier using HuggingFace Inference API (Mistral-7B-Instruct).
Returns exactly one category from the fixed enum, validated with Pydantic.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal

from huggingface_hub import InferenceClient
from pydantic import BaseModel, field_validator

from src.config import CATEGORIES, HF_API_TOKEN, HF_LLM_MODEL

# ── Dynamic Enum from config ──────────────────────────────────────────────────
CategoryEnum = Enum("CategoryEnum", {c: c for c in CATEGORIES})  # type: ignore


# ── Pydantic response model ───────────────────────────────────────────────────
class CategoryResponse(BaseModel):
    category: str

    @field_validator("category")
    @classmethod
    def must_be_valid(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in CATEGORIES:
            raise ValueError(f"'{v}' is not a valid category. Must be one of: {CATEGORIES}")
        return upper


# ── HuggingFace client ────────────────────────────────────────────────────────
_client = InferenceClient(token=HF_API_TOKEN)

_SYSTEM_PROMPT = f"""You are a resume-category classifier.
Given a user query about resumes or job seekers, return EXACTLY one JSON object:
{{"category": "<CATEGORY>"}}

Valid categories (use exactly as written):
{', '.join(CATEGORIES)}

Do NOT add any explanation, markdown, or extra text — only the JSON object."""


def classify_category(user_query: str) -> str:
    """
    Classify *user_query* into one of the fixed resume categories.
    Returns the category string (upper-cased, validated).
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    response = _client.chat_completion(
        model=HF_LLM_MODEL,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
    )

    raw: str = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract a category word directly
        for cat in CATEGORIES:
            if cat.lower() in raw.lower():
                return cat
        raise ValueError(f"Could not parse category from model response: {raw!r}")

    validated = CategoryResponse(**data)
    return validated.category

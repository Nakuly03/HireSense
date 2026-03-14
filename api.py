"""
HireSense REST API — FastAPI layer over the RAG pipeline.

Endpoints:
  POST /search          — retrieve top-k matching resumes
  POST /ask             — retrieve + generate an answer
  GET  /health          — liveness check
  GET  /categories      — list all supported resume categories

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import API_HOST, API_PORT, CATEGORIES
from src.query_pipeline import answer, retrieve

app = FastAPI(
    title="HireSense API",
    description=(
        "Agentic resume search engine powered by Llama-3.1-8B-Instruct "
        "and Pinecone vector search."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, example="Senior Python engineer with cloud experience")
    top_k: int = Field(5, ge=1, le=20, description="Number of resumes to return")


class ResumeDoc(BaseModel):
    id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    tool_used: str
    reasoning: str
    docs: list[ResumeDoc]
    trace: list[dict[str, Any]]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, example="Find me a finance professional with CFA certification")
    top_k: int = Field(5, ge=1, le=20)


class AskResponse(BaseModel):
    query: str
    tool_used: str
    reasoning: str
    answer: str
    docs: list[ResumeDoc]
    trace: list[dict[str, Any]]



@app.get("/health", tags=["Meta"])
def health() -> dict[str, str]:
    """Liveness check — returns 200 if the service is up."""
    return {"status": "ok", "service": "HireSense API v2"}


@app.get("/categories", tags=["Meta"])
def list_categories() -> dict[str, list[str]]:
    """Return the full list of supported resume categories."""
    return {"categories": CATEGORIES}


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search(req: SearchRequest) -> SearchResponse:
    """
    Retrieve the top-k most relevant resumes for a query.
    The agent decides the retrieval strategy (category filter, keyword filter,
    multi-category, or broad search).
    """
    try:
        result = retrieve(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=req.query,
        tool_used=result["plan"]["tool"],
        reasoning=result["plan"]["reasoning"],
        docs=[ResumeDoc(**d) for d in result["docs"]],
        trace=result["trace"],
    )


@app.post("/ask", response_model=AskResponse, tags=["Search"])
def ask(req: AskRequest) -> AskResponse:
    """
    Retrieve resumes AND generate a plain-English answer to the query.
    Combines /search with an LLM answer generation step.
    """
    try:
        retrieved = retrieve(req.query, top_k=req.top_k)
        ans       = answer(req.query, retrieved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        query=req.query,
        tool_used=retrieved["plan"]["tool"],
        reasoning=retrieved["plan"]["reasoning"],
        answer=ans["answer"],
        docs=[ResumeDoc(**d) for d in retrieved["docs"]],
        trace=retrieved["trace"] + [ans["trace"]],
    )



if __name__ == "__main__":
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)

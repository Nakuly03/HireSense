# 🔍 HireSense v2 — Agentic Resume Search Engine

> A production-grade RAG system with an **agentic query planner**, **FastAPI REST layer**, **LangSmith observability**, and **RAGAS evaluation** — deployable to HuggingFace Spaces in one command.

[![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)](http://localhost:8000/docs)
[![LangSmith](https://img.shields.io/badge/LangSmith-traced-brightgreen?logo=langchain)](https://smith.langchain.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-evaluated-blue)](https://docs.ragas.io)

---

## What's new in v2

| Feature | v1 | v2 |
|---------|----|----|
| Query planning | Hardcoded category classifier | **Agentic planner** (4 tools, LLM decides strategy) |
| API | Streamlit only | **FastAPI** `/search` + `/ask` endpoints |
| Observability | UI trace only | **LangSmith** full prompt/response/latency tracing |
| Evaluation | None | **RAGAS** (faithfulness, relevancy, precision, recall) |
| Deployment | Local only | **HuggingFace Spaces** + Docker |

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent (Llama-3.1-8B-Instruct via HuggingFace API)      │
│  Reasons about the query and picks one of 4 tools:      │
│    • category_search   → single-category vector search  │
│    • multi_category    → 2-3 related categories         │
│    • keyword_filter    → category + skill keywords      │
│    • broad_search      → no filter, full index          │
└────────────────────────┬────────────────────────────────┘
                         │ tool + params
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Embedding  (sentence-transformers/all-MiniLM-L6-v2)    │
│  Local, free, 384-dim cosine                            │
└────────────────────────┬────────────────────────────────┘
                         │ vector
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Pinecone Vector Search                                 │
│  Metadata filter from agent plan · top-k results       │
└────────────────────────┬────────────────────────────────┘
                         │ docs
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Answer Generation  (Llama-3.1-8B-Instruct)            │
│  @traceable → LangSmith logs every call                 │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Streamlit UI           FastAPI REST
       (app.py)               (api.py)
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| **Agent / LLM** | `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| **Vector DB** | Pinecone (serverless) |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Observability** | LangSmith |
| **Evaluation** | RAGAS |
| **Deployment** | HuggingFace Spaces + Docker |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/HireSense
cd HireSense
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: HF_API_TOKEN, PINECONE_API_KEY
# Optional: LANGSMITH_API_KEY (for tracing)
```

Get keys:
- HuggingFace token → https://huggingface.co/settings/tokens *(request Llama access first)*
- Pinecone key → https://app.pinecone.io *(free tier is enough)*
- LangSmith key → https://smith.langchain.com *(free tier)*

### 3. Upsert resume data (one-time)

```bash
python scripts/upsert_resumes.py --csv Resume/Resume_cleaned.csv
```

### 4. Run Streamlit UI

```bash
streamlit run app.py
```

### 5. Run REST API

```bash
uvicorn api:app --reload
# Docs at: http://localhost:8000/docs
```

### 6. Run RAGAS evaluation

```bash
python scripts/evaluate.py
# Results saved to eval_results.json
```

---

## API Reference

### `POST /search`
Retrieve top-k resumes. The agent decides the retrieval strategy.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Senior Python engineer with AWS", "top_k": 5}'
```

```json
{
  "query": "Senior Python engineer with AWS",
  "tool_used": "keyword_filter",
  "reasoning": "Query contains specific technologies; using keyword filter on INFORMATION-TECHNOLOGY",
  "docs": [{"id": "row_42", "score": 0.891, "text": "..."}],
  "trace": [...]
}
```

### `POST /ask`
Retrieve resumes AND generate a plain-English answer.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Find a healthcare professional with ICU experience"}'
```

### `GET /health`
Liveness check.

### `GET /categories`
List all 24 supported resume categories.

Interactive docs at **http://localhost:8000/docs** (Swagger UI).

---

## Deploy to HuggingFace Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    SDK: Streamlit, Hardware: CPU Basic (free)

# 2. Add secrets in Space Settings:
#    HF_API_TOKEN, PINECONE_API_KEY, LANGSMITH_API_KEY (optional)

# 3. Push the repo
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/HireSense
git push space main
```

The `Dockerfile` is also included for container-based deployments (Railway, Render, etc.).

---

## Evaluation (RAGAS)

Run the evaluation suite to measure RAG quality:

```bash
python scripts/evaluate.py
```

Example output:
```
RAGAS Evaluation Results
────────────────────────────────────────
  faithfulness         0.87  █████████████████
  answer_relevancy     0.91  ██████████████████
  context_precision    0.78  ███████████████
  context_recall       0.83  ████████████████
```

Use `--questions my_questions.json` to evaluate against your own question set.

---

## Observability (LangSmith)

With `LANGSMITH_API_KEY` set, every `generate_answer()` call is automatically traced:
- Full prompt and response text
- Token usage and latency
- Run metadata and tags

View traces at https://smith.langchain.com → project `hiresense`.

---

## Repository Layout

```
HireSense/
├── app.py                     # Streamlit UI
├── api.py                     # FastAPI REST API
├── Dockerfile                 # Container deployment
├── requirements.txt
├── .env.example
├── scripts/
│   ├── upsert_resumes.py      # Embed + upsert CSV to Pinecone
│   └── evaluate.py            # RAGAS evaluation suite
├── src/
│   ├── config.py              # All config + env vars
│   ├── agent.py               # Agentic query planner (4 tools)
│   ├── embed.py               # sentence-transformers helpers
│   ├── pinecone_client.py     # Pinecone index management
│   ├── tools.py               # Legacy category classifier (kept for reference)
│   ├── llm.py                 # LLM answer generation + LangSmith @traceable
│   └── query_pipeline.py      # Full RAG orchestration
└── Resume/
    └── Resume_cleaned.csv
```

# 🔍 HireSense v2 — Agentic Resume Search Engine

> A production-grade RAG system that finds the best-matching resumes for any hiring query — powered by open-source LLMs, semantic vector search, and a clean REST API.

[![HuggingFace](https://img.shields.io/badge/LLM-Llama--3.1--8B-yellow?logo=huggingface)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](http://localhost:8000/docs)
[![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-brightgreen?logo=langchain)](https://smith.langchain.com)
[![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-blue)](https://pinecone.io)

---

## What is HireSense?

HireSense is a retrieval-augmented generation (RAG) system built for resume search. You type a natural language hiring query, and it:

1. Classifies the query into a job domain using an LLM
2. Converts the query into a semantic vector locally (no API cost)
3. Searches a Pinecone vector database filtered by that domain
4. Returns the top 5 matching resumes with an LLM-generated answer

Everything is observable — every LLM call is traced end-to-end via LangSmith.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Category Classifier                                    │
│  Llama-3.1-8B-Instruct via HuggingFace Inference API   │
│  → classifies query into one of 24 job categories      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Query Embedding                                        │
│  sentence-transformers/all-MiniLM-L6-v2 (local, free)  │
│  → 384-dimensional cosine vector                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Pinecone Vector Search                                 │
│  Metadata filter: { category: <predicted> }            │
│  → top-5 most similar resumes                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Answer Generation                                      │
│  Llama-3.1-8B-Instruct via HuggingFace Inference API   │
│  @traceable → every call logged to LangSmith           │
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
| **LLM** | `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` — local, free, no API cost |
| **Vector DB** | Pinecone serverless (cosine similarity, 384-dim) |
| **REST API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Observability** | LangSmith — full prompt, response, and latency tracing |

> No OpenAI key required. Only paid service is Pinecone (free tier is sufficient).

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Nakuly03/HireSense
cd HireSense
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in HF_API_TOKEN and PINECONE_API_KEY
# LANGSMITH_API_KEY is optional
```

| Key | Where to get it |
|-----|----------------|
| `HF_API_TOKEN` | https://huggingface.co/settings/tokens |
| `PINECONE_API_KEY` | https://app.pinecone.io |
| `LANGSMITH_API_KEY` | https://smith.langchain.com (optional) |

> **Note:** Llama-3.1-8B-Instruct is a gated model. Request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct before running.

### 3. Upsert resume data (one-time)

```bash
python scripts/upsert_resumes.py --csv Resume/Resume_cleaned.csv
```

This embeds all resumes locally and uploads them to Pinecone with category metadata.

### 4. Run the Streamlit UI

```bash
streamlit run app.py
```

### 5. Run the REST API

```bash
uvicorn api:app --reload
# Swagger docs at: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/categories` | List all 24 supported categories |
| `POST` | `/search` | Retrieve top-k matching resumes |
| `POST` | `/ask` | Retrieve resumes + generate answer |

### Example — `/ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Senior Python engineer with AWS experience", "top_k": 5}'
```

```json
{
  "query": "Senior Python engineer with AWS experience",
  "category": "INFORMATION-TECHNOLOGY",
  "answer": "Based on the retrieved resumes, row_142 is a strong match...",
  "docs": [
    {"id": "row_142", "score": 0.891, "text": "..."},
    {"id": "row_89",  "score": 0.873, "text": "..."}
  ]
}
```

Interactive docs available at **http://localhost:8000/docs**

---

## Supported Categories

```
HR · DESIGNER · INFORMATION-TECHNOLOGY · TEACHER · ADVOCATE
BUSINESS-DEVELOPMENT · HEALTHCARE · FITNESS · AGRICULTURE · BPO
SALES · CONSULTANT · DIGITAL-MEDIA · AUTOMOBILE · CHEF · FINANCE
APPAREL · ENGINEERING · ACCOUNTANT · CONSTRUCTION · PUBLIC-RELATIONS
BANKING · ARTS · AVIATION
```

---

## Repository Layout

```
HireSense/
├── app.py                     # Streamlit UI
├── api.py                     # FastAPI REST API
├── Dockerfile                 # Container deployment
├── requirements.txt
├── .env.example               # Environment variable template
├── scripts/
│   └── upsert_resumes.py      # Embed + upsert CSV to Pinecone
└── src/
    ├── __init__.py
    ├── config.py              # All config and env vars
    ├── embed.py               # sentence-transformers helpers
    ├── pinecone_client.py     # Pinecone index management
    ├── tools.py               # Category classifier
    ├── llm.py                 # Answer generation + LangSmith tracing
    └── query_pipeline.py      # Full RAG orchestration
```

---

## Observability

With `LANGSMITH_API_KEY` set in `.env`, every LLM call is automatically traced:
- Full prompt and response
- Token usage and latency
- Run metadata

View traces at https://smith.langchain.com → project `hiresense`.

---

## Deploy to HuggingFace Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    SDK: Streamlit | Hardware: CPU Basic (free)

# 2. Add secrets in Space Settings:
#    HF_API_TOKEN, PINECONE_API_KEY, LANGSMITH_API_KEY

# 3. Push
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/HireSense
git push space main
```

---

## About

Built as a portfolio project demonstrating production RAG patterns:
- Open-source LLMs (no OpenAI dependency)
- Local embeddings (zero embedding cost)
- Vector search with metadata filtering
- REST API layer on top of the pipeline
- Full LLM observability with LangSmith

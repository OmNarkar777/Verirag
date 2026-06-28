<div align="center">

# VeriRAG
### Production RAG Evaluation & Observability Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)](https://postgresql.org)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.1.9-orange)](https://docs.ragas.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Evaluate, version, and debug RAG pipeline quality with RAGAS metrics — with automatic regression detection when your pipeline changes.**

</div>

---

## The Problem This Solves

Running RAGAS once is easy. The hard part is knowing **when a pipeline change made quality worse** — did switching to a new chunking strategy hurt faithfulness? Did adding a reranker drop context recall?

VeriRAG answers this automatically. Every eval run is compared against the previous one. If any metric drops by more than 10 points, a regression is flagged instantly with a before/after breakdown shown on the dashboard.
---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         VeriRAG System                               │
│                                                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐  │
│  │   React Frontend        │    │   FastAPI Backend               │  │
│  │   (Vite + Tailwind)     │◄──►│   (async, Python 3.12)         │  │
│  │                         │    │                                 │  │
│  │  Dashboard              │    │  POST /eval/run ──► EvalService │  │
│  │  ├─ MetricCards         │    │       │              │          │  │
│  │  ├─ TrendChart          │    │   BackgroundTask  RagasRunner   │  │
│  │  ├─ RegressionAlert  ◄──┼────┼───── has_regression ◄──────────┤  │
│  │  └─ RunsTable           │    │                    │            │  │
│  │                         │    │  POST /pipeline/query           │  │
│  │  RunDetails             │    │       │                         │  │
│  │  ├─ MetricCards         │    │  RAGPipeline                    │  │
│  │  ├─ ScoreDistribution   │    │  ├─ ChromaDB (MMR retrieval)   │  │
│  │  └─ CaseTable (paged)   │    │  └─ Groq LLM (Llama 3.3 70B)  │  │
│  │                         │    │                                 │  │
│  │  Pipeline               │    │  RegressionService              │  │
│  │  ├─ IngestPanel         │    │  └─ detect_and_store()         │  │
│  │  └─ QueryPanel          │    │                                 │  │
│  └─────────────────────────┘    └──────────┬──────────────────────┘  │
│                                            │                         │
│         PostgreSQL 15                 ChromaDB               LangSmith│
│         ├─ eval_runs                  (vectors +              (traces) │
│         │  ├─ version_tag             MMR search)                     │
│         │  ├─ avg_* scores                                            │
│         │  ├─ has_regression                                          │
│         │  └─ regression_details                                      │
│         └─ eval_cases                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | Groq (Llama 3.3 70B) | Fast inference, free tier, strong reasoning for eval |
| Embeddings | all-MiniLM-L6-v2 (local) | No API cost, no rate limits, reproducible results |
| Vector Store | ChromaDB | In-process, MMR retrieval implemented from scratch |
| Evaluation | RAGAS 0.1.9 | Purpose-built for RAG, LLM-as-judge pattern |
| Database | PostgreSQL 15 + asyncpg | JSONB metadata, ARRAY contexts, fully async |
| Backend | FastAPI (async) | RAGAS runs 5+ min in thread pool, never blocks |
| Frontend | React 18 + Recharts | Live polling, score distribution histograms |
| Tracing | LangSmith | Full trace tree per query for debugging |
| Migrations | Alembic | Versioned schema evolution, not `create_all()` |

---

## RAGAS Metrics — What They Actually Measure

### Faithfulness (0–1)
**Does the answer contain ONLY information from the retrieved context?**

RAGAS decomposes the answer into atomic claims, then verifies each against the context. Score = `supported_claims / total_claims`. A score of 0.85 means 85% of claims were grounded — the other 15% were hallucinated. **Fix:** strengthen the system prompt to restrict the model to provided context only.

### Answer Relevancy (0–1)
**Does the answer address the question that was actually asked?**

RAGAS generates reverse questions from the answer and measures cosine similarity to the original question. Low score = off-topic answers. **Fix:** improve retrieval to surface more directly relevant chunks.

### Context Precision (0–1)
**Are useful chunks ranked at the top of the retrieved list?**

For each chunk, RAGAS asks: did this contribute to the answer? Then measures ranking quality. Low score = irrelevant chunks mixed with relevant ones. **Fix:** MMR retrieval (already implemented), reranker model.

### Context Recall (0–1)
**Does the retrieved context cover everything in the ground truth?**

RAGAS decomposes the ground truth and checks coverage. Low score = relevant info exists in corpus but wasn't retrieved. **Fix:** larger `top_k`, smaller chunk size, hybrid BM25 search.

---

## The Key Feature: Regression Detection

After every eval run completes, VeriRAG automatically compares scores against the most recent prior run. If any metric drops ≥ 10 points (configurable), the run is flagged:

```json
{
  "has_regression": true,
  "regression_details": {
    "faithfulness": {
      "previous": 0.87,
      "current": 0.71,
      "delta": -0.16,
      "is_regression": true
    }
  }
}
```

The dashboard shows a red banner immediately. Stored as JSONB in PostgreSQL with a partial index on `has_regression = true` for O(1) dashboard queries. This is what separates a production eval system from a one-off script.

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.12
- Node.js 20+
- [Groq API key](https://console.groq.com) — free tier

### 1. Clone and configure
```bash
git clone https://github.com/OmNarkar777/verirag
cd verirag
cp .env.example .env
# Edit .env — add GROQ_API_KEY and DATABASE_URL
```

### 2. Start infrastructure
```bash
docker-compose up -d postgres
```

### 3. Install and migrate
```bash
pip install -r requirements.txt
alembic upgrade head
```

### 4. Start the API
```bash
uvicorn backend.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Start the frontend
```bash
cd frontend
npm install
npm run dev
# Dashboard: http://localhost:3000
```

### 6. Run your first evaluation
```bash
# Ingest a document
curl -X POST http://localhost:8000/api/v1/pipeline/ingest/text \
  -F "text=Your document text here" \
  -F "filename=my-doc.txt"

# Start sample evaluation (10 QA pairs, takes ~5-10 min)
curl -X POST "http://localhost:8000/api/v1/eval/run/sample?version_tag=v1.0.0-baseline"
```

Watch the dashboard at `http://localhost:3000` — metric cards fill in automatically when the eval completes.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health — DB + ChromaDB status |
| POST | `/api/v1/eval/run` | Start RAGAS evaluation (202 + background task) |
| POST | `/api/v1/eval/run/sample` | Run built-in 10-case AI/ML sample |
| GET | `/api/v1/eval/runs` | List all runs with regression flags |
| GET | `/api/v1/eval/runs/{id}` | Full run with per-case scores |
| GET | `/api/v1/eval/runs/{id}/cases` | Paginated case scores + CSV export |
| **GET** | **`/api/v1/eval/regressions`** | **All runs where a metric regressed** |
| GET | `/api/v1/eval/status` | Concurrent eval count + rate limit info |
| POST | `/api/v1/pipeline/ingest` | Upload PDF or text file |
| POST | `/api/v1/pipeline/query` | Query RAG pipeline — answer + chunks |
| GET | `/api/v1/pipeline/stats` | ChromaDB collection statistics |

Full interactive docs: `http://localhost:8000/docs`

---

## Production Patterns

**Async Everything** — asyncpg + SQLAlchemy async sessions. RAGAS (synchronous) runs in `asyncio.run_in_executor` — never blocks the event loop during 5-minute evaluations. `expire_on_commit=False` prevents the classic async SQLAlchemy lazy-load gotcha.

**Concurrency Control** — `asyncio.Semaphore(5)` caps concurrent RAGAS runs. Each run makes ~200 Groq API calls; uncapped concurrency causes cascading rate-limit failures at Groq's free tier (30 req/min).

**Error Isolation** — RAGAS failures never crash the server. Every eval run has a `status` field (`pending → running → completed/failed`). Failed runs store the full error message in PostgreSQL — no phantom "running" ghosts.

**Version Tracking** — Semantic version tags (`v1.0.0-baseline`, `v1.1.0-mmr`, `v2.0.0-reranker`) on every eval run enable SQL queries like `SELECT version_tag, avg_faithfulness FROM eval_runs ORDER BY created_at DESC`.

**MMR Retrieval** — Maximal Marginal Relevance implemented from scratch (Carbonell & Goldstein 1998). Score = λ × relevance(query, doc) − (1−λ) × max_similarity(doc, selected). Improves `context_precision` by diversifying the retrieved set.

**Schema Evolution** — Two Alembic migrations with explicit PostgreSQL types (JSONB, ARRAY). `alembic downgrade -1` rolls back. Migration files are version-controlled diffs, not `create_all()`.

**Zero-Touch Deploy** — `alembic upgrade head` runs automatically in the FastAPI lifespan on every cold start. New schema changes land the instant a deployment goes live; no manual migration step.

---

## Project Structure

```
verirag/
├── backend/
│   ├── evaluator/
│   │   ├── ragas_runner.py       # RAGAS engine with LangchainLLMWrapper
│   │   ├── metrics.py            # Thresholds + explanations per metric
│   │   └── dataset_builder.py    # 10 curated AI/ML QA pairs
│   ├── rag/
│   │   ├── pipeline.py           # End-to-end: ingest → retrieve → generate
│   │   ├── vectorstore.py        # ChromaDB + MMR from scratch
│   │   └── retriever.py          # LangSmith-traced retrieval layer
│   ├── routers/
│   │   ├── eval.py               # Eval endpoints + semaphore rate limiting
│   │   └── pipeline.py           # Ingest + query endpoints
│   └── services/
│       ├── regression_service.py # Regression detection engine
│       ├── eval_service.py       # Business logic layer
│       └── langsmith_service.py  # Optional LangSmith integration
├── frontend/src/
│   ├── components/dashboard/     # MetricCard, TrendChart, RegressionAlert
│   ├── components/eval/          # RunDetails, CaseTable, ScoreDistribution
│   └── components/pipeline/      # IngestPanel (drag-drop), QueryPanel
├── alembic/versions/
│   ├── 001_initial_schema.py
│   └── 002_add_regression_fields.py
└── tests/
    ├── test_eval.py
    └── test_pipeline.py
```

---

## Deployment

### Vercel + Supabase (Production)

**1. Create a Supabase project** at [supabase.com](https://supabase.com) (free tier). Copy the **Transaction Pooler** URI from Settings → Database → URI.

**2. Deploy to Vercel:**
```bash
npx vercel --prod
```

**3. Set environment variables** in Vercel Dashboard → Project → Environment Variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | `postgresql+asyncpg://postgres.XXXX:PASS@aws-0-REGION.pooler.supabase.com:6543/postgres` |
| `GROQ_API_KEY` | ✅ | Free at [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | Optional | [HuggingFace token](https://huggingface.co/settings/tokens) — faster embedding rate limits |
| `LANGCHAIN_API_KEY` | Optional | For LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | Optional | `true` to enable tracing |

**4.** Alembic migrations run automatically on first cold start — no manual migration step.

**Note on ChromaDB:** Vercel serverless doesn't include the OpenMP runtime (`libgomp.so.1`) required by ChromaDB's HNSW index. The `/health` endpoint reports `chromadb: unavailable` on Vercel — all other features (evaluations, API docs, regression detection) work normally. Use the Docker Compose setup for full RAG pipeline functionality.

---

## License

MIT — see [LICENSE](LICENSE)

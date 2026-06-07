# Smart Comply — AML/CFT Screening API

> AI/ML Engineer Technical Assessment Submission  
> Smartcomply · Built with FastAPI · OpenAI · FAISS · Cohere

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Overview](#overview)
3. [Quick Start](#quick-start)
4. [Running with Docker](#running-with-docker)
5. [API Reference](#api-reference)
6. [Section-by-Section Notes](#section-by-section-notes)
   - [Section 1 — ML Fundamentals](#section-1--ml-fundamentals-written)
   - [Section 2a — Screening Pipeline](#section-2a--screening-pipeline)
   - [Section 2b — FastAPI Service](#section-2b--fastapi-service)
   - [Section 2c — RAG Pipeline](#section-2c--rag-pipeline)
   - [Section 3 — MLOps & Production](#section-3--mlops--production-written)
   - [Section 4 — System Design](#section-4--system-design-written)
7. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
8. [Environment Variables](#environment-variables)
9. [Assumptions](#assumptions)

---

## Project Structure

```
smart-comply/
│
├── answers/                        # Section 1, 3, 4 — written answers
│   ├── machine_learning.md         # Section 1 — ML Fundamentals
│   ├── mlops.md                    # Section 3 — MLOps & Production
│   ├── system_design.md            # Section 4 — System Design Case Study
│   └── production_break.md         # Production failure mode analysis (2c)
│
├── src/                            # Section 2 — all working code
│   ├── main.py                     # Entrypoint — starts uvicorn server
│   ├── Dockerfile                  # Container build for the FastAPI service
│   ├── requirements.txt            # Pinned dependencies
│   ├── .env                        # API keys (not committed — see below)
│   │
│   ├── data/                       # Synthetic datasets
│   │   ├── watchlist_data.py       # 25 sanctioned entity entries with aliases
│   │   ├── news.py                 # 18 news snippets (adverse + benign) + RAG corpus
│   │   └── corpus.py               # 30+ synthetic regulatory/news document chunks
│   │
│   └── package/                    # Core application modules
│       ├── __init__.py
│       ├── api.py                  # FastAPI app — /screen, /health, middleware
│       ├── pipeline.py             # Orchestrator — ties watchlist + adverse media together
│       ├── watchlist.py            # Fuzzy + phonetic + embedding watchlist matcher
│       ├── adverse_media.py        # LLM-based zero-shot adverse media classifier
│       └── rag.py                  # FAISS RAG pipeline with Cohere reranking
│
└── README.md                       # This file
```

---

## Overview

**Smart Comply** is a compliance screening service that combines three signals into a single entity risk result:

| Signal | Technique | Model / Library |
|---|---|---|
| Watchlist matching | Fuzzy + phonetic + embedding hybrid | RapidFuzz, Jellyfish, OpenAI `text-embedding-3-small` |
| Adverse media classification | Zero-shot LLM structured output | GPT-4o-mini via `client.responses.parse` |
| RAG risk summary | Dense retrieval + cross-encoder reranking | FAISS, OpenAI embeddings, Cohere Rerank |

The FastAPI layer wraps all three into a single `POST /screen` endpoint returning a structured JSON result per entity.

---

## Quick Start

### Prerequisites

- Python 3.11+
- An OpenAI API key
- A Cohere API key (for the RAG reranker in `rag.py`)

### 1. Clone the repository

```bash
git clone https://github.com/Darkdev007/smart-comply.git
cd smart-comply
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
cd src
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file inside `src/`:

```env
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

> **Note:** The `.env` file is listed in `.gitignore` and should never be committed.

### 5. Run the server

```bash
# From the src/ directory
python main.py
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Running with Docker

```bash
cd src

# Build
docker build -t smart-comply .

# Run
docker run --env-file .env -p 8000:8000 smart-comply
```

The container exposes port `8000` and starts the uvicorn server directly via the `CMD` in the Dockerfile.

---

## Testing Individual Modules

All commands below must be run from inside the `src/` directory with the virtual environment active.


### Watchlist matcher

Runs the hybrid fuzzy + phonetic + embedding scorer against five test queries and prints ranked results to the terminal.

```bash
# From src/
python -m package.watchlist
```

Expected output — top-3 matches per query with scores:

```
Query: 'Aleksandr Petrov'
---------------------------------------------
  #1  Alexander Petrov                     score=0.9142  [RU]
  #2  Alexei Petrov                        score=0.8731  [RU]
  #3  Oleksandr Kovalenko                  score=0.6210  [UA]
```

### Adverse media classifier

Runs the GPT-4o-mini zero-shot classifier across all 18 news snippets and prints each result with its adverse flag, confidence score, and rationale.

```bash
# From src/
python -m package.adverse_media
```

Expected output:

```
[s_001] ADVERSE ⚠️   conf=0.97  — Snippet references OFAC sanctions designation.
[s_002] benign  ✓    conf=0.95  — Routine earnings announcement, no compliance relevance.
```

### Full screening pipeline (without the API)

Runs the complete `screen_entity()` orchestrator directly and saves results to `screening_result.json`.

```bash
# From src/
python -m package.pipeline
```

### RAG pipeline

Builds (or loads) the FAISS index and demonstrates vanilla retrieval vs. Cohere reranked retrieval for a sample query.

```bash
# From src/
python -m package.rag
```

> **Note:** The first run calls the OpenAI embeddings API to build the index and saves `faiss.index` and `embeddings.npy` to disk. Subsequent runs load from disk and are much faster.

---

## API Reference

### `POST /screen`

Screens a named entity against the watchlist and adverse media corpus.

**Request body**

```json
{
  "query": "Aleksandr Petrov"
}
```

| Field | Type | Constraints |
|---|---|---|
| `query` | `string` | 2–200 characters, required |

**Response**

```json
{
  "query": "Aleksandr Petrov",
  "watchlist_hits": [
    {
      "watchlist_id": "W-007",
      "matched_name": "Alexander Petrov",
      "score": 0.9142,
      "country": "RU"
    },
    ...
  ],
  "adverse_media": [
    {
      "snippet_id": "s_001",
      "adverse": true,
      "rationale": "Snippet references sanctions designation by OFAC against named individual.",
      "confidence": 0.97
    },
    ...
  ]
}
```

---

### `GET /health`

Returns service liveness and model metadata.

```json
{
  "status": "ok",
  "model_version": "1.0.0",
  "embedding_model": "text-embedding-3-small"
}
```

---

## Section-by-Section Notes

### Section 1 — ML Fundamentals (written)

Full answers in [`answers/machine_learning.md`](answers/machine_learning.md).

Covers:
- **1a** — Bias-variance diagnosis of the overfitting scenario; three concrete interventions with trade-offs; metric prioritisation under class imbalance (precision-recall AUC over accuracy).
- **1b** — Dense vs. sparse representations; hybrid retrieval justification for transliterated name matching; embedding model selection and evaluation approach.
- **1c** — Isolation Forest vs. Autoencoder comparison for velocity anomaly detection; concept drift handling; evaluation strategy without reliable ground truth.
- **1d** — 8 derived features for PEP screening; missing value and high-cardinality handling; data leakage risks and prevention.

---

### Section 2a — Screening Pipeline

**Files:** `src/package/watchlist.py`, `src/package/adverse_media.py`, `src/data/watchlist_data.py`, `src/data/news.py`

#### Watchlist matching approach

Three signals are combined into a single confidence score:

| Signal | Weight | Rationale |
|---|---|---|
| RapidFuzz (token sort + partial + WRatio) | 45% | Handles spelling variants and word-order differences |
| Jellyfish Soundex phonetic match | 20% | Catches sound-alike transliterations (e.g. "Kaddafi" → "Gaddafi") |
| OpenAI `text-embedding-3-small` cosine similarity | 35% | Handles multilingual and script-transliterated variants |

Names are normalised before scoring (lowercased, hyphens/apostrophes removed, whitespace collapsed). The index is built once at import time and reused across requests; deduplication is done per `watchlist_id` so aliases don't produce redundant hits.

#### Adverse media classifier

A zero-shot GPT-4o-mini classifier receives each snippet with a structured system prompt defining what constitutes adverse media in the AML/CFT context. Responses are parsed directly into a Pydantic `AdverseMediaResult` schema using `client.responses.parse`, eliminating brittle JSON post-processing. Each result includes `adverse` (bool), `confidence` (float), and a `rationale` string.

The 18 synthetic snippets span: OFAC designations, criminal indictments, bribery allegations, asset freezes, regulatory enforcement actions, and genuinely benign content (sports results, earnings announcements, community events).

---

### Section 2b — FastAPI Service

**Files:** `src/package/api.py`, `src/main.py`, `src/Dockerfile`

- **`POST /screen`** — Pydantic-validated input (`min_length=2`, `max_length=200`), calls `screen_entity()` from the pipeline orchestrator.
- **`GET /health`** — Returns `status`, `model_version`, and `embedding_model`.
- **Latency middleware** — An `@app.middleware("http")` wrapper measures `time.perf_counter()` wall time per request and emits a structured log entry.
- **Structured JSON logging** — A custom `JSONFormatter` emits every log line as a JSON object with `timestamp`, `level`, `message`, and an `extra` dict containing request-specific fields (method, path, status, latency_ms, query, hit counts).
- **Lifespan events** — Startup and shutdown messages are emitted via the FastAPI `lifespan` async context manager.

The Dockerfile uses `python:3.11-slim`, installs only build-essential system dependencies, copies requirements first for layer caching, and starts the service with uvicorn directly.

---

### Section 2c — RAG Pipeline

**Files:** `src/package/rag.py`, `src/data/corpus.py`

**Corpus:** 30+ synthetic document chunks covering OFAC designations, FinCEN advisories, regulatory enforcement notices, and adverse news articles.

**Pipeline:**

```
Query
  └─► OpenAI text-embedding-3-small  →  FAISS IndexFlatL2  →  top-10 vanilla hits
                                                                       │
                                                              Cohere Rerank API
                                                              (cross-encoder)
                                                                       │
                                                              top-3 reranked chunks
                                                                       │
                                                          GPT-4o-mini risk synthesis
                                                                       │
                                                          Structured risk narrative
```

**Why reranking improves retrieval:**  
Vanilla cosine similarity on dense embeddings tends to surface semantically adjacent but topically loose matches — e.g. a chunk mentioning "financial sanctions" in a general context scores highly even if it has nothing to do with the queried entity. Cohere's cross-encoder reranker evaluates the (query, chunk) pair jointly rather than independently, surfacing chunks where the entity name and risk signal co-occur meaningfully.

**Production failure mode discussed:** Index staleness — the FAISS index is built once and cached to disk (`embeddings.npy`, `faiss.index`). If new sanctions designations are added, they will not appear in retrieval results until the index is explicitly rebuilt. Mitigation strategy: a scheduled nightly rebuild job triggered by a document store change event, plus a version hash check at startup that invalidates the cache if the corpus has changed.

---

### Section 3 — MLOps & Production (written)

Full answers in [`answers/mlops.md`](answers/mlops.md).

Covers:
- **3a** — Monitoring strategy: PSI for data drift, proxy metrics for delayed labels, alerting thresholds, Evidently AI + Grafana tooling.
- **3b** — Scaling: GPU batching for embedding inference, HNSW indexing for sub-200ms vector search, Redis Streams for async burst buffering, request lifecycle diagram.
- **3c** — CI/CD pipeline: DVC for dataset versioning, MLflow for artefact tracking, quality gates per stage, canary rollback criteria.

---

### Section 4 — System Design (written)

Full answer in [`answers/system_design.md`](answers/system_design.md).

End-to-end AML platform design covering: component breakdown (screening, behavioural analytics, narrative generation), data flow from raw transaction ingestion to the compliance dashboard, two architectural trade-offs (sync vs. async screening path; explainability depth vs. latency), false positive management strategy, and what would change with 10x engineering resources.

---

## Design Decisions & Trade-offs

**Hybrid scoring over pure embedding similarity**  
Pure cosine similarity on name embeddings struggles with very short strings (2–3 tokens) where small perturbations produce large distance shifts. Adding fuzzy and phonetic signals as a weighted ensemble makes the overall score more robust to the specific failure modes common in sanctions screening (transliteration, abbreviation, character substitution).

**Zero-shot LLM classifier over a trained model**  
Training a binary classifier requires labelled adverse media data, which is scarce and domain-specific. A zero-shot GPT-4o-mini classifier with a well-specified system prompt achieves high precision on the synthetic test set without a labelling pipeline, at the cost of slightly higher inference latency (~300–500ms per snippet) and OpenAI API dependency. For production at scale, this would be replaced with a fine-tuned smaller model or a distilled classifier.

**FAISS `IndexFlatL2` over HNSW for the RAG corpus**  
At 30–50 document chunks, an exact-search flat index is fast enough and avoids the approximation error of HNSW. For a production corpus of millions of documents, the index type would be upgraded to `IndexHNSWFlat` or a managed vector store (Pinecone, Qdrant) to maintain sub-200ms p99 latency.

**Module-level index building in `watchlist.py`**  
`build_watchlist_index()` runs at import time so the index is ready before the first request arrives. The trade-off is a cold-start penalty (~2–4 seconds for embedding API calls). In production this would be moved into the FastAPI `lifespan` startup hook with a readiness probe gate.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Used for embeddings (`text-embedding-3-small`) and LLM calls (GPT-4o-mini) |
| `COHERE_API_KEY` | Yes | Used for cross-encoder reranking in the RAG pipeline |

---

## Assumptions

- The watchlist and news corpus are synthetic but representative of real AML/CFT screening scenarios. No real PII or sanctioned entity data was used.
- The `adverse_media` result in `POST /screen` currently runs the classifier across the full static news corpus rather than fetching entity-specific news dynamically. In production, this step would be preceded by a targeted news retrieval step filtered on the queried entity name.
- The RAG pipeline (`rag.py`) is implemented as a standalone module demonstrating the retrieval and reranking logic. It is not wired into the `POST /screen` endpoint by default to avoid compounding API latency in the assessment environment; it can be invoked independently.
- All dependencies are pinned in `requirements.txt` against the versions used during development (Python 3.11).
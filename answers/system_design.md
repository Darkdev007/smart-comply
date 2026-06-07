# Section 4 — System Design Case Study

---

## End-to-End AML Screening Platform Design

### Overview

Every time an entity appears, a new customer, a transaction counterparty,  the platform must screen them against sanctions lists, check for adverse news, analyse their transaction behaviour, and hand a compliance officer a readable explanation of any flags, all within 3 seconds. That is a hard real-time constraint on a genuinely complex multi-step pipeline.

The design prioritises three things: the 3-second SLA on screening results, a 90-day behavioural window for pattern detection, and full auditability of every decision made.

---

### Component Breakdown

| Component | ML Model / Technique | Inputs | Outputs |
|---|---|---|---|
| Entity Screening Service | Distilled multilingual embeddings + BM25 + RRF fusion + FAISS | Entity name, country, DOB, aliases | Top-5 watchlist/PEP hits with calibrated confidence scores |
| Adverse Media Classifier | Zero-shot LLM classifier (GPT-4o mini or Mistral-7B) via RAG pipeline | Retrieved news chunks | Adverse flag, risk category, rationale string |
| Behavioural Analytics Engine | Isolation Forest + LSTM Autoencoder on 90-day rolling window | Transaction history, velocity features, graph centrality scores | Anomaly score (0–1), top contributing features, SHAP values |
| Transaction Network Graph | GraphSAGE + PageRank + community detection | sender_id, receiver_id, amount, timestamp edges | Node risk scores, community membership, suspicious cluster flags |
| Risk Narrative Generator | RAG pipeline with LLM; structured prompt with retrieved evidence | All upstream scores + evidence chunks | Auditable plain-English risk narrative per entity |
| Explainability Layer | SHAP for tabular models; attention weights + retrieved chunks for LLM | Model inputs + predictions | Feature importance rankings, evidence citations, decision trace |

---

### Component Detail

**Entity Screening Service**

This is a hybrid retrieval system consisting of two complementary approaches:

1. **BM25:** Keyword-level matching that is fast and exact. Catches straightforward name matches efficiently.
2. **Dense embeddings via FAISS:** Semantic and multilingual similarity. Catches transliteration variants, phonetic near-misses, and cross-script equivalents that BM25 completely misses.
3. **Reciprocal Rank Fusion (RRF):** Combines the ranked lists from both systems into a single ranking. An entity that appears in the top 5 of both lists receives a higher combined score than one that only appears in one. This is more robust than picking a winner between the two approaches.

Raw similarity scores are then passed through Platt scaling to calibrate them, converting raw cosine similarity into an interpretable probability.

---

**Adverse Media Classifier**

Runs via a RAG pipeline. Rather than feeding raw news texts directly to the LLM, a retrieval step first pulls the most relevant chunks from an indexed document store. The LLM then classifies only the retrieved evidence. The risk of hallucination is reduced because the model can only reference what was explicitly passed to it.

Output per snippet: an adverse flag, a risk category (e.g. sanctions evasion, bribery), and a one-sentence rationale tied to specific retrieved text.

---

**Behavioural Analytics Engine**

Two models working in combination:

- **Isolation Forest** for point anomalies: a single transaction or feature value that stands out from the norm.
- **LSTM Autoencoder** for sequential anomalies: a pattern across time that looks wrong even when individual events appear normal.

---

**Transaction Network Graph**

Individual entity models are blind to network structure. An entity can appear clean in isolation but sit at the centre of a suspicious transaction cluster , this gets missed if the network is not analysed.

GraphSAGE learns node embeddings that incorporate neighbourhood structure, so a node's risk score reflects not just its own transactions but the behaviour of its counterparties and their counterparties. PageRank identifies structurally important nodes. Community detection (Louvain algorithm) identifies clusters of entities that transact heavily with each other but minimally with the outside world.Ex: the signature of a shell company network.

---

**Risk Narrative Generator**

An LLM that receives a strictly structured prompt containing only facts retrieved from upstream components, watchlist hits, adverse media snippets, anomaly scores, SHAP values, and graph flags. The prompt includes a hard instruction to reference only the provided evidence. Outputs are validated against source facts before being surfaced. The narrative is designed to be readable in under 30 seconds by a compliance officer reviewing dozens of alerts per day.

---

### Data Flow — Raw Ingestion to Compliance Dashboard

```
Raw transaction ingestion
        │
        ▼
Kafka (transactions.raw)
        │
        ▼
Apache Flink (stream processor)
    — Validation and deduplication
    — Currency normalisation
    — Merchant category enrichment
    — Real-time velocity feature computation
        │
        ├──► PostgreSQL (transaction store, partitioned by month)
        │
        └──► Redis (90-day rolling feature windows, updated continuously)
                │
                ▼
        Behavioural Analytics Engine
        (nightly batch, scores cached in Redis)

═══════════════════════════════════════════
        ENTITY CHECK TRIGGERED (real-time)
═══════════════════════════════════════════
        │
        ▼
Screening API (FastAPI) — 3-second SLA starts here
        │
        ├──► [PARALLEL] Entity Screening Service
        │         FAISS vector search (~3ms)
        │         Elasticsearch BM25 search (~5ms)
        │         RRF fusion + Platt calibration (~2ms)
        │         → Top-5 calibrated watchlist hits
        │
        ├──► [PARALLEL] Behavioural Score
        │         Redis cache lookup (~5ms)
        │         → Anomaly score + SHAP values
        │
        ├──► [PARALLEL] Network Graph Score
        │         Neo4j/NetworkX node lookup (~10ms)
        │         → Node risk score + community flag
        │
        └──► [ASYNC] Adverse Media RAG Pipeline
                  Chunk retrieval from Qdrant
                  LLM classification
                  → adverse_flag + rationale
                  (result streamed via SSE when ready)
        │
        ▼
Risk Fusion Layer
    — Weighted combination of all scores
    — Confidence tier assignment (auto-escalate / review / auto-clear)
    — SHAP explanation assembly
        │
        ▼
Risk Narrative Generator (LLM, async via SSE)
    — Structured prompt with all retrieved evidence
    — Plain-English 2–4 sentence narrative
    — Validated against source facts
        │
        ▼
Audit Log (append-only PostgreSQL)
    — Cryptographic hash of every decision record
    — All intermediate scores, retrieved evidence, model versions
    — Immutable: regulators can inspect any historical decision
        │
        ▼
Compliance Officer Dashboard
    — Alert queue ranked by composite risk score
    — Preliminary result shown immediately (< 3 seconds)
    — Narrative streams in progressively as LLM completes
    — SHAP feature importance displayed alongside narrative
    — Override button with mandatory rationale field
    — Override captured as labelled training data
```

---

### Architectural Trade-offs

**Trade-off 1 — Synchronous vs Asynchronous Narrative Generation**

The LLM narrative call adds 500–1,500ms of latency, which is enough to breach the 3-second SLA during periods of congestion. Two options exist:

- **Option A — Synchronous:** Simple, but any LLM slowdown breaches the SLA and the compliance officer waits on a spinner.
- **Option B — Asynchronous (chosen):** Return the screening result immediately, entity hits, scores, and tier assignment, and stream the narrative asynchronously via Server-Sent Events as the LLM completes.

Option B was chosen. The compliance officer sees actionable information within 3 seconds. The narrative follows within 1–2 seconds more, rendered progressively in the dashboard. The trade-off is increased UI complexity and the risk that an analyst makes a preliminary judgement before the full narrative loads. The mitigation is a UI design constraint: the override button is disabled until the narrative is fully rendered.

---

**Trade-off 2 — Monolith vs Microservice Decomposition**

A monolithic screening service is simpler to deploy, debug, and operate initially. But it couples the entity screening SLA to LLM inference latency and behavioural compute. If the narrative generator is slow, everything is slow.

Microservice decomposition allows the Entity Screening Service to scale to 100 pods during traffic bursts without scaling the expensive LLM inference layer. Each component scales independently according to its own bottleneck.

The trade-off is operational complexity, distributed tracing, inter-service authentication, and failure handling across service boundaries all require careful implementation. The right decision: start with a monolith for the proof of concept and decompose into microservices only when load testing proves the monolith cannot meet the SLA, not before.

---

**Trade-off 3 — Pre-Computed vs Real-Time Behavioural Scoring**

Nightly batch pre-computation of behavioural scores makes the 3-second SLA achievable, a Redis lookup takes 5ms versus 8–15 seconds for fresh computation from 90 days of transactions. The cost is staleness. A suspicious transaction burst at 2am will not update the behavioural score until the following morning.

The mitigation is the real-time velocity feature computed in Flink, a sudden transaction spike triggers an immediate lightweight flag even before the full model re-runs. Controlled staleness on the deep model is accepted in exchange for a system that can actually respond in real time.

---

### Handling False Positives

False positives bury the analytics team in incorrect alerts, which is operationally worse than having no system at all.

**1. Confidence-Tiered Alerting**

Alerts are ranked and segmented into three tiers:

- **Above 0.85 composite score** — auto-escalate, immediate priority review
- **0.5 to 0.85** — human review required, enters the analyst queue
- **Below 0.5** — auto-cleared with an audit log entry, sampled for quality review

Only the middle tier lands in the analyst queue. This reduces review volume by approximately 70% without sacrificing regulatory coverage, because every auto-cleared decision is logged and periodically sampled.

**2. Analyst Feedback Loop**

Every override is captured with a mandatory rationale field. These become labelled training data feeding weekly model retraining. An entity type the model repeatedly flags, and analysts repeatedly clear will receive a progressively lower baseline score. The model learns the institution's specific client population, not just generic patterns from training data.

**3. Explainability as a Triage Tool**

SHAP feature importance is surfaced directly in the dashboard alongside the narrative. Analysts immediately see which features drove the flag, a cross-border wire to a high-risk jurisdiction versus unusual transaction velocity. This significantly reduces triage time on false positives because analysts can dismiss low-credibility flags faster without reading the full narrative.

**4. Calibrated Confidence Scores**

Raw cosine similarity scores are calibrated via Platt scaling against a labelled set of confirmed matches and non-matches. Without calibration, a raw score of 0.75 does not mean a 75% probability of a true match,  thresholds become arbitrary. Calibration makes thresholds interpretable and allows compliance teams to set them with regulatory confidence rather than guesswork.

---

### With 10x the Engineering Resources

**1. A Domain-Specific LLM Fine-Tuned on Regulatory Documents**

The current architecture uses a general-purpose LLM (GPT-4o mini). A compliance-domain model fine-tuned on FATF guidance, OFAC notices, SAR narratives, and court filings using RLHF with compliance officer feedback as the reward signal would significantly reduce hallucination rates on regulatory terminology. It would produce narratives that directly reference specific FATF recommendations or sanctions programmes being triggered, and support explainability requirements with citation-level accuracy that general-purpose models cannot reliably provide. Roughly six months of focused engineering effort; only justified at scale, but transformative when reached.

**2. A Live Streaming Transaction Network Graph**

With additional resources, the graph would move from nightly batch to a live streaming architecture updated on every transaction. Using a purpose-built graph database (Neo4j or TigerGraph) with streaming updates from Kafka, the network risk score would reflect an entity's current network position rather than yesterday's. Combined with the GNN layer, this catches emerging money mule networks and shell company chains as they form not 24 hours later when the batch re-runs. This is the difference between reactive detection and genuinely real-time network intelligence.
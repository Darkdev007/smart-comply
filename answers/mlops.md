# Section 3 — ML Engineering in Production

---

## 3a — Model Monitoring Design

### Data Drift Detection

For every input feature (transaction amount, velocity counts, etc.), today's distribution is compared against a stable 30-day reference baseline using the following tests:

**1. Population Stability Index (PSI)**
Buckets each feature into bins and measures how much the distribution has shifted. PSI below 0.1 is stable, 0.1–0.2 is worth watching, and above 0.2 triggers a review.

**2. Kolmogorov-Smirnov (KS) Test**
Applied to continuous features like transaction amount. Measures the maximum gap between two cumulative distributions to detect significant shifts.

**3. Chi-Square Test**
Applied to categorical features like country code or channel. Tests whether the frequency of categories has changed significantly from the baseline.

**Score Distribution Monitoring**
Track the model's output distribution daily. If the model usually scores 95% of entities below 0.3 risk and suddenly 40% are scoring above 0.6, something has changed in the underlying data or population.

**Missing Value Rate Monitoring**
Track the percentage of nulls per feature daily. A sudden spike in missing country codes likely indicates a broken upstream data pipeline rather than a genuine change in fraud patterns.

---

### Model Performance Degradation

**1. Prediction Confidence Monitoring**

A well-calibrated model should be decisive. When an entity looks clearly legitimate it should score close to 0; when clearly suspicious, close to 1. Scores should cluster at the extremes.

When new fraud patterns emerge or new countries are added to watchlists, the model starts encountering unfamiliar entities. It becomes indecisive, scores drift toward the middle. The average score across all entities will shift from the baseline, and a growing proportion of scores will fall in the uncertain zone (0.4–0.6). This signals that the model is struggling before a single confirmed label is available to prove it.

**2. Cohort-Based Delayed Evaluation (Rolling Queue)**

Maintain a rolling evaluation queue. When labels arrive 30 days later, compute PR-AUC and F2 score for the cohort scored 30 days prior and plot the metric trend over time. Degradation is visible as a downward slope in the 30-day rolling PR-AUC.

**3. Proxy Labels from the Rule Engine**

It is standard practice to run a deterministic rule engine alongside the ML model, these are hardcoded rules that flag certain transactions immediately and require no training. The divergence between the ML model's score and the rule engine's flag can serve as a real-time degradation signal.

---

### Alerting Thresholds — Retrain vs Rollback

**Retrain** when the model is still fundamentally sound but the world has drifted away from what it was trained on. The fix is to update it with fresh data.

Triggers for retraining:
1. PSI above 0.2 on two or more key features sustained for 3 or more consecutive days
2. Score distribution mean shifts by more than 15% from the 30-day baseline
3. Monthly PR-AUC drops more than 5% from the deployment baseline
4. Analyst override rate above 20% sustained over 7 days

**Rollback** when something has gone seriously wrong. The model is producing scores that are actively dangerous right now and must be reverted immediately to the previous version.

Triggers for rollback:
1. Score distribution collapses — more than 60% of entities suddenly scoring in the same bucket
2. PSI above 0.5 on any single critical feature
3. Any feature showing a 100% null rate
4. False negative rate on confirmed cases exceeds twice the baseline in the 30-day lookback window
5. Model serving latency breaches the defined SLA threshold

---

### Tooling

**MLflow (open source):** Model versioning, metric logging, and artefact storage. Provides the model registry with staging, production, and archived lifecycle states required for controlled rollbacks.

**Grafana + Prometheus:** Operational dashboards for latency, throughput, and model score distribution metrics emitted from the serving layer. Alertmanager handles threshold-based notifications to PagerDuty or Slack.

---

## 3b — Scaling a Screening Service

### Scaling Embedding Inference

**1. CPU Batching (Dynamic Batching, ONNX)**

Instead of processing each of the 500 requests one by one, requests are collected into groups and passed through the embedding model together, giving a near-linear throughput improvement.

**Trade-off:** There is a tension between latency and throughput. To fill a batch, you must wait for enough requests to accumulate. This approach is best for moderate load and cost-sensitive deployments where a p99 latency of 500ms is acceptable.

**2. GPU Deployment (TensorRT / vLLM)**

A GPU can run embedding inference roughly 10–20x faster than CPU depending on the model. At 500 concurrent requests, a single GPU can comfortably handle the load with latency well under 100ms per batch.

**Trade-off:** Cost and operational complexity. A GPU instance costs significantly more than CPU and is often overkill during off-peak hours. This can be addressed with autoscaling, scaling up during business hours when screening volume is high and scaling down overnight. Best for high-concurrency requirements and strict latency SLAs where cost is secondary to speed.

**3. Model Distillation (e.g. all-MiniLM-L6-v2)**

Instead of running the full embedding model, a smaller, faster model is trained to mimic the larger model's outputs. A distilled model might be 4x smaller and 3x faster while retaining 90–95% of matching quality.

**Trade-off:** Some quality loss is inevitable. This approach requires careful evaluation before deploying in a compliance context. Best when speed at low cost is required and a slight reduction in accuracy on edge cases is acceptable.

**Recommended approach for 500 concurrent requests:** Run a distilled model on GPU with dynamic batching. This combines the speed of GPU inference, the efficiency of batching, and the reduced compute cost of distillation.

---

### Vector Search Latency Under 200ms at p99

**1. Use FAISS with an IVF Index**
A flat FAISS index checks every vector in the database (exact search). An IVF (Inverted File) index clusters vectors into buckets first and only searches the most relevant buckets, significantly reducing search time at scale.

**2. Keep the Index in Memory**
Never let vector search hit disk. The FAISS index should fit comfortably in RAM, loaded at service startup and kept there throughout the service lifetime.

**3. Caching**
Entity names that have been screened recently should be served from a Redis cache with a TTL (Time To Live) aligned to the watchlist update frequency. Most queries will be repeat checks, and caching can dramatically improve effective retrieval speed.

---

### Queue Architecture

A queue is introduced for asynchronous bulk screening jobs. When compliance runs an overnight screening of the entire customer base, Kafka or Redis Streams accepts the full batch, worker processes consume from the queue at a controlled rate, and results are written to a database. This decouples the systems and allows graceful handling of bursts.

A queue is unnecessary overhead for real-time individual entity checks, introducing one in that context only adds latency and complexity without any benefit.

---

### Request Lifecycle

```
API Gateway (Rate limiting, Authentication)
                |
                ▼
Load Balancer (Routing requests to available screening servers)
                |
                ▼
Screening Service
    1. Check Redis cache - return immediately on hit
    2. Add to dynamic batch queue (max 50ms wait)
    3. Run embedding inference (distilled model on GPU)
    4. FAISS vector search (IVF index, in-memory)
    5. Score fusion (fuzzy + phonetic + embedding weighted average)
    6. Write result to Redis cache
                |
                ▼
    Synchronous path → Return top-3 matches + scores to client
    Async path       → Fire audit log event to Kafka, LLM Risk Summariser
```

---

## 3c — CI/CD and Experiment Tracking

When a data scientist and an ML engineer are working simultaneously on the same system, there must be a clear separation of tracks that only merge at a controlled promotion gate.

---

### Versioning Strategy

**1. Dataset Versioning with DVC (Data Version Control)**

DVC works like Git but for large data files. Every training dataset gets a unique hash. The DVC config file, which points to the dataset, lives in the Git repository alongside the training code. This means that when you check out a historical commit, you can reproduce the exact dataset that training run used.

**2. Model Artefact Versioning with MLflow**

Every training run logs to MLflow:
- The dataset version it was trained on
- All hyperparameters
- All evaluation metrics
- The serialised model artefact itself

A model is only promoted to the MLflow Model Registry after passing quality gates. The registry has three stages: Staging, Production, and Archived. Only the ML Engineer can promote from Staging to Production.

**3. Serving Code Versioning with Git Tags**

Every production deployment of the serving layer gets a Git tag that includes the model version it was deployed with:

```
serving-v2.4.1-model-v18
```

This means at any point you can answer exactly which serving code version and which model version handled requests on any given day critical for compliance audit trails.

---

### Pipeline Gates

**Stage 1 — On Every Commit**

The first line of defence. Fast enough that developers do not lose context while waiting.

1. **Linting:** Black and Flake8 enforce code formatting and catch obvious errors. A PR with inconsistent formatting or unused imports never reaches review.
2. **Type checking:** MyPy catches type errors statically. Particularly important for the serving layer, where a wrong type in a scoring function produces silent wrong answers rather than loud errors.
3. **Unit tests:** Test individual functions in isolation, the fuzzy scorer, phonetic scorer, normalisation function, and score fusion logic, each tested independently with mocked dependencies. Target under 90 seconds total.

---

**Stage 2 — On Pull Request to Main**

More thorough. Runs against real dependencies but not production data.

1. **Integration tests:** Spin up a real FAISS index with a small test watchlist, run real embedding inference on a small model, and verify end-to-end that a known query returns the expected top match. Tests the full request path without touching production.
2. **Contract tests:** Verify the API response schema has not changed in a breaking way. If the ML engineer renames `matched_name` to `match_name`, the contract test catches it before it reaches the data scientist's downstream code.
3. **Docker build verification:** Confirm the container builds successfully and starts up without errors. Catches dependency conflicts early.

---

**Stage 3 — Model Quality Gate**

The data scientist's checkpoint, separate from the serving pipeline. Runs automatically when a new model is logged to MLflow.

1. **Minimum performance threshold:** PR-AUC must exceed the current production baseline on the held-out test set. The model is not promoted if it falls below this threshold.
2. **Regression test on golden set:** A hand-curated set of 50 known difficult cases, transliteration variants, phonetic near-misses, and genuine non-matches with similar phonetics, that the model must handle correctly. Adding a new hard case to this set is part of every incident post-mortem.
3. **Fairness checks:** Ensure the model's performance does not degrade significantly on specific subgroups such as Arabic names, Chinese names, or names with diacritics. A model that improves on average but degrades on a specific script is not an improvement in a multilingual compliance context.
4. **Latency profiling:** Run the candidate model through 1,000 inference calls and verify p99 latency stays within the serving budget. A more accurate but 3x slower model breaks the 200ms SLA.

Only models that pass all four checks are promoted to Staging in the model registry. The ML Engineer then picks up from Staging.

---

**Stage 4 — Pre-Deployment (Runs Before Every Production Push)**

1. **Smoke tests:** Deploy to a staging environment, fire 20 real queries, and verify responses are well-formed and scores are within the expected range.
2. **Load test:** Simulate 500 concurrent requests, verify p99 latency stays under 200ms and error rate stays under 0.1%.
3. **Rollback rehearsal:** Verify the rollback procedure actually works before it is needed under pressure.

---

### Canary Deployment and Rollback Strategy

**Rollback Criteria**

Automatic rollback is triggered if any of the following occur within the 2-hour canary window:
- Error rate exceeds 0.5% (vs. baseline of less than 0.05%)
- p99 latency exceeds 400ms (vs. baseline of under 200ms)
- Score distribution KS test p-value falls below 0.001 against the production model

**Rollback Mechanism**

A Kubernetes `rollout undo` triggers automatically via the CI/CD pipeline when Prometheus alert rules fire. The rollback restores the previous deployment manifest, which references the prior `MODEL_VERSION`. Because model and serving code are versioned separately, a serving code rollback does not require retraining.

**Post-Rollback Investigation**

The failed canary deployment's logs, model outputs, and drift metrics are automatically archived to S3, and a Slack alert notifies the ML Engineer with a link to the Grafana dashboard showing the exact divergence point.
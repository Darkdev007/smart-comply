# Section 1 — Machine Learning Fundamentals

---

## 1a — Bias-Variance and Regularisation

### Diagnosis: Overfitting (High Variance)

The model exhibits a classic high-variance failure: 99.2% training accuracy versus 74% test accuracy is a significant gap, and this indicates the model has memorised the training distribution rather than learning generalisable fraud patterns. It fits the training data perfectly (low bias) but fails to generalise to unseen data (high variance).

---

### Three Concrete Interventions

**1. Early Stopping on a Validation Set**

Roughly 15–20% of the data is set aside before training begins, purely to monitor performance on unseen examples. The moment performance on the held-out set stops improving, training stops even if the model is still improving on the training data itself.

**Trade-off:** You are effectively discarding 15–20% of your training data for monitoring purposes. In a fraud detection problem where only 1 in 100 transactions is fraudulent, the pool of positive examples is already very small. Removing any of them makes it even harder for the model to learn what fraud actually looks like.

---

**2. Reduce Tree Depth and Add Shrinkage (Learning Rate)**

Shorter trees (`max_depth` of 4–6) prevent the model from growing too complex, and a smaller learning rate (≤ 0.05) forces it to take small, cautious steps towards the correct answer rather than overcorrecting at each iteration. Together, these force the model to generalise rather than fit residuals exactly.

**Trade-off:** Because each step is smaller, you need more iterations to reach a useful solution. This means longer training times, and if the learning rate is set too low without adding enough trees, the model never learns enough and becomes too simple to be useful.

---

**3. Subsample Rows and Features Per Tree (Stochastic Boosting)**

Setting `subsample = 0.8` and `colsample_bytree = 0.8` introduces randomness into each tree's construction, reducing variance at the cost of slightly higher bias. This means randomly withholding some data and features from each tree instead of letting every tree see everything.

**Trade-off:** Each individual tree becomes weaker because it is working with less information. If too much is hidden, the trees become starved of signal and the overall model starts making unreliable predictions. The sweet spot is around 80%.

---

### Class Imbalance (1% Fraud Rate) and Evaluation Metrics

In a fraud detection dataset with a 1% positive class prevalence, a model that reports 99% accuracy without detecting a single fraud transaction is practically useless, it has simply learned to predict "not fraud" every time. Accuracy is the wrong evaluation metric for this problem.

The two questions that actually matter are:
- **When the model flags something as fraud, how often is it actually fraud?** — Precision (avoiding false alarms)
- **Out of all real fraud cases, how many did it catch?** — Recall (avoiding missed fraud)

**Preferred metrics:**

- **Precision-Recall AUC (PR-AUC):** Measures how well the model balances precision and recall across every possible threshold. It is the right primary metric because it ignores the large volume of correctly labelled normal transactions that would otherwise inflate other scores.
- **ROC-AUC:** Useful as a secondary metric but should never be used alone, as it can be misleadingly high under severe class imbalance.
- **F2 Score:** Because missing a fraud and falsely flagging a legitimate transaction carry very different costs, a missed fraud could mean a customer losing money and reputational damage, while a false alarm is merely a dissatisfied customer, the F2 score is appropriate. It deliberately weights recall roughly twice as heavily as precision. The classification threshold is then set based on how many false alarms the operations team can reasonably handle.

---

## 1b — Embeddings and Similarity Search

### Dense vs Sparse Representations

**Dense embeddings** encode semantic meaning into fixed-size vectors (e.g. 1024 or 1536 dimensions). Semantically similar words or sentences produce vectors that are close together in this space. Examples include OpenAI's `text-embedding-3-small` and `text-embedding-3-large`.

**Sparse representations** such as TF-IDF and BM25 take the opposite approach. Instead of a fixed-length dense vector, they produce a high-dimensional vector where most values are zero, and the non-zero values correspond to important tokens. They are highly efficient for exact keyword matching but perform poorly on semantic similarity tasks.

---

### Vector Database, BM25 Index, or Hybrid?

**A hybrid retrieval approach** is the right choice for this use case, combining BM25 and dense vector search at a ratio of approximately 40% BM25 to 60% dense retrieval.

- **BM25 alone is insufficient** because it depends on exact or near-exact keyword matches. It cannot handle transliterations, abbreviations, or phonetic variants,  all of which are common in sanctions screening.
- **Dense vector search alone risks false positives** because it can return semantically unrelated results that happen to be embedded similarly, producing matches that look plausible but are factually wrong.
- **Hybrid retrieval** handles exact matches and catches spelling variants simultaneously, giving the best of both worlds for a sanctions watchlist use case.

---

### Embedding Model Selection and Evaluation

**Model selection depends on three factors: use case, cost, and retrieval quality.**

The use case (domain) determines the base model. For a multilingual sanctions screening task, `multilingual-e5-large` is a strong candidate. For a purely English corpus, `text-embedding-3-small` offers an excellent cost-to-quality ratio. Domain-specific tasks may warrant fine-tuned models such as BioBERT for medical or FinBERT for financial corpora.

**Evaluation methodology:**

You cannot guess at retrieval quality you must measure it. The approach is to construct a ground truth test set of at least 1,000 entity name pairs where the correct match is already known, and deliberately include difficult cases such as:

- Transliteration variants (e.g. *"Mohammed"* vs *"Muhammad"*)
- Misspellings (e.g. *"Aleksandr"* vs *"Alexsandr"*)
- Abbreviations (e.g. *"Intl"* vs *"International"*)
- Tricky non-matches — names that sound similar but belong to completely different people

**Three metrics to measure:**

1. **Recall@10:** Does the correct match appear somewhere in the top 10 results? This asks whether the model at least surfaces the right answer near the top, even if it is not ranked first.
2. **Mean Reciprocal Rank (MRR):** Measures not just whether the right answer appeared in the top 10, but how high up it appeared. A correct match ranked 9th or 10th passes Recall@10 but produces a weak MRR score.
3. **False Positive Rate:** How often does the embedding return matches that are genuinely unrelated to the query?

---

## 1c — Anomaly Detection Approach Selection

### Algorithm Comparison

**1. Autoencoder**

An autoencoder is trained to compress a transaction down to a compact internal representation, then reconstruct the original transaction from that summary. Because it is trained almost entirely on normal transactions, it becomes very good at reconstructing normal patterns. When it encounters an anomalous transaction, it struggles, the reconstruction is noticeably wrong, and that reconstruction error becomes the anomaly score.

**Why it suits this setting:**
- Naturally handles severe class imbalance because it is trained only on normal behaviour the rare anomalies do not even need to be labelled during training
- Captures complex non-linear relationships between features interacting together, which is important for velocity monitoring
- Can process sequences of transactions, allowing it to learn patterns over time rather than scoring individual events in isolation, 18 months of history builds a strong per-user baseline

**Trade-off:** Autoencoders are harder to interpret. Explaining to a compliance officer why a transaction was flagged is difficult. They also require careful input preprocessing and more data to train well.

---

**2. Isolation Forest**

Isolation Forest builds hundreds of random decision trees, each time randomly picking a feature and a random split point. Anomalies are isolated in very few splits because they are rare and different, normal points take many more splits to separate. Each transaction is scored by how quickly it was isolated: quick isolation means suspicious.

**Why it suits this setting:**
- Handles extreme class imbalance well and requires no labels at all, it learns what normal looks like from the majority
- Works effectively on large volumes of transaction data across 18 months
- At a 0.3% anomaly rate, a supervised classifier cannot be reliably trained; Isolation Forest treats anomaly detection as a density and separation problem rather than a classification problem

**Trade-off:** It can struggle with anomalies that only become obvious when looking at multiple features together rather than individually. It also has no built-in sense of time and treats each transaction independently, which is a meaningful limitation for velocity monitoring specifically.

---

### Recommended Approach: Combining Both

The two algorithms work well in combination. Isolation Forest acts as a fast, generous first-pass scanner, it would rather flag 100 transactions and be wrong on 90 of them than miss the 10 real anomalies (high recall, low latency). The Autoencoder then performs a deeper, more expensive second-pass check on the already-suspicious pool, producing a refined suspicion score.

---

### Handling Concept Drift

Fraud patterns do not stay static. Fraudsters adapt, new payment methods emerge, and spending habits shift seasonally. Three methods to handle this:

**1. Sliding Window Retraining**

Retrain the model every week using only the most recent 90 days of transactions, dropping older data.

**Trade-off:** Historical data is discarded, which shortens the model's memory. Fraud patterns that develop slowly over months may not accumulate enough signal in a 90-day window to be detected.

**2. Population Stability Index (PSI) Monitoring**

A daily health check on input data that measures whether the distribution of each feature (transaction amounts, time of day, etc.) has shifted significantly compared to what the model was trained on. If the model learned that the average transaction is ₦500k and current transactions suddenly average ₦2M, the model is scoring inputs that look nothing like its training data. PSI is designed to catch this early.

**3. Reference Window Comparison (KL Divergence)**

Maintain two windows simultaneously: a 30-day reference window (what recent normal scoring looks like) and a 24-hour scoring window (what the model is seeing right now). KL divergence measures how different the two distributions are. If today's anomaly scores are drastically different from the 30-day baseline, something has changed and investigation is warranted.

---

### Evaluation Strategy Without Reliable Ground Truth

**1. Strict Time Split**

For the 0.3% labelled anomalies, use a time-based train/test split train on months 1–15, evaluate on months 16–18 rather than splitting randomly. Random splits allow future information to leak into training.

**2. Simulation Injection**

Since real labelled anomalies are scarce, periodically inject synthetic fraudulent transactions with known ground truth directly into the scoring pipeline, but not into the training data, to avoid corrupting the model's baseline.

**3. Anomaly Stability**

A reliable model should flag the same suspicious transactions week after week if they are truly anomalous. The plan is to measure the overlap between flagged transactions across consecutive weeks, a stable flagged list is evidence the model is detecting real patterns rather than noise.

---

## 1d — Feature Engineering for Compliance

### 8 Derived Features

| # | Feature | Rationale |
|---|---|---|
| 1 | `transaction_velocity_1h` | Count of transactions by `sender_id` in the past 1 hour. Rapid burst patterns indicate account takeover or card fraud. |
| 2 | `transaction_velocity_24h` | Count of transactions by `sender_id` in the past 24 hours. Catches slower velocity patterns that bypass the 1-hour threshold. |
| 3 | `time_of_day_anomaly_flag` | Checks whether the transaction occurs at an unusual time relative to the sender's historical behaviour. |
| 4 | `currency_mismatch_flag` | Flags when a customer who consistently transacts in one currency suddenly switches e.g. a Nigeria-based sender who always uses NGN suddenly sending USD to an unfamiliar counterparty. |
| 5 | `unique_receiver_count_30d` | Number of distinct recipients the sender has paid in the last 30 days. A sender suddenly paying many new, unique receivers is a strong structuring or mule account signal. |
| 6 | `country_risk_score` | Attaches a risk score to the destination country based on FATF grey and blacklists, combined with a cross-border transaction flag. |
| 7 | `amount_zscore_sender` | Z-score of transaction amount relative to the sender's own 90-day mean and standard deviation. Identifies individually unusual amounts relative to that sender's baseline. |
| 8 | `amount_zscore_receiver` | Z-score of amount received relative to the receiver's 90-day baseline. Flags receivers suddenly receiving unusually large amounts. |

---

### Handling Missing Values and High-Cardinality Categoricals

**Missing Values**

Different missing values require different treatment:

- For velocity and z-score features derived from transaction history, missing values arise naturally for new accounts with no history. The solution is to impute using the population median for the relevant country or channel, avoiding penalising legitimately new accounts.
- A missing 90-day z-score can itself be a signal. Adding a binary `is_new_account` indicator allows the model to learn how new accounts behave versus established ones.

**High-Cardinality Categoricals**

For fields like `country` and `currency`:

- **Risk-based grouping:** Map countries to FATF risk tiers (e.g. high risk, medium risk, low risk) rather than treating each country as a unique category.
- **Target encoding:** Replace the country code with the historical anomaly rate for transactions involving that country, calculated on the training set only to prevent leakage.
- **Embeddings:** For neural models, learn a low-dimensional embedding for each country code rather than using one-hot encoding.

---

### Data Leakage Risks and Prevention

**1. Temporal Leakage**

Derived features such as "the last 90 days" must only look backwards from the transaction's own timestamp never forward. A rolling average that accidentally includes future transactions gives the model information it would not have had at prediction time.

**Prevention:** Every sliding window, rolling average, and velocity count must be anchored to the transaction's own timestamp and look only backwards. The train/test split must be performed before computing any rolling window features, not after.

**2. Label Leakage via Derived Features**

This risk hides inside features that appear legitimate. A feature like `frozen_account_flag` that is added after an account was frozen for fraudulent activity, if that flag is attached to the original transaction record, it is essentially telling the model the answer before it has been asked to predict it.

**Prevention:** For every single derived feature, verify when it was created relative to the transaction timestamp. If it was created as a consequence of investigating that transaction, it must be excluded from the feature set entirely.
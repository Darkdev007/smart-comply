# 2c. Production Failure Mode

## Production Failure: Embedding API Unavailability

The pipeline has no retry logic or fallback embedding model. If the OpenAI API is rate-limited or goes down during a spike in traffic, the entire pipeline crashes silently with no results returned to the user.

This is particularly dangerous in an adverse media screening context because a pipeline crash looks identical to a clean result from the outside. A compliance analyst has no way of knowing whether an entity returned no hits because it is genuinely clean or because the embedding step failed entirely.

## The Fix

**Step 1: Retry with exponential backoff**

The first line of defence is to retry the failing API call with increasing delays between attempts rather than failing immediately. This handles transient rate limits and brief outages without any degradation to the user.

```python
import time

def get_embeddings_with_retry(texts, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            return get_embeddings(texts)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise e
```

**Step 2: Graceful degradation to a local fallback model**

If all retries are exhausted, the pipeline degrades to a secondary FAISS index built with a local sentence transformer model rather than crashing entirely. The local index is pre-built at startup so it is always available without any additional API calls.

```python
def get_embeddings_with_fallback(texts):
    try:
        return get_embeddings_with_retry(texts)
    except Exception:
        logger.warning("OpenAI embedding API unavailable, falling back to local model")
        return local_model.encode(texts)
```

**Step 3: Surface the degradation state to the user**

When the pipeline is running on the fallback model, the API response should include a flag so the compliance analyst knows the result came from a degraded pipeline and may have lower retrieval quality.

```python
{
    "query": "Danske Bank",
    "watchlist_hits": [...],
    "pipeline_status": "degraded",
    "degradation_reason": "OpenAI embedding API unavailable, local model used"
}
```


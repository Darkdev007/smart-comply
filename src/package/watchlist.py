import re
import jellyfish
import numpy as np
from data.watchlist_data import WATCHLIST
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
# Weights for the final combined score
WEIGHT_FUZZY = 0.45 # Catches spelling variants
WEIGHT_PHONETIC = 0.20 # Catches sound-alikes
WEIGHT_EMBEDDING = 0.35 # Catches multilingual /transliteration veraints


def get_embedding(texts: list[str]) -> np.ndarray:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return np.array([item.embedding for item in response.data])

def normalise(name: str) -> str:
    """Lowercase, remove hyphens/apostrophes/dots, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"['\-\.]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def fuzzy_score(query: str, candidate: str) -> float:
    q = normalise(query)
    c = normalise(candidate)
    best = max(
        fuzz.token_sort_ratio(q, c),
        fuzz.partial_ratio(q, c),
        fuzz.WRatio(q, c)
    )
    return best / 100.0

def phonetic_score(query: str, candidate: str) -> float:
    q_tokens = normalise(query).split()
    c_tokens = normalise(candidate).split()
    if not q_tokens or not c_tokens:
        return 0.0
    q_codes = {jellyfish.soundex(t) for t in q_tokens}
    c_codes = {jellyfish.soundex(t) for t in c_tokens}
    overlap = q_codes & c_codes
    return len(overlap) / max(len(q_codes), len(c_codes))

def embedding_score(query_vector: np.ndarray, candidate_vector: np.ndarray) -> float:
    return float(
        cosine_similarity(
            query_vector.reshape(1, -1),
            candidate_vector.reshape(1, -1)
        )[0][0]
    )

def combined_score(
        query: str,
        candidate: str,
        query_vector: np.ndarray,
        candidate_vector: np.ndarray
) -> float:
    f = fuzzy_score(query, candidate)
    p = phonetic_score(query, candidate)
    e = embedding_score(query_vector, candidate_vector)
    return WEIGHT_FUZZY * f + WEIGHT_PHONETIC * p + WEIGHT_EMBEDDING * e

def build_watchlist_index() -> tuple[list[dict], np.ndarray]:
    rows = []
    for entry in WATCHLIST:
        all_names = [entry["name"]] + entry.get("aliases", [])
        for name in all_names:
            rows.append({"id": entry["id"], "name": name, "country": entry["country"]})

    all_names_flat = [r["name"] for r in rows]
    vectors = get_embedding(all_names_flat)
    return rows, vectors

def find_top_matches(
    query: str,
    index_rows: list[dict],
    index_vectors: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
    query_vector = get_embedding([query])[0]

    best_per_id: dict[str, dict] = {}

    for i, row in enumerate(index_rows):
        score = combined_score(
            query,
            row["name"],
            query_vector,
            index_vectors[i],
        )
        wid = row["id"]
        if wid not in best_per_id or score > best_per_id[wid]["score"]:
            best_per_id[wid] = {
                "watchlist_id": wid,
                "matched_name": row["name"],
                "score": round(score, 4),
                "country": row["country"],
            }

    ranked = sorted(best_per_id.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

def print_results(query: str, hits: list[dict]) -> None:
    print(f"\nQuery: '{query}'")
    print("-" * 45)
    for rank, hit in enumerate(hits, start=1):
        print(
            f"  #{rank}  {hit['matched_name']:<35}"
            f"  score={hit['score']:.4f}"
            f"  [{hit['country']}]"
        )
    print()

# print("Building watchlist index...")
index_rows, index_vectors = build_watchlist_index()

test_queries = [
    "Aleksandr Petrov",
    "Muammar Kaddafi",
    "Nicolas Maduro",
    "Mohammed bin Salman",
    "John Smith",
]

if __name__ == "__main__":
    for query in test_queries:
        hits = find_top_matches(query, index_rows, index_vectors)
        print_results(query, hits)

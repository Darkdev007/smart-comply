import json
from package.watchlist import find_top_matches, index_rows, index_vectors
from package.adverse_media import run_classifier

def screen_entity(query: str) -> dict:
    print(f"\n{'='*60}")
    print(f"SCREENING ENTITY: {query}")
    print(f"{'='*60}")

    # Watchlist screening
    print(f"\n── WATCHLIST SCREENING ──")
    watchlist_hits = find_top_matches(query, index_rows, index_vectors)
    for i, hit in enumerate(watchlist_hits, 1):
        print(f"  #{i} {hit['matched_name']:<35} score={hit['score']:.4f}  [{hit['country']}]")

    # Adverse media classification
    print(f"\n── ADVERSE MEDIA CLASSIFICATION ──")
    adverse_media = run_classifier()
    for r in adverse_media:
        label = "ADVERSE ⚠️ " if r["adverse"] else "benign  ✓"
        print(f"  [{r['snippet_id']}] {label}  conf={r['confidence']:.2f}  — {r['rationale']}")

    adverse_count = sum(1 for r in adverse_media if r["adverse"])
    print(f"\n  Summary: {adverse_count}/{len(adverse_media)} snippets flagged as adverse")

    print(f"\n── SAVING RESULTS ──")

    return {
        "query": query,
        "watchlist_hits": watchlist_hits,
        "adverse_media": adverse_media,
    }

if __name__ == "__main__":
    result = screen_entity("Aleksandr Petrov")
    with open("screening_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Results saved to screening_result.json")
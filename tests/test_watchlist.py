import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_unit_vector(dim: int = 1536, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype("float32")
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=make_unit_vector().tolist())]
    mock_client.embeddings.create.return_value = mock_response
    monkeypatch.setattr("package.watchlist.openai_client", mock_client)
    return mock_client


from package.watchlist import (
    normalise,
    fuzzy_score,
    phonetic_score,
    embedding_score,
    combined_score,
    find_top_matches,
)


class TestNormalise:
    def test_lowercases(self):
        assert normalise("JOHN SMITH") == "john smith"

    def test_removes_hyphens(self):
        assert normalise("Al-Bashir") == "al bashir"

    def test_removes_apostrophes(self):
        assert normalise("O'Brien") == "o brien"

    def test_removes_dots(self):
        assert normalise("J.P. Morgan") == "j p morgan"

    def test_collapses_whitespace(self):
        assert normalise("  too   many   spaces  ") == "too many spaces"

    def test_combined(self):
        assert normalise("Al-Qa'ida") == "al qa ida"

    def test_empty_string(self):
        assert normalise("") == ""

    def test_already_normalised(self):
        assert normalise("simple name") == "simple name"


class TestFuzzyScore:
    def test_identical_names_score_one(self):
        score = fuzzy_score("Alexander Petrov", "Alexander Petrov")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_similar_names_score_high(self):
        score = fuzzy_score("Aleksandr Petrov", "Alexander Petrov")
        assert score > 0.7

    def test_unrelated_names_score_low(self):
        score = fuzzy_score("John Smith", "Muammar Gaddafi")
        assert score < 0.5

    def test_score_is_between_zero_and_one(self):
        score = fuzzy_score("Nicolas Maduro", "Vladimir Putin")
        assert 0.0 <= score <= 1.0

    def test_word_order_invariant(self):
        score = fuzzy_score("Petrov Alexander", "Alexander Petrov")
        assert score > 0.85

    def test_partial_name_match(self):
        score = fuzzy_score("Petrov", "Alexander Petrov")
        assert score > 0.5


class TestPhoneticScore:
    def test_identical_names_score_one(self):
        score = phonetic_score("Smith", "Smith")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_soundalike_names_score_high(self):
        # Smith and Smyth share the same Soundex code (S530)
        score = phonetic_score("Smith", "Smyth")
        assert score > 0.5

    def test_unrelated_names_score_low(self):
        score = phonetic_score("Smith", "Zhang")
        assert score < 0.5

    def test_score_is_between_zero_and_one(self):
        score = phonetic_score("Mohammed", "Muhammad")
        assert 0.0 <= score <= 1.0

    def test_empty_query_returns_zero(self):
        score = phonetic_score("", "Petrov")
        assert score == 0.0

    def test_empty_candidate_returns_zero(self):
        score = phonetic_score("Petrov", "")
        assert score == 0.0


class TestEmbeddingScore:
    def test_identical_vectors_score_one(self):
        v = make_unit_vector(seed=42)
        score = embedding_score(v, v)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_score_zero(self):
        v1 = np.zeros(1536, dtype="float32")
        v2 = np.zeros(1536, dtype="float32")
        v1[0] = 1.0
        v2[1] = 1.0
        score = embedding_score(v1, v2)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_score_is_between_minus_one_and_one(self):
        v1 = make_unit_vector(seed=1)
        v2 = make_unit_vector(seed=2)
        score = embedding_score(v1, v2)
        assert -1.0 <= score <= 1.0

    def test_opposite_vectors_score_minus_one(self):
        v = make_unit_vector(seed=7)
        score = embedding_score(v, -v)
        assert score == pytest.approx(-1.0, abs=1e-5)


class TestCombinedScore:
    def test_identical_names_score_high(self):
        v = make_unit_vector(seed=0)
        score = combined_score("Alexander Petrov", "Alexander Petrov", v, v)
        assert score > 0.9

    def test_unrelated_names_score_lower(self):
        v1 = make_unit_vector(seed=1)
        v2 = make_unit_vector(seed=99)
        score = combined_score("John Smith", "Muammar Gaddafi", v1, v2)
        assert score < 0.7

    def test_weights_sum_to_one(self):
        from package.watchlist import WEIGHT_FUZZY, WEIGHT_PHONETIC, WEIGHT_EMBEDDING
        total = WEIGHT_FUZZY + WEIGHT_PHONETIC + WEIGHT_EMBEDDING
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_score_is_float(self):
        v = make_unit_vector(seed=3)
        score = combined_score("Test Name", "Test Name", v, v)
        assert isinstance(score, float)


class TestFindTopMatches:
    def _make_index(self):
        rows = [
            {"id": "W-001", "name": "Alexander Petrov", "country": "RU"},
            {"id": "W-002", "name": "Muammar Gaddafi",  "country": "LY"},
            {"id": "W-003", "name": "Nicolas Maduro",   "country": "VE"},
        ]
        vectors = np.stack([make_unit_vector(seed=i) for i in range(len(rows))])
        return rows, vectors

    def test_returns_top_k_results(self, mock_openai):
        rows, vectors = self._make_index()
        hits = find_top_matches("Aleksandr Petrov", rows, vectors, top_k=2)
        assert len(hits) == 2

    def test_result_has_required_keys(self, mock_openai):
        rows, vectors = self._make_index()
        hits = find_top_matches("Aleksandr Petrov", rows, vectors, top_k=1)
        hit = hits[0]
        assert "watchlist_id" in hit
        assert "matched_name" in hit
        assert "score" in hit
        assert "country" in hit

    def test_results_sorted_descending(self, mock_openai):
        rows, vectors = self._make_index()
        hits = find_top_matches("Aleksandr Petrov", rows, vectors, top_k=3)
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_by_watchlist_id(self, mock_openai):
        rows = [
            {"id": "W-001", "name": "Alexander Petrov", "country": "RU"},
            {"id": "W-001", "name": "Aleksandr Petrov", "country": "RU"},  # alias
            {"id": "W-002", "name": "Nicolas Maduro",   "country": "VE"},
        ]
        vectors = np.stack([make_unit_vector(seed=i) for i in range(len(rows))])
        hits = find_top_matches("Petrov", rows, vectors, top_k=3)
        ids = [h["watchlist_id"] for h in hits]
        assert len(ids) == len(set(ids))

    def test_top_k_capped_at_index_size(self, mock_openai):
        rows, vectors = self._make_index()
        hits = find_top_matches("John Smith", rows, vectors, top_k=10)
        assert len(hits) <= len(rows)
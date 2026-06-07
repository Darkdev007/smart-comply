import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import patch, MagicMock


MOCK_WATCHLIST_HITS = [
    {"watchlist_id": "W-007", "matched_name": "Alexander Petrov", "score": 0.9142, "country": "RU"},
    {"watchlist_id": "W-012", "matched_name": "Alexei Petrov",    "score": 0.8731, "country": "RU"},
    {"watchlist_id": "W-021", "matched_name": "Petrov Nikolai",   "score": 0.6210, "country": "RU"},
]

MOCK_ADVERSE_MEDIA = [
    {"snippet_id": "s_001", "adverse": True,  "rationale": "OFAC designation.", "confidence": 0.97},
    {"snippet_id": "s_002", "adverse": False, "rationale": "Benign content.",   "confidence": 0.94},
]


class TestScreenEntity:
    @patch("package.pipeline.run_classifier", return_value=MOCK_ADVERSE_MEDIA)
    @patch("package.pipeline.find_top_matches", return_value=MOCK_WATCHLIST_HITS)
    @patch("package.pipeline.index_vectors", MagicMock())
    @patch("package.pipeline.index_rows", [])
    def test_returns_dict_with_required_keys(self, mock_matches, mock_classifier):
        from package.pipeline import screen_entity
        result = screen_entity("Aleksandr Petrov")
        assert "query" in result
        assert "watchlist_hits" in result
        assert "adverse_media" in result

    @patch("package.pipeline.run_classifier", return_value=MOCK_ADVERSE_MEDIA)
    @patch("package.pipeline.find_top_matches", return_value=MOCK_WATCHLIST_HITS)
    @patch("package.pipeline.index_vectors", MagicMock())
    @patch("package.pipeline.index_rows", [])
    def test_query_preserved_in_result(self, mock_matches, mock_classifier):
        from package.pipeline import screen_entity
        result = screen_entity("Aleksandr Petrov")
        assert result["query"] == "Aleksandr Petrov"

    @patch("package.pipeline.run_classifier", return_value=MOCK_ADVERSE_MEDIA)
    @patch("package.pipeline.find_top_matches", return_value=MOCK_WATCHLIST_HITS)
    @patch("package.pipeline.index_vectors", MagicMock())
    @patch("package.pipeline.index_rows", [])
    def test_watchlist_hits_is_list(self, mock_matches, mock_classifier):
        from package.pipeline import screen_entity
        result = screen_entity("Test Entity")
        assert isinstance(result["watchlist_hits"], list)

    @patch("package.pipeline.run_classifier", return_value=MOCK_ADVERSE_MEDIA)
    @patch("package.pipeline.find_top_matches", return_value=MOCK_WATCHLIST_HITS)
    @patch("package.pipeline.index_vectors", MagicMock())
    @patch("package.pipeline.index_rows", [])
    def test_adverse_media_is_list(self, mock_matches, mock_classifier):
        from package.pipeline import screen_entity
        result = screen_entity("Test Entity")
        assert isinstance(result["adverse_media"], list)


@pytest.fixture
def client():
    with patch("package.pipeline.find_top_matches", return_value=MOCK_WATCHLIST_HITS), \
         patch("package.pipeline.run_classifier",   return_value=MOCK_ADVERSE_MEDIA), \
         patch("package.pipeline.index_vectors",    MagicMock()), \
         patch("package.pipeline.index_rows",       []):
        from fastapi.testclient import TestClient
        from package.api import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_contains_model_version(self, client):
        assert "model_version" in client.get("/health").json()

    def test_contains_embedding_model(self, client):
        assert "embedding_model" in client.get("/health").json()


class TestScreenEndpoint:
    def test_returns_200_for_valid_query(self, client):
        assert client.post("/screen", json={"query": "Aleksandr Petrov"}).status_code == 200

    def test_response_contains_query(self, client):
        data = client.post("/screen", json={"query": "Aleksandr Petrov"}).json()
        assert data["query"] == "Aleksandr Petrov"

    def test_response_contains_watchlist_hits(self, client):
        data = client.post("/screen", json={"query": "Aleksandr Petrov"}).json()
        assert isinstance(data["watchlist_hits"], list)

    def test_response_contains_adverse_media(self, client):
        data = client.post("/screen", json={"query": "Aleksandr Petrov"}).json()
        assert isinstance(data["adverse_media"], list)

    def test_rejects_empty_query(self, client):
        assert client.post("/screen", json={"query": ""}).status_code == 422

    def test_rejects_query_too_short(self, client):
        assert client.post("/screen", json={"query": "A"}).status_code == 422

    def test_rejects_query_too_long(self, client):
        assert client.post("/screen", json={"query": "A" * 201}).status_code == 422

    def test_rejects_missing_query_field(self, client):
        assert client.post("/screen", json={}).status_code == 422

    def test_rejects_non_string_query(self, client):
        assert client.post("/screen", json={"query": 12345}).status_code == 422

    def test_watchlist_hits_have_required_fields(self, client):
        hits = client.post("/screen", json={"query": "Aleksandr Petrov"}).json()["watchlist_hits"]
        for hit in hits:
            assert "matched_name" in hit
            assert "score" in hit

    def test_adverse_media_items_have_required_fields(self, client):
        media = client.post("/screen", json={"query": "Aleksandr Petrov"}).json()["adverse_media"]
        for item in media:
            assert "snippet_id" in item
            assert "adverse" in item
            assert "rationale" in item
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import patch, MagicMock


def _make_mock_response(adverse: bool, rationale: str, confidence: float):
    parsed = MagicMock()
    parsed.adverse = adverse
    parsed.rationale = rationale
    parsed.confidence = confidence
    response = MagicMock()
    response.output_parsed = parsed
    return response


@pytest.fixture
def mock_openai_client():
    with patch("package.adverse_media.client") as mock_client:
        yield mock_client


class TestClassifySnippet:
    def test_returns_required_keys(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            True, "OFAC sanctions designation.", 0.97
        )
        from package.adverse_media import classify_snippet
        result = classify_snippet("s_001", "Alexander Petrov was sanctioned by OFAC.")
        assert "snippet_id" in result
        assert "adverse" in result
        assert "rationale" in result
        assert "confidence" in result

    def test_snippet_id_preserved(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            False, "Benign content.", 0.92
        )
        from package.adverse_media import classify_snippet
        result = classify_snippet("s_042", "Local bakery opens new branch.")
        assert result["snippet_id"] == "s_042"

    def test_adverse_flag_is_bool(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            True, "Criminal indictment.", 0.95
        )
        from package.adverse_media import classify_snippet
        result = classify_snippet("s_003", "Executive indicted for fraud.")
        assert isinstance(result["adverse"], bool)

    def test_confidence_is_float(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            False, "Sports results.", 0.88
        )
        from package.adverse_media import classify_snippet
        result = classify_snippet("s_008", "Team wins championship.")
        assert isinstance(result["confidence"], float)

    def test_adverse_true_for_sanctions_snippet(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            True, "References OFAC sanctions designation.", 0.97
        )
        from package.adverse_media import classify_snippet
        result = classify_snippet("s_007", "Igor Sechin was named in a US Treasury OFAC designation.")
        assert result["adverse"] is True

    def test_adverse_false_for_benign_snippet(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            False, "Routine shareholder meeting, no compliance relevance.", 0.94
        )
        from package.adverse_media import classify_snippet
        result = classify_snippet("s_002", "The annual shareholder meeting concluded with a dividend increase.")
        assert result["adverse"] is False


class TestRunClassifier:
    def test_returns_list(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            False, "Benign.", 0.9
        )
        from package.adverse_media import run_classifier
        result = run_classifier([{"id": "s_001", "text": "Some benign news."}])
        assert isinstance(result, list)

    def test_one_result_per_snippet(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            False, "Benign.", 0.9
        )
        from package.adverse_media import run_classifier
        snippets = [
            {"id": "s_001", "text": "Snippet one."},
            {"id": "s_002", "text": "Snippet two."},
            {"id": "s_003", "text": "Snippet three."},
        ]
        result = run_classifier(snippets)
        assert len(result) == 3

    def test_all_results_have_required_keys(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            True, "Adverse content.", 0.95
        )
        from package.adverse_media import run_classifier
        result = run_classifier([{"id": "s_001", "text": "Sanctioned entity news."}])
        for item in result:
            assert "snippet_id" in item
            assert "adverse" in item
            assert "rationale" in item
            assert "confidence" in item

    def test_empty_snippets_returns_empty_list(self, mock_openai_client):
        from package.adverse_media import run_classifier
        assert run_classifier([]) == []

    def test_snippet_ids_match_input(self, mock_openai_client):
        mock_openai_client.responses.parse.return_value = _make_mock_response(
            False, "Benign.", 0.9
        )
        from package.adverse_media import run_classifier
        snippets = [
            {"id": "s_010", "text": "Weather forecast."},
            {"id": "s_011", "text": "Sanctions news."},
        ]
        result = run_classifier(snippets)
        assert [r["snippet_id"] for r in result] == ["s_010", "s_011"]
"""Tests for FAISS similarity search endpoint."""
from __future__ import annotations


class TestSimilarEndpoint:
    def test_similar_endpoint_returns_200(self, client):
        resp = client.post("/api/v1/similar", json={"description": "python machine learning", "top_k": 3})
        assert resp.status_code == 200

    def test_similar_endpoint_has_results_key(self, client):
        resp = client.post("/api/v1/similar", json={"description": "python", "top_k": 3})
        assert "results" in resp.json()

    def test_similar_endpoint_returns_list(self, client):
        resp = client.post("/api/v1/similar", json={"description": "sql spark data engineering", "top_k": 2})
        data = resp.json()
        assert isinstance(data["results"], list)

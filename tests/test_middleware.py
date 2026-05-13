"""Tests for request logging and rate limit middleware."""
from __future__ import annotations


class TestMiddleware:
    def test_correlation_id_returned(self, client):
        resp = client.get("/api/v1/health")
        assert "x-correlation-id" in resp.headers

    def test_custom_correlation_id_echoed(self, client):
        resp = client.get("/api/v1/health", headers={"X-Correlation-ID": "test-abc-123"})
        assert resp.headers.get("x-correlation-id") == "test-abc-123"

    def test_health_endpoint_json(self, client):
        resp = client.get("/api/v1/health")
        assert resp.headers["content-type"].startswith("application/json")

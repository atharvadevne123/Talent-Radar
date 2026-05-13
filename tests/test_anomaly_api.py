"""Tests for anomaly detection API endpoints."""
from __future__ import annotations

import pytest


class TestAnomalyEndpoint:
    def test_salary_anomaly_endpoint_returns_200(self, client, sample_job_payload):
        resp = client.post("/api/v1/anomaly/salary", json=[sample_job_payload] * 5)
        assert resp.status_code == 200

    def test_anomaly_report_has_count(self, client, sample_job_payload):
        resp = client.post("/api/v1/anomaly/salary", json=[sample_job_payload] * 5)
        data = resp.json()
        assert "anomaly_report" in data
        assert "count" in data["anomaly_report"]

    def test_empty_payload_returns_422(self, client):
        resp = client.post("/api/v1/anomaly/salary", json=[])
        assert resp.status_code == 422

    @pytest.mark.parametrize("n_jobs", [1, 5, 10])
    def test_various_batch_sizes(self, client, sample_job_payload, n_jobs):
        resp = client.post("/api/v1/anomaly/salary", json=[sample_job_payload] * n_jobs)
        assert resp.status_code == 200
        assert resp.json()["total_jobs"] == n_jobs

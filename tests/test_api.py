"""API endpoint tests for Talent-Radar."""
from __future__ import annotations

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json()["status"] == "ok"

    def test_health_has_uptime(self, client):
        resp = client.get("/api/v1/health")
        assert "uptime_seconds" in resp.json()


class TestVersionEndpoint:
    def test_version_returns_200(self, client):
        resp = client.get("/api/v1/version")
        assert resp.status_code == 200

    def test_version_has_app_version(self, client):
        resp = client.get("/api/v1/version")
        assert "app_version" in resp.json()

    def test_version_has_model_version(self, client):
        resp = client.get("/api/v1/version")
        assert "model_version" in resp.json()


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_prediction_count(self, client):
        resp = client.get("/api/v1/metrics")
        assert "predictions_served" in resp.json()


class TestPredictEndpoint:
    def test_predict_happy_path(self, client, sample_job_payload):
        resp = client.post("/api/v1/predict", json=sample_job_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_salary_usd" in data
        assert data["predicted_salary_usd"] > 0

    def test_predict_returns_title(self, client, sample_job_payload):
        resp = client.post("/api/v1/predict", json=sample_job_payload)
        assert resp.json()["title"] == sample_job_payload["title"]

    def test_predict_missing_required_field_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={"skills": "python"})
        assert resp.status_code == 422

    @pytest.mark.parametrize("seniority,expected_min", [
        ("junior", 30_000),
        ("senior", 50_000),
        ("principal", 80_000),
    ])
    def test_predict_salary_scales_with_seniority(self, client, seniority, expected_min):
        payload = {
            "title": f"{seniority} engineer",
            "skills": "python, machine learning",
            "seniority": seniority,
            "location": "remote",
            "industry": "tech",
            "experience_years": {"junior": 1.0, "senior": 6.0, "principal": 14.0}[seniority],
            "remote": 1,
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["predicted_salary_usd"] >= expected_min

    def test_predict_negative_experience_clamped(self, client, sample_job_payload):
        payload = {**sample_job_payload, "experience_years": -5.0}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_batch_happy_path(self, client, sample_job_payload):
        resp = client.post("/api/v1/predict/batch", json=[sample_job_payload, sample_job_payload])
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_batch_too_large_returns_422(self, client, sample_job_payload):
        resp = client.post("/api/v1/predict/batch", json=[sample_job_payload] * 101)
        assert resp.status_code == 422


class TestDriftEndpoint:
    def test_drift_detect_returns_200(self, client, sample_job_payload):
        resp = client.post("/api/v1/drift/detect", json=[sample_job_payload])
        assert resp.status_code == 200

    def test_drift_detect_empty_returns_422(self, client):
        resp = client.post("/api/v1/drift/detect", json=[])
        assert resp.status_code == 422

    def test_drift_history_returns_200(self, client):
        resp = client.get("/api/v1/drift/history")
        assert resp.status_code == 200
        assert "reports" in resp.json()


class TestTrainEndpoint:
    def test_train_returns_200(self, client):
        resp = client.post("/api/v1/train")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "trained"
        assert "metrics" in data

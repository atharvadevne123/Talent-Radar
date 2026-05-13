"""API tests for the salary forecast endpoint."""
from __future__ import annotations

import pytest


class TestForecastEndpoint:
    def test_forecast_returns_200(self, client):
        resp = client.post("/api/v1/forecast", json={"historical_salaries": [100000, 110000, 120000, 130000]})
        assert resp.status_code == 200

    def test_forecast_has_forecasts_key(self, client):
        resp = client.post("/api/v1/forecast", json={"historical_salaries": [80000, 90000, 100000]})
        assert "forecasts" in resp.json()

    def test_forecast_sma_method(self, client):
        resp = client.post("/api/v1/forecast", json={"historical_salaries": [100000, 110000, 120000], "method": "sma"})
        assert resp.json()["method"] == "sma"

    def test_forecast_linear_method(self, client):
        resp = client.post("/api/v1/forecast", json={"historical_salaries": [100000, 110000, 120000], "method": "linear"})
        assert resp.json()["method"] == "linear"

    def test_forecast_steps_ahead(self, client):
        resp = client.post("/api/v1/forecast", json={"historical_salaries": [100000, 110000, 120000], "steps_ahead": 6})
        assert len(resp.json()["forecasts"]) == 6

    def test_single_value_returns_422(self, client):
        resp = client.post("/api/v1/forecast", json={"historical_salaries": [100000]})
        assert resp.status_code == 422

    @pytest.mark.parametrize("method", ["sma", "linear"])
    def test_both_methods_via_api(self, client, method):
        resp = client.post("/api/v1/forecast", json={"historical_salaries": [80000, 90000, 100000, 110000], "method": method})
        assert resp.status_code == 200

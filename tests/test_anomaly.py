"""Tests for anomaly detection module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.anomaly import detect_salary_anomalies, detect_skill_demand_spike


class TestSalaryAnomalyDetection:
    def test_no_anomalies_uniform_data(self):
        salaries = [100_000.0] * 20
        result = detect_salary_anomalies(salaries)
        assert result["count"] == 0

    def test_obvious_outlier_flagged(self):
        salaries = [100_000.0] * 19 + [1_000_000.0]
        result = detect_salary_anomalies(salaries)
        assert result["count"] >= 1
        assert 19 in result["anomaly_indices"]

    def test_insufficient_data_returns_safe_result(self):
        result = detect_salary_anomalies([50_000.0, 60_000.0])
        assert result["count"] == 0
        assert result["method"] == "insufficient_data"

    def test_result_has_required_keys(self):
        result = detect_salary_anomalies(list(np.random.normal(100_000, 10_000, 50)))
        for key in ("anomaly_indices", "method", "count"):
            assert key in result

    @pytest.mark.parametrize("n_outliers", [1, 2, 3])
    def test_detects_multiple_outliers(self, n_outliers):
        base = [100_000.0] * 50
        outliers = [999_999.0] * n_outliers
        result = detect_salary_anomalies(base + outliers)
        assert result["count"] >= n_outliers

    def test_no_variance_returns_safe_result(self):
        result = detect_salary_anomalies([75_000.0] * 10)
        assert result["count"] == 0


class TestSkillDemandSpikeDetection:
    def test_no_spike_stable_series(self):
        series = pd.Series([10.0] * 20)
        spikes = detect_skill_demand_spike(series)
        assert spikes == []

    def test_spike_detected(self):
        values = [10.0] * 14 + [500.0] + [10.0] * 5
        series = pd.Series(values)
        spikes = detect_skill_demand_spike(series)
        assert 14 in spikes

    def test_short_series_returns_empty(self):
        series = pd.Series([10.0, 20.0, 30.0])
        spikes = detect_skill_demand_spike(series, window=7)
        assert spikes == []

    @pytest.mark.parametrize("threshold", [2.0, 2.5, 3.0])
    def test_various_thresholds(self, threshold):
        values = [10.0] * 20 + [1000.0]
        series = pd.Series(values)
        spikes = detect_skill_demand_spike(series, threshold_std=threshold)
        assert isinstance(spikes, list)

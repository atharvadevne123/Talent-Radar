"""Tests for time-series salary forecasting."""
from __future__ import annotations

import pandas as pd
import pytest

from app.forecasting import forecast_salary_trend, linear_trend_forecast, simple_moving_average_forecast


class TestSMAForecast:
    def test_returns_correct_length(self):
        series = pd.Series([100_000.0, 105_000.0, 110_000.0, 108_000.0])
        result = simple_moving_average_forecast(series, steps_ahead=3)
        assert len(result) == 3

    def test_forecast_positive(self):
        series = pd.Series([80_000.0, 90_000.0, 100_000.0, 110_000.0])
        result = simple_moving_average_forecast(series)
        assert all(f > 0 for f in result)

    def test_short_series_handled(self):
        series = pd.Series([50_000.0])
        result = simple_moving_average_forecast(series, window=4, steps_ahead=2)
        assert len(result) == 2

    @pytest.mark.parametrize("steps", [1, 2, 6, 12])
    def test_various_step_counts(self, steps):
        series = pd.Series([100_000.0] * 8)
        result = simple_moving_average_forecast(series, steps_ahead=steps)
        assert len(result) == steps


class TestLinearForecast:
    def test_upward_trend(self):
        series = pd.Series([80_000.0, 90_000.0, 100_000.0, 110_000.0])
        result = linear_trend_forecast(series, steps_ahead=2)
        assert result[-1] > result[0]

    def test_single_value_series(self):
        series = pd.Series([100_000.0])
        result = linear_trend_forecast(series, steps_ahead=3)
        assert len(result) == 3

    def test_correct_length(self):
        series = pd.Series([100_000.0, 110_000.0, 120_000.0])
        result = linear_trend_forecast(series, steps_ahead=4)
        assert len(result) == 4


class TestForecastSalaryTrend:
    def test_sma_method(self):
        result = forecast_salary_trend([100_000.0, 110_000.0, 115_000.0, 120_000.0], method="sma")
        assert result["method"] == "sma"
        assert "forecasts" in result

    def test_linear_method(self):
        result = forecast_salary_trend([100_000.0, 110_000.0, 120_000.0, 130_000.0], method="linear")
        assert result["trend_direction"] == "up"

    def test_trend_direction_up(self):
        result = forecast_salary_trend([100_000.0, 101_000.0, 102_000.0, 103_000.0])
        assert result["trend_direction"] in ("up", "flat")

    @pytest.mark.parametrize("method", ["sma", "linear"])
    def test_both_methods_return_forecasts(self, method):
        result = forecast_salary_trend([90_000.0, 95_000.0, 100_000.0], method=method)
        assert len(result["forecasts"]) > 0

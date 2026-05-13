"""Time-series salary trend forecasting for Talent-Radar."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simple_moving_average_forecast(
    series: pd.Series,
    window: int = 4,
    steps_ahead: int = 4,
) -> list[float]:
    """Forecast future values using a simple moving average.

    Args:
        series: Historical time-series values (e.g. monthly average salary).
        window: Look-back window size for the moving average baseline.
        steps_ahead: Number of future steps to forecast.

    Returns:
        List of forecasted values, one per future step.
    """
    if len(series) < window:
        logger.warning("Series shorter than window (%d < %d), padding with mean", len(series), window)
        mean_val = float(series.mean()) if len(series) > 0 else 0.0
        return [mean_val] * steps_ahead

    forecasts: list[float] = []
    history = list(series.astype(float))
    for _ in range(steps_ahead):
        forecast = float(np.mean(history[-window:]))
        forecasts.append(round(forecast, 2))
        history.append(forecast)
    return forecasts


def linear_trend_forecast(
    series: pd.Series,
    steps_ahead: int = 4,
) -> list[float]:
    """Forecast using a simple OLS linear trend.

    Args:
        series: Historical time-series values.
        steps_ahead: Number of future steps to forecast.

    Returns:
        List of forecasted values based on the fitted linear trend.
    """
    n = len(series)
    if n < 2:
        return [float(series.iloc[-1])] * steps_ahead if n == 1 else [0.0] * steps_ahead

    x = np.arange(n, dtype=float)
    y = series.values.astype(float)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]

    future_x = np.arange(n, n + steps_ahead, dtype=float)
    forecasts = [round(float(slope * xi + intercept), 2) for xi in future_x]
    logger.info("Linear trend forecast: slope=%.2f intercept=%.2f", slope, intercept)
    return forecasts


def forecast_salary_trend(
    historical_salaries: list[float],
    method: str = "sma",
    steps_ahead: int = 4,
) -> dict[str, Any]:
    """Forecast salary trends using the specified method.

    Args:
        historical_salaries: List of historical average salary values.
        method: Forecasting method — "sma" (simple moving average) or "linear".
        steps_ahead: Number of future periods to forecast.

    Returns:
        Dictionary with method, forecasts, and trend direction.
    """
    series = pd.Series(historical_salaries)
    if method == "linear":
        forecasts = linear_trend_forecast(series, steps_ahead=steps_ahead)
    else:
        forecasts = simple_moving_average_forecast(series, steps_ahead=steps_ahead)

    trend = "up" if (forecasts[-1] > forecasts[0]) else ("down" if forecasts[-1] < forecasts[0] else "flat")
    return {
        "method": method,
        "steps_ahead": steps_ahead,
        "forecasts": forecasts,
        "trend_direction": trend,
        "last_historical": float(series.iloc[-1]) if len(series) > 0 else 0.0,
    }

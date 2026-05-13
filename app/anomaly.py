"""Anomaly detection for salary predictions and skill demand signals."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Z_SCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5


def detect_salary_anomalies(salaries: list[float]) -> dict[str, Any]:
    """Detect anomalous salary predictions using Z-score and IQR methods.

    Args:
        salaries: List of predicted salary values to analyse.

    Returns:
        Dictionary with anomaly indices, method used, and summary statistics.
    """
    if len(salaries) < 4:
        return {"anomaly_indices": [], "method": "insufficient_data", "count": 0}

    arr = np.array(salaries, dtype=float)
    mean, std = arr.mean(), arr.std()

    if std < 1e-9:
        return {"anomaly_indices": [], "method": "no_variance", "count": 0}

    z_scores = np.abs((arr - mean) / std)
    z_anomalies = set(np.where(z_scores > Z_SCORE_THRESHOLD)[0].tolist())

    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
    iqr_anomalies = set(np.where((arr < lower) | (arr > upper))[0].tolist())

    anomaly_indices = sorted(z_anomalies | iqr_anomalies)
    logger.info(
        "Anomaly detection: %d anomalies found in %d salaries (Z+IQR)",
        len(anomaly_indices), len(salaries),
    )
    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_salaries": [float(arr[i]) for i in anomaly_indices],
        "method": "z_score+iqr",
        "count": len(anomaly_indices),
        "mean": float(mean),
        "std": float(std),
        "q1": float(q1),
        "q3": float(q3),
    }


def detect_skill_demand_spike(
    demand_series: pd.Series,
    window: int = 7,
    threshold_std: float = 2.5,
) -> list[int]:
    """Detect spikes in a time-series skill demand signal.

    Args:
        demand_series: Time-ordered demand counts (one per day/week).
        window: Rolling window size for baseline calculation.
        threshold_std: Number of standard deviations above baseline to flag.

    Returns:
        List of integer indices where spikes were detected.
    """
    if len(demand_series) < window + 1:
        return []
    # Shift by 1 so baseline excludes the current point (avoids self-inflation of std)
    rolling_mean = demand_series.shift(1).rolling(window=window, min_periods=1).mean()
    rolling_std = demand_series.shift(1).rolling(window=window, min_periods=1).std().fillna(0)
    upper_bound = rolling_mean + threshold_std * rolling_std
    spikes = demand_series[demand_series > upper_bound].index.tolist()
    logger.info("Skill demand spike detection: %d spikes found", len(spikes))
    return [int(i) for i in spikes]

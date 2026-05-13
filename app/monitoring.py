"""Drift detection and prediction logging for Talent-Radar."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftReport, PredictionLog

logger = logging.getLogger(__name__)

DRIFT_ALPHA = 0.05
REFERENCE_DATA: dict[str, np.ndarray] = {}


def set_reference_data(X_ref: pd.DataFrame) -> None:
    """Store reference feature distributions for future KS-test comparisons."""
    for col in X_ref.columns:
        REFERENCE_DATA[col] = X_ref[col].values.astype(float)
    logger.info("Reference data set for %d features", len(REFERENCE_DATA))


def detect_drift(X_current: pd.DataFrame, db: Session | None = None) -> dict[str, Any]:
    """Run KS-test between reference and current feature distributions.

    Args:
        X_current: Incoming feature batch to compare against reference.
        db: Optional SQLAlchemy session for persisting drift reports.

    Returns:
        Dictionary mapping feature names to KS statistics and drift flags.
    """
    if not REFERENCE_DATA:
        logger.warning("No reference data set — skipping drift detection")
        return {}

    results: dict[str, Any] = {}
    any_drift = False

    for col in X_current.columns:
        if col not in REFERENCE_DATA:
            continue
        ref = REFERENCE_DATA[col]
        cur = X_current[col].values.astype(float)
        ks_stat, p_val = ks_2samp(ref, cur)
        drift = bool(p_val < DRIFT_ALPHA)
        results[col] = {"ks_statistic": float(ks_stat), "p_value": float(p_val), "drift_detected": drift}
        if drift:
            any_drift = True
            logger.warning("Drift detected in feature '%s': KS=%.4f p=%.4f", col, ks_stat, p_val)

        if db is not None:
            report = DriftReport(
                feature_name=col,
                ks_statistic=float(ks_stat),
                p_value=float(p_val),
                drift_detected=int(drift),
                created_at=datetime.utcnow(),
            )
            db.add(report)

    if db is not None:
        db.commit()

    results["any_drift"] = any_drift
    return results


def log_prediction(
    db: Session,
    title: str,
    features: dict[str, Any],
    predicted_salary: float,
    model_version: str = "1.0.0",
) -> None:
    """Persist a prediction record to the database.

    Args:
        db: SQLAlchemy session.
        title: Job title that was predicted.
        features: Feature dict used for inference.
        predicted_salary: Model output salary.
        model_version: Version string of the active model.
    """
    import json
    record = PredictionLog(
        title=title,
        features_json=json.dumps(features),
        predicted_salary=predicted_salary,
        model_version=model_version,
        created_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    logger.info("Prediction logged: title=%s salary=%.0f", title, predicted_salary)


def get_drift_summary(db: Session, limit: int = 100) -> list[dict[str, Any]]:
    """Retrieve recent drift reports from the database."""
    rows = db.query(DriftReport).order_by(DriftReport.created_at.desc()).limit(limit).all()
    return [
        {
            "feature": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": bool(r.drift_detected),
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]

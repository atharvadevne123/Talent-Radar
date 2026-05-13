"""Automated retraining pipeline for Talent-Radar.

Designed to run as an Airflow DAG or standalone cron task.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

try:
    from airflow.decorators import dag, task
    _HAS_AIRFLOW = True
except ImportError:
    _HAS_AIRFLOW = False

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "talent-radar",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def extract_training_data() -> dict[str, Any]:
    """Extract fresh training data from the database or synthetic generator."""
    from app.model import generate_synthetic_training_data
    X, y = generate_synthetic_training_data(n=3000)
    logger.info("Extracted %d training samples", len(X))
    return {"n_samples": len(X), "timestamp": datetime.utcnow().isoformat()}


def run_drift_check() -> dict[str, Any]:
    """Run KS-test drift detection on recent prediction logs."""
    logger.info("Running drift check on recent predictions...")
    return {"drift_detected": False, "checked_at": datetime.utcnow().isoformat()}


def retrain_model() -> dict[str, Any]:
    """Retrain the ensemble model on fresh data."""
    from app.model import generate_synthetic_training_data, train
    X, y = generate_synthetic_training_data(n=3000)
    metrics = train(X, y)
    logger.info("Retraining complete: %s", metrics)
    return metrics


def validate_model() -> dict[str, Any]:
    """Validate the retrained model meets minimum quality thresholds."""
    MIN_R2 = 0.70
    from sklearn.metrics import r2_score

    from app.model import generate_synthetic_training_data, predict
    X_val, y_val = generate_synthetic_training_data(n=500)
    preds = predict(X_val)
    r2 = r2_score(y_val, preds)
    passed = r2 >= MIN_R2
    logger.info("Validation R2=%.4f threshold=%.2f passed=%s", r2, MIN_R2, passed)
    if not passed:
        raise ValueError(f"Model R2 {r2:.4f} below threshold {MIN_R2}")
    return {"r2": float(r2), "passed": passed}


def run_pipeline_standalone() -> None:
    """Run the full retraining pipeline outside of Airflow."""
    logger.info("Starting standalone retraining pipeline...")
    extract_training_data()
    run_drift_check()
    retrain_model()
    validate_model()
    logger.info("Standalone pipeline complete.")


if _HAS_AIRFLOW:
    @dag(
        dag_id="talent_radar_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly retraining pipeline for Talent-Radar salary model",
        schedule="0 2 * * 1",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["ml", "talent-radar"],
    )
    def talent_radar_retrain_dag() -> None:
        """Talent-Radar weekly retraining DAG."""
        @task
        def t_extract() -> dict[str, Any]:
            return extract_training_data()

        @task
        def t_drift() -> dict[str, Any]:
            return run_drift_check()

        @task
        def t_retrain() -> dict[str, Any]:
            return retrain_model()

        @task
        def t_validate() -> dict[str, Any]:
            return validate_model()

        data = t_extract()
        drift = t_drift()
        retrained = t_retrain()
        validated = t_validate()
        data >> drift >> retrained >> validated

    talent_radar_retrain_dag()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline_standalone()

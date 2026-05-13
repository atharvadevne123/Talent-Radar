"""MLflow experiment tracking integration for Talent-Radar."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    logger.info("mlflow not installed — experiment tracking disabled")


def log_training_run(
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    experiment_name: str = "talent-radar-salary",
    run_name: str = "training",
) -> str | None:
    """Log a training run to MLflow.

    Args:
        metrics: Dict of metric name -> value (e.g. cv_mae, cv_r2).
        params: Dict of hyperparameter name -> value.
        experiment_name: MLflow experiment name.
        run_name: Display name for this run.

    Returns:
        MLflow run ID if tracking is available, otherwise None.
    """
    if not _HAS_MLFLOW:
        logger.info("Skipping MLflow logging (not installed): %s", metrics)
        return None

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        if params:
            mlflow.log_params(params)
        logger.info("MLflow run logged: %s", run.info.run_id)
        return run.info.run_id

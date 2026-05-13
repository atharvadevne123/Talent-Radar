"""MLflow experiment tracking integration for Talent-Radar."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    logger.info("mlflow not installed — experiment tracking disabled")

_DEFAULT_PARAMS = {
    "lgbm_weight": 0.4,
    "xgb_weight": 0.4,
    "rf_weight": 0.2,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "cv_folds": 5,
    "feature_count": 9,
}


def log_training_run(
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    experiment_name: str = "talent-radar-salary",
    run_name: str = "training",
    model_path: Path | None = None,
) -> str | None:
    """Log a training run to MLflow.

    Args:
        metrics: Dict of metric name -> value (e.g. cv_mae, cv_r2).
        params: Dict of hyperparameter name -> value.
        experiment_name: MLflow experiment name.
        run_name: Display name for this run.
        model_path: Optional path to log the model artifact.

    Returns:
        MLflow run ID if tracking is available, otherwise None.
    """
    if not _HAS_MLFLOW:
        logger.info("Skipping MLflow logging (not installed): %s", metrics)
        return None

    all_params = {**_DEFAULT_PARAMS, **(params or {})}
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_params({k: str(v) for k, v in all_params.items()})
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(str(model_path))
        logger.info("MLflow run logged: %s metrics=%s", run.info.run_id, metrics)
        return run.info.run_id


def get_best_run(experiment_name: str = "talent-radar-salary", metric: str = "cv_r2") -> dict[str, Any] | None:
    """Retrieve the best run from an MLflow experiment by metric.

    Args:
        experiment_name: MLflow experiment to search.
        metric: Metric to rank by (higher is better).

    Returns:
        Dictionary with run_id and metrics, or None if unavailable.
    """
    if not _HAS_MLFLOW:
        return None
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return None
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        return {"run_id": run.info.run_id, "metrics": dict(run.data.metrics)}
    except Exception:
        logger.exception("Failed to get best MLflow run")
        return None

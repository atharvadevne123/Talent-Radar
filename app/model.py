"""ML model training, evaluation, and prediction for Talent-Radar."""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_val_score

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/talent_radar.pkl"))
MODEL_VERSION = "1.0.0"

try:
    import lightgbm as lgb
    import xgboost as xgb
    _HAS_BOOSTING = True
except ImportError:
    _HAS_BOOSTING = False
    logger.warning("LightGBM/XGBoost not available — falling back to RandomForest only")


def _build_ensemble() -> Any:
    """Build LightGBM + XGBoost + RandomForest voting ensemble."""
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    if not _HAS_BOOSTING:
        return rf
    lgbm = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    xgbm = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    return VotingRegressor(
        estimators=[("lgbm", lgbm), ("xgb", xgbm), ("rf", rf)],
        weights=[0.4, 0.4, 0.2],
    )


def train(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Train the ensemble model with 5-fold CV and return metrics.

    Args:
        X: Feature DataFrame with FEATURE_COLS columns.
        y: Target salary series.

    Returns:
        Dictionary containing cv_mae, cv_r2, and model_version.
    """
    model = _build_ensemble()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1).mean()
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1).mean()

    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metrics = {"cv_mae": float(cv_mae), "cv_r2": float(cv_r2), "model_version": MODEL_VERSION}
    logger.info("Model trained: %s", metrics)
    return metrics


def load_model() -> Any:
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(X: pd.DataFrame) -> list[float]:
    """Run inference and return predicted salaries.

    Args:
        X: Feature DataFrame aligned to FEATURE_COLS.

    Returns:
        List of predicted salary values.
    """
    model = load_model()
    preds = model.predict(X)
    return [float(p) for p in preds]


def generate_synthetic_training_data(n: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic job posting data for bootstrapping the model."""
    rng = np.random.default_rng(42)
    seniority_scores = rng.integers(0, 8, n).astype(float)
    experience_years = np.clip(seniority_scores * 2.5 + rng.normal(0, 1.5, n), 0, 30)
    skill_score = rng.uniform(2.0, 15.0, n)
    skill_count = rng.integers(2, 12, n).astype(float)
    location_score = rng.choice([1.0, 1.1, 1.15, 1.4], n, p=[0.4, 0.2, 0.2, 0.2])
    industry_score = rng.choice([0.85, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], n)
    remote = rng.integers(0, 2, n).astype(float)
    exp_squared = experience_years ** 2
    exp_log = np.log1p(experience_years)

    base_salary = (
        40_000
        + seniority_scores * 18_000
        + experience_years * 3_500
        + skill_score * 2_000
        + skill_count * 500
    )
    salary = base_salary * location_score * industry_score + rng.normal(0, 5_000, n)
    salary = np.clip(salary, 30_000, 500_000)

    X = pd.DataFrame({
        "skill_score": skill_score, "skill_count": skill_count,
        "seniority_score": seniority_scores, "location_score": location_score,
        "industry_score": industry_score, "experience_years": experience_years,
        "exp_squared": exp_squared, "exp_log": exp_log, "remote": remote,
    })
    return X, pd.Series(salary, name="salary")

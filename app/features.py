"""Feature engineering pipeline for Talent-Radar salary prediction."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

SKILL_WEIGHTS: dict[str, float] = {
    "python": 1.2, "machine learning": 1.5, "deep learning": 1.6,
    "pytorch": 1.4, "tensorflow": 1.3, "sql": 1.0, "spark": 1.3,
    "kubernetes": 1.4, "aws": 1.2, "gcp": 1.2, "azure": 1.2,
    "react": 1.1, "typescript": 1.1, "go": 1.3, "rust": 1.4,
    "java": 1.0, "scala": 1.2, "data engineering": 1.3, "mlops": 1.5,
}

SENIORITY_MAP: dict[str, int] = {
    "intern": 0, "junior": 1, "mid": 2, "senior": 3,
    "staff": 4, "principal": 5, "director": 6, "vp": 7,
}

INDUSTRY_MAP: dict[str, float] = {
    "finance": 1.4, "tech": 1.3, "healthcare": 1.1, "retail": 0.9,
    "education": 0.85, "government": 0.9, "consulting": 1.2, "startup": 1.15,
    "media": 1.0, "energy": 1.1,
}


class SkillScoreTransformer(BaseEstimator, TransformerMixin):
    """Computes a weighted skill demand score from a comma-separated skills string."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SkillScoreTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["skill_score"] = X["skills"].apply(self._score_skills)
        X["skill_count"] = X["skills"].apply(
            lambda s: len([x.strip() for x in str(s).split(",") if x.strip()])
        )
        return X

    def _score_skills(self, skills_str: str) -> float:
        skills = [s.strip().lower() for s in str(skills_str).split(",")]
        return sum(SKILL_WEIGHTS.get(s, 0.8) for s in skills if s)


class SeniorityEncoder(BaseEstimator, TransformerMixin):
    """Encodes seniority strings to numeric ordinal values."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SeniorityEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["seniority_score"] = X["seniority"].apply(
            lambda s: SENIORITY_MAP.get(str(s).lower().strip(), 2)
        )
        return X


class LocationEncoder(BaseEstimator, TransformerMixin):
    """Encodes location into a cost-of-living multiplier."""

    _HIGH_COL = {"san francisco", "new york", "seattle", "boston", "washington dc"}
    _MED_COL = {"austin", "denver", "chicago", "los angeles", "atlanta"}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LocationEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["location_score"] = X["location"].apply(self._encode)
        return X

    def _encode(self, loc: str) -> float:
        loc_lower = str(loc).lower().strip()
        if any(h in loc_lower for h in self._HIGH_COL):
            return 1.4
        if any(m in loc_lower for m in self._MED_COL):
            return 1.15
        if "remote" in loc_lower:
            return 1.1
        return 1.0


class IndustryEncoder(BaseEstimator, TransformerMixin):
    """Encodes industry into a salary multiplier."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "IndustryEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["industry_score"] = X["industry"].apply(
            lambda i: INDUSTRY_MAP.get(str(i).lower().strip(), 1.0)
        )
        return X


class ExperiencePolynomialTransformer(BaseEstimator, TransformerMixin):
    """Adds squared and log experience features to capture non-linearity."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ExperiencePolynomialTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        exp = X["experience_years"].clip(0, 30).astype(float)
        X["exp_squared"] = exp ** 2
        X["exp_log"] = np.log1p(exp)
        return X


class RemoteFlagTransformer(BaseEstimator, TransformerMixin):
    """Ensures the remote flag is an integer 0/1."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RemoteFlagTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["remote"] = X["remote"].astype(int)
        return X


FEATURE_COLS = [
    "skill_score", "skill_count", "seniority_score",
    "location_score", "industry_score", "experience_years",
    "exp_squared", "exp_log", "remote",
]


def build_feature_pipeline() -> Pipeline:
    """Return a sklearn Pipeline that transforms raw job data into numeric features."""
    return Pipeline([
        ("skills", SkillScoreTransformer()),
        ("seniority", SeniorityEncoder()),
        ("location", LocationEncoder()),
        ("industry", IndustryEncoder()),
        ("experience_poly", ExperiencePolynomialTransformer()),
        ("remote_flag", RemoteFlagTransformer()),
    ])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature pipeline and return only the model feature columns."""
    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(df)
    return transformed[FEATURE_COLS]

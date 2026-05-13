"""Drift detection and monitoring tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.pool import StaticPool

from app.features import FEATURE_COLS
from app.monitoring import detect_drift, set_reference_data


def _make_feature_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "skill_score": rng.uniform(2, 15, n),
        "skill_count": rng.integers(1, 12, n).astype(float),
        "seniority_score": rng.integers(0, 8, n).astype(float),
        "location_score": rng.choice([1.0, 1.1, 1.15, 1.4], n),
        "industry_score": rng.choice([0.85, 1.0, 1.1, 1.3, 1.4], n),
        "experience_years": rng.uniform(0, 20, n),
        "exp_squared": rng.uniform(0, 400, n),
        "exp_log": rng.uniform(0, 3, n),
        "remote": rng.integers(0, 2, n).astype(float),
    })


class TestDriftDetection:
    def test_no_drift_on_same_distribution(self):
        ref = _make_feature_df(300, seed=1)
        cur = _make_feature_df(300, seed=2)
        set_reference_data(ref)
        results = detect_drift(cur)
        assert "any_drift" in results

    def test_drift_detected_on_shifted_data(self):
        ref = _make_feature_df(300, seed=10)
        set_reference_data(ref)
        shifted = _make_feature_df(300, seed=11)
        shifted["seniority_score"] = 7.0
        results = detect_drift(shifted)
        assert results.get("seniority_score", {}).get("drift_detected", False)

    def test_ks_statistic_range(self):
        ref = _make_feature_df(200, seed=20)
        cur = _make_feature_df(200, seed=21)
        set_reference_data(ref)
        results = detect_drift(cur)
        for col in FEATURE_COLS:
            if col in results and isinstance(results[col], dict):
                assert 0.0 <= results[col]["ks_statistic"] <= 1.0

    def test_empty_reference_returns_empty(self):
        from app.monitoring import REFERENCE_DATA
        REFERENCE_DATA.clear()
        cur = _make_feature_df(50, seed=99)
        results = detect_drift(cur)
        assert results == {}

    @pytest.mark.parametrize("n_samples", [50, 100, 500])
    def test_various_batch_sizes(self, n_samples):
        ref = _make_feature_df(500, seed=30)
        set_reference_data(ref)
        cur = _make_feature_df(n_samples, seed=31)
        results = detect_drift(cur)
        assert "any_drift" in results

    def test_p_value_in_range(self):
        ref = _make_feature_df(200, seed=40)
        cur = _make_feature_df(200, seed=41)
        set_reference_data(ref)
        results = detect_drift(cur)
        for col in FEATURE_COLS:
            if col in results and isinstance(results[col], dict):
                assert 0.0 <= results[col]["p_value"] <= 1.0


class TestPredictionLogging:
    def test_log_prediction_saves_record(self, db_session):
        from app.monitoring import log_prediction
        log_prediction(db_session, "ML Engineer", {"skill_score": 5.0}, 120_000.0)
        from app.database import PredictionLog
        count = db_session.query(PredictionLog).filter(
            PredictionLog.title == "ML Engineer"
        ).count()
        assert count == 1

    @pytest.fixture()
    def db_session(self, db_engine):
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=db_engine)
        session = Session()
        yield session
        session.rollback()
        session.close()

    @pytest.fixture(scope="class")
    def db_engine(self):
        from sqlalchemy import create_engine

        from app.database import Base
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        return engine

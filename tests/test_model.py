"""Model training, prediction, and evaluation tests."""
from __future__ import annotations

import pandas as pd
import pytest

from app.model import MODEL_VERSION, generate_synthetic_training_data, predict, train


@pytest.fixture(scope="module")
def trained_model_metrics():
    """Train on small dataset once per module."""
    X, y = generate_synthetic_training_data(n=300)
    return train(X, y)


@pytest.fixture(scope="module")
def val_data():
    return generate_synthetic_training_data(n=100)


class TestSyntheticData:
    def test_shape(self):
        X, y = generate_synthetic_training_data(n=50)
        assert len(X) == 50
        assert len(y) == 50

    def test_salary_range(self):
        _, y = generate_synthetic_training_data(n=200)
        assert y.min() >= 30_000
        assert y.max() <= 500_000

    def test_feature_columns_present(self):
        from app.features import FEATURE_COLS
        X, _ = generate_synthetic_training_data(n=10)
        for col in FEATURE_COLS:
            assert col in X.columns

    @pytest.mark.parametrize("n", [10, 100, 500])
    def test_various_sizes(self, n):
        X, y = generate_synthetic_training_data(n=n)
        assert len(X) == n


class TestTraining:
    def test_returns_metrics_dict(self, trained_model_metrics):
        assert isinstance(trained_model_metrics, dict)

    def test_cv_mae_positive(self, trained_model_metrics):
        assert trained_model_metrics["cv_mae"] > 0

    def test_cv_r2_reasonable(self, trained_model_metrics):
        assert trained_model_metrics["cv_r2"] > 0.5

    def test_model_version_in_metrics(self, trained_model_metrics):
        assert trained_model_metrics["model_version"] == MODEL_VERSION

    def test_model_file_created(self):
        from app.model import MODEL_PATH
        assert MODEL_PATH.exists()


class TestPrediction:
    def test_predict_returns_list(self, val_data):
        X, _ = val_data
        result = predict(X)
        assert isinstance(result, list)
        assert len(result) == len(X)

    def test_predict_all_positive(self, val_data):
        X, _ = val_data
        result = predict(X)
        assert all(p > 0 for p in result)

    def test_predict_single_row(self, sample_feature_df):
        result = predict(sample_feature_df)
        assert len(result) == 1

    def test_predict_senior_higher_than_junior(self):
        senior = pd.DataFrame([{
            "skill_score": 8.0, "skill_count": 5, "seniority_score": 3.0,
            "location_score": 1.0, "industry_score": 1.0,
            "experience_years": 7.0, "exp_squared": 49.0, "exp_log": 2.08,
            "remote": 0,
        }])
        junior = pd.DataFrame([{
            "skill_score": 3.0, "skill_count": 2, "seniority_score": 1.0,
            "location_score": 1.0, "industry_score": 1.0,
            "experience_years": 1.0, "exp_squared": 1.0, "exp_log": 0.69,
            "remote": 0,
        }])
        assert predict(senior)[0] > predict(junior)[0]

    @pytest.fixture()
    def sample_feature_df(self):
        return pd.DataFrame([{
            "skill_score": 5.2, "skill_count": 3, "seniority_score": 3,
            "location_score": 1.4, "industry_score": 1.3,
            "experience_years": 6.0, "exp_squared": 36.0, "exp_log": 1.946,
            "remote": 0,
        }])

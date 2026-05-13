"""Integration tests for the full prediction pipeline."""
from __future__ import annotations

import pandas as pd
import pytest


class TestEndToEndPipeline:
    """Test the full path: raw job dict -> features -> model -> salary."""

    @pytest.fixture(scope="class")
    def trained(self):
        from app.model import generate_synthetic_training_data, train
        X, y = generate_synthetic_training_data(n=200)
        return train(X, y)

    def test_pipeline_produces_positive_salary(self, trained):
        from app.features import prepare_features
        from app.model import predict
        row = pd.DataFrame([{
            "title": "Staff Engineer",
            "skills": "python, go, kubernetes",
            "seniority": "staff",
            "location": "San Francisco",
            "industry": "tech",
            "experience_years": 10.0,
            "remote": 0,
        }])
        features = prepare_features(row)
        salary = predict(features)[0]
        assert salary > 0

    def test_high_seniority_higher_salary_than_low(self, trained):
        from app.features import prepare_features
        from app.model import predict

        def salary_for(seniority: str, exp: float) -> float:
            row = pd.DataFrame([{
                "title": f"{seniority} eng",
                "skills": "python",
                "seniority": seniority,
                "location": "remote",
                "industry": "tech",
                "experience_years": exp,
                "remote": 1,
            }])
            return predict(prepare_features(row))[0]

        assert salary_for("senior", 8.0) > salary_for("junior", 1.0)

    @pytest.mark.parametrize("location,expected_order", [
        ("San Francisco", "new york"),
        ("new york", "remote"),
    ])
    def test_high_col_city_above_lower(self, trained, location, expected_order):
        from app.features import prepare_features
        from app.model import predict

        def salary_for(loc: str) -> float:
            row = pd.DataFrame([{
                "title": "Engineer",
                "skills": "python",
                "seniority": "mid",
                "location": loc,
                "industry": "tech",
                "experience_years": 4.0,
                "remote": 0,
            }])
            return predict(prepare_features(row))[0]

        assert salary_for(location) >= salary_for(expected_order)

    def test_tech_industry_above_education(self, trained):
        from app.features import prepare_features
        from app.model import predict

        def salary_for(industry: str) -> float:
            row = pd.DataFrame([{
                "title": "Engineer",
                "skills": "python",
                "seniority": "mid",
                "location": "remote",
                "industry": industry,
                "experience_years": 4.0,
                "remote": 1,
            }])
            return predict(prepare_features(row))[0]

        assert salary_for("tech") > salary_for("education")

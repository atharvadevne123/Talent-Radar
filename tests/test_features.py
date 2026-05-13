"""Feature engineering pipeline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    FEATURE_COLS,
    ExperiencePolynomialTransformer,
    IndustryEncoder,
    LocationEncoder,
    SeniorityEncoder,
    SkillScoreTransformer,
    build_feature_pipeline,
    prepare_features,
)


def _base_row(**kwargs) -> pd.DataFrame:
    defaults = {
        "title": "Data Engineer",
        "skills": "python, sql, spark",
        "seniority": "mid",
        "location": "remote",
        "industry": "tech",
        "experience_years": 4.0,
        "remote": 1,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


class TestSkillScoreTransformer:
    def test_known_skill_adds_score(self):
        df = pd.DataFrame([{"skills": "python, machine learning"}])
        out = SkillScoreTransformer().fit_transform(df)
        assert out["skill_score"].iloc[0] > 0

    def test_empty_skills_zero_score(self):
        df = pd.DataFrame([{"skills": ""}])
        out = SkillScoreTransformer().fit_transform(df)
        assert out["skill_score"].iloc[0] == 0.0

    def test_skill_count_correct(self):
        df = pd.DataFrame([{"skills": "python, sql, spark"}])
        out = SkillScoreTransformer().fit_transform(df)
        assert out["skill_count"].iloc[0] == 3

    @pytest.mark.parametrize("skills,expected_min", [
        ("python", 1.0),
        ("python, machine learning, deep learning", 3.5),
    ])
    def test_score_scales_with_skills(self, skills, expected_min):
        df = pd.DataFrame([{"skills": skills}])
        out = SkillScoreTransformer().fit_transform(df)
        assert out["skill_score"].iloc[0] >= expected_min


class TestSeniorityEncoder:
    @pytest.mark.parametrize("seniority,expected_score", [
        ("junior", 1), ("mid", 2), ("senior", 3), ("principal", 5),
    ])
    def test_known_seniority(self, seniority, expected_score):
        df = pd.DataFrame([{"seniority": seniority}])
        out = SeniorityEncoder().fit_transform(df)
        assert out["seniority_score"].iloc[0] == expected_score

    def test_unknown_seniority_defaults_to_mid(self):
        df = pd.DataFrame([{"seniority": "unknown_level"}])
        out = SeniorityEncoder().fit_transform(df)
        assert out["seniority_score"].iloc[0] == 2


class TestLocationEncoder:
    def test_sf_high_col(self):
        df = pd.DataFrame([{"location": "San Francisco"}])
        out = LocationEncoder().fit_transform(df)
        assert out["location_score"].iloc[0] == 1.4

    def test_remote_mid_col(self):
        df = pd.DataFrame([{"location": "remote"}])
        out = LocationEncoder().fit_transform(df)
        assert out["location_score"].iloc[0] == 1.1

    def test_unknown_location_defaults_to_one(self):
        df = pd.DataFrame([{"location": "Timbuktu"}])
        out = LocationEncoder().fit_transform(df)
        assert out["location_score"].iloc[0] == 1.0


class TestIndustryEncoder:
    @pytest.mark.parametrize("industry,expected", [
        ("finance", 1.4), ("tech", 1.3), ("education", 0.85), ("unknown", 1.0),
    ])
    def test_industry_scores(self, industry, expected):
        df = pd.DataFrame([{"industry": industry}])
        out = IndustryEncoder().fit_transform(df)
        assert out["industry_score"].iloc[0] == expected


class TestExperiencePolynomialTransformer:
    def test_adds_exp_squared(self):
        df = pd.DataFrame([{"experience_years": 4.0}])
        out = ExperiencePolynomialTransformer().fit_transform(df)
        assert abs(out["exp_squared"].iloc[0] - 16.0) < 1e-6

    def test_adds_exp_log(self):
        df = pd.DataFrame([{"experience_years": 4.0}])
        out = ExperiencePolynomialTransformer().fit_transform(df)
        assert abs(out["exp_log"].iloc[0] - np.log1p(4.0)) < 1e-6

    def test_clips_high_experience(self):
        df = pd.DataFrame([{"experience_years": 100.0}])
        out = ExperiencePolynomialTransformer().fit_transform(df)
        assert out["exp_squared"].iloc[0] == 900.0  # 30**2


class TestFullPipeline:
    def test_prepare_features_output_cols(self):
        df = _base_row()
        features = prepare_features(df)
        assert list(features.columns) == FEATURE_COLS

    def test_prepare_features_no_nulls(self):
        df = _base_row()
        features = prepare_features(df)
        assert features.isna().sum().sum() == 0

    def test_pipeline_is_sklearn_pipeline(self):
        from sklearn.pipeline import Pipeline
        p = build_feature_pipeline()
        assert isinstance(p, Pipeline)

    @pytest.mark.parametrize("exp", [0.0, 5.0, 15.0, 30.0])
    def test_experience_values(self, exp):
        df = _base_row(experience_years=exp)
        features = prepare_features(df)
        assert features["experience_years"].iloc[0] == exp

"""Pytest fixtures and test database setup for Talent-Radar."""
from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("MODEL_PATH", "/tmp/test_talent_radar.pkl")


@pytest.fixture(scope="session")
def db_engine():
    """Create an in-memory SQLite engine for tests."""
    from app.database import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def db_session(db_engine):
    """Return a transactional session that rolls back after each test."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def client():
    """Return a FastAPI TestClient with a seeded in-memory DB."""
    from app.database import Base, get_db
    from app.main import app
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def override_get_db():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_job_payload() -> dict:
    """A minimal valid job payload for API tests."""
    return {
        "title": "Senior Data Scientist",
        "skills": "python, machine learning, sql",
        "seniority": "senior",
        "location": "San Francisco",
        "industry": "tech",
        "experience_years": 6.0,
        "remote": 0,
    }


@pytest.fixture()
def sample_feature_df() -> pd.DataFrame:
    """Minimal feature DataFrame aligned to FEATURE_COLS."""
    return pd.DataFrame([{
        "skill_score": 5.2, "skill_count": 3, "seniority_score": 3,
        "location_score": 1.4, "industry_score": 1.3,
        "experience_years": 6.0, "exp_squared": 36.0, "exp_log": 1.946,
        "remote": 0,
    }])

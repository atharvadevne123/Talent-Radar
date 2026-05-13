"""Tests for SQLAlchemy models and database operations."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="module")
def mem_db():
    from app.database import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestDatabaseModels:
    def test_create_job_posting(self, mem_db):
        from app.database import JobPosting
        jp = JobPosting(
            title="Data Scientist",
            company="Acme Corp",
            location="remote",
            skills="python, sql",
            salary_min=80_000,
            salary_max=120_000,
            experience_years=4.0,
            remote=1,
            seniority="senior",
            industry="tech",
        )
        mem_db.add(jp)
        mem_db.commit()
        assert jp.id is not None

    def test_create_prediction_log(self, mem_db):
        from app.database import PredictionLog
        log = PredictionLog(
            title="ML Engineer",
            features_json='{"skill_score": 5.0}',
            predicted_salary=130_000.0,
            model_version="1.0.0",
        )
        mem_db.add(log)
        mem_db.commit()
        assert log.id is not None

    def test_create_drift_report(self, mem_db):
        from app.database import DriftReport
        report = DriftReport(
            feature_name="skill_score",
            ks_statistic=0.15,
            p_value=0.03,
            drift_detected=1,
        )
        mem_db.add(report)
        mem_db.commit()
        assert report.id is not None

    def test_query_prediction_logs(self, mem_db):
        from app.database import PredictionLog
        count = mem_db.query(PredictionLog).count()
        assert count >= 1

    def test_query_job_postings(self, mem_db):
        from app.database import JobPosting
        postings = mem_db.query(JobPosting).filter(JobPosting.title == "Data Scientist").all()
        assert len(postings) >= 1

    @pytest.mark.parametrize("industry", ["tech", "finance", "healthcare"])
    def test_job_posting_industry(self, mem_db, industry):
        from app.database import JobPosting
        jp = JobPosting(title="Eng", company="X", industry=industry, skills="python", experience_years=3.0)
        mem_db.add(jp)
        mem_db.commit()
        found = mem_db.query(JobPosting).filter(JobPosting.industry == industry).first()
        assert found is not None

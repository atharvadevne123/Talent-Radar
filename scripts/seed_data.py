"""Seed the database with synthetic job postings for development."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SAMPLE_JOBS = [
    {"title": "Senior Data Scientist", "company": "TechCorp", "location": "San Francisco",
     "skills": "python, machine learning, sql, pytorch", "salary_min": 150_000, "salary_max": 200_000,
     "experience_years": 6, "remote": 0, "seniority": "senior", "industry": "tech"},
    {"title": "ML Engineer", "company": "FinanceCo", "location": "New York",
     "skills": "python, mlops, kubernetes, aws", "salary_min": 140_000, "salary_max": 190_000,
     "experience_years": 5, "remote": 1, "seniority": "mid", "industry": "finance"},
    {"title": "Junior Data Analyst", "company": "HealthCo", "location": "remote",
     "skills": "sql, python, tableau", "salary_min": 65_000, "salary_max": 90_000,
     "experience_years": 1, "remote": 1, "seniority": "junior", "industry": "healthcare"},
    {"title": "Principal Engineer", "company": "StartupX", "location": "Austin",
     "skills": "python, rust, distributed systems, mlops", "salary_min": 200_000, "salary_max": 280_000,
     "experience_years": 14, "remote": 1, "seniority": "principal", "industry": "startup"},
    {"title": "Data Engineer", "company": "RetailCo", "location": "Chicago",
     "skills": "spark, python, sql, airflow", "salary_min": 110_000, "salary_max": 145_000,
     "experience_years": 4, "remote": 0, "seniority": "mid", "industry": "retail"},
]


def seed(n: int = 5) -> None:
    """Insert sample job postings into the database.

    Args:
        n: Number of sample jobs to insert (up to len(SAMPLE_JOBS)).
    """
    os.environ.setdefault("DATABASE_URL", "sqlite:///./talent_radar.db")
    from app.database import JobPosting, SessionLocal, init_db
    init_db()
    db = SessionLocal()
    try:
        for job_data in SAMPLE_JOBS[:n]:
            db.add(JobPosting(**job_data))
        db.commit()
        logger.info("Seeded %d job postings", min(n, len(SAMPLE_JOBS)))
        print(f"Seeded {min(n, len(SAMPLE_JOBS))} job postings.")
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed()

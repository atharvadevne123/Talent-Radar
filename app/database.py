"""SQLAlchemy models and database session management."""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./talent_radar.db")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class JobPosting(Base):
    """Represents a job posting ingested from external sources."""

    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255))
    location = Column(String(255))
    skills = Column(Text)
    salary_min = Column(Float)
    salary_max = Column(Float)
    salary_predicted = Column(Float)
    experience_years = Column(Float)
    remote = Column(Integer, default=0)
    seniority = Column(String(64))
    industry = Column(String(128))
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Log of every salary prediction made by the model."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    features_json = Column(Text)
    predicted_salary = Column(Float)
    model_version = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    """KS-test drift detection reports."""

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(128))
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    """Yield a database session, closing it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

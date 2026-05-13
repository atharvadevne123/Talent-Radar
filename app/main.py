"""FastAPI application for Talent-Radar salary prediction and job intelligence."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import __version__
from app.database import get_db, init_db
from app.features import prepare_features
from app.model import MODEL_VERSION, generate_synthetic_training_data, train
from app.monitoring import detect_drift, get_drift_summary, log_prediction, set_reference_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_START_TIME = time.time()
_PREDICTION_COUNT = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Bootstrap DB and seed model on startup."""
    init_db()
    X_ref, y_ref = generate_synthetic_training_data(500)
    set_reference_data(X_ref)
    try:
        from app.model import MODEL_PATH
        if not MODEL_PATH.exists():
            logger.info("No model found — training bootstrap model...")
            X_train, y_train = generate_synthetic_training_data(2000)
            train(X_train, y_train)
    except Exception:
        logger.exception("Bootstrap training failed")
    yield


app = FastAPI(
    title="Talent-Radar",
    description="ML platform for salary prediction and skill demand intelligence.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
    """Attach a correlation ID to every request for tracing."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


class JobInput(BaseModel):
    """Input schema for salary prediction."""

    title: str = Field(..., description="Job title")
    skills: str = Field(..., description="Comma-separated list of required skills")
    seniority: str = Field("mid", description="Seniority level: junior/mid/senior/staff/principal")
    location: str = Field("remote", description="City or 'remote'")
    industry: str = Field("tech", description="Industry sector")
    experience_years: float = Field(3.0, ge=0, le=40, description="Years of experience required")
    remote: int = Field(0, ge=0, le=1, description="1 if remote, 0 if on-site")


class SimilarJobQuery(BaseModel):
    """Input schema for RAG similarity search."""

    description: str = Field(..., description="Free-text job description")
    top_k: int = Field(5, ge=1, le=20)


@app.get("/api/v1/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Return service health and uptime."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 1), "version": __version__}


@app.get("/api/v1/version", tags=["ops"])
def version() -> dict[str, str]:
    """Return app and model version strings."""
    return {"app_version": __version__, "model_version": MODEL_VERSION}


@app.get("/api/v1/metrics", tags=["ops"])
def metrics() -> dict[str, Any]:
    """Return basic operational metrics."""
    return {
        "predictions_served": _PREDICTION_COUNT,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "model_version": MODEL_VERSION,
    }


@app.post("/api/v1/predict", tags=["prediction"])
def predict_salary(job: JobInput, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Predict salary for a single job posting."""
    global _PREDICTION_COUNT
    try:
        row = pd.DataFrame([job.model_dump()])
        features = prepare_features(row)
        from app.model import predict
        salary = predict(features)[0]
        _PREDICTION_COUNT += 1
        log_prediction(db, job.title, features.iloc[0].to_dict(), salary)
        return {
            "title": job.title,
            "predicted_salary_usd": round(salary, 2),
            "model_version": MODEL_VERSION,
            "features": features.iloc[0].to_dict(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/api/v1/predict/batch", tags=["prediction"])
def predict_batch(jobs: list[JobInput], db: Session = Depends(get_db)) -> dict[str, Any]:
    """Predict salaries for a batch of job postings (max 100)."""
    global _PREDICTION_COUNT
    if len(jobs) > 100:
        raise HTTPException(status_code=422, detail="Batch size must not exceed 100")
    try:
        rows = pd.DataFrame([j.model_dump() for j in jobs])
        features = prepare_features(rows)
        from app.model import predict
        salaries = predict(features)
        _PREDICTION_COUNT += len(salaries)
        results = [
            {"title": j.title, "predicted_salary_usd": round(s, 2)}
            for j, s in zip(jobs, salaries)
        ]
        return {"count": len(results), "predictions": results, "model_version": MODEL_VERSION}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc


@app.post("/api/v1/drift/detect", tags=["monitoring"])
def run_drift_detection(jobs: list[JobInput], db: Session = Depends(get_db)) -> dict[str, Any]:
    """Run KS-test drift detection on the provided batch of job features."""
    if not jobs:
        raise HTTPException(status_code=422, detail="No jobs provided")
    rows = pd.DataFrame([j.model_dump() for j in jobs])
    features = prepare_features(rows)
    results = detect_drift(features, db=db)
    return {"drift_results": results}


@app.get("/api/v1/drift/history", tags=["monitoring"])
def drift_history(limit: int = 50, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return recent drift detection reports."""
    return {"reports": get_drift_summary(db, limit=limit)}


@app.post("/api/v1/train", tags=["model"])
def trigger_training() -> dict[str, Any]:
    """Trigger model retraining on synthetic data (demo endpoint)."""
    try:
        X, y = generate_synthetic_training_data(2000)
        metrics = train(X, y)
        return {"status": "trained", "metrics": metrics}
    except Exception as exc:
        logger.exception("Training error")
        raise HTTPException(status_code=500, detail="Training failed") from exc

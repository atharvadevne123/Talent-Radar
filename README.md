# Talent-Radar

[![CI](https://github.com/atharvadevne123/Talent-Radar/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Talent-Radar/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ML platform for **salary prediction** and **skill demand intelligence**, powered by a LightGBM + XGBoost + RandomForest ensemble, FAISS-based job similarity search, and KS-test drift monitoring — all served through a FastAPI REST API backed by PostgreSQL.

---

## Features

- **Ensemble Salary Model** — LightGBM (40%) + XGBoost (40%) + RandomForest (20%) voting regressor
- **7-Stage Feature Pipeline** — skill scoring, seniority encoding, location CoL multiplier, industry multiplier, polynomial experience, remote flag
- **5-Fold Cross-Validation** — R2 and MAE metrics reported at training time
- **KS-Test Drift Detection** — per-feature Kolmogorov-Smirnov tests against reference distribution
- **RAG Job Similarity** — FAISS flat L2 index for bag-of-skills nearest-neighbour retrieval
- **Automated Retraining** — Airflow DAG (weekly) with extract -> drift check -> retrain -> validate
- **FastAPI REST API** — versioned endpoints (/api/v1/) with OpenAPI docs
- **Docker + PostgreSQL** — production-ready docker-compose setup
- **GitHub Actions CI** — lint (ruff), test (pytest + coverage), type-check (mypy)

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

API available at `http://localhost:8000` -- docs at `http://localhost:8000/docs`.

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=sqlite:///./talent_radar.db
uvicorn app.main:app --reload
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/v1/health` | Service health and uptime |
| `GET`  | `/api/v1/version` | App and model version |
| `GET`  | `/api/v1/metrics` | Prediction count and uptime |
| `POST` | `/api/v1/predict` | Predict salary for a single job |
| `POST` | `/api/v1/predict/batch` | Batch prediction (max 100 jobs) |
| `POST` | `/api/v1/drift/detect` | Run KS-test drift detection |
| `GET`  | `/api/v1/drift/history` | Recent drift reports |
| `POST` | `/api/v1/train` | Trigger model retraining |

### Example prediction request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior ML Engineer",
    "skills": "python, machine learning, pytorch, mlops",
    "seniority": "senior",
    "location": "San Francisco",
    "industry": "tech",
    "experience_years": 7,
    "remote": 0
  }'
```

Expected response:

```json
{
  "title": "Senior ML Engineer",
  "predicted_salary_usd": 198450.00,
  "model_version": "1.0.0"
}
```

---

## Architecture

```
[Client]
   |
[FastAPI /api/v1/]
   |         \
[ML Ensemble] [KS-Drift Monitor]
   |
[Feature Pipeline: 7 stages (sklearn)]
   |
[PostgreSQL via SQLAlchemy]

[FAISS Index]  - job similarity search
[Airflow DAG]  - weekly retraining
[CI/CD]        - GitHub Actions
```

---

## Testing

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

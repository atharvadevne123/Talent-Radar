# Changelog

All notable changes to Talent-Radar are documented here.

## [1.0.0] - 2026-05-13

### Added
- LightGBM + XGBoost + RandomForest ensemble salary prediction model
- 7-stage sklearn feature engineering pipeline
- 5-fold cross-validation with R2 and MAE metrics
- KS-test drift detection per feature column
- RAG + FAISS job similarity search (bag-of-skills vectors)
- FastAPI REST API with versioned endpoints (/api/v1/)
- Single and batch salary prediction endpoints
- Drift detection and drift history endpoints
- On-demand model retraining endpoint
- SQLAlchemy ORM with JobPosting, PredictionLog, DriftReport models
- PostgreSQL support via docker-compose
- Automated Airflow retraining DAG (weekly)
- Correlation ID middleware for request tracing
- CORS middleware
- Bootstrap model training on startup
- GitHub Actions CI (lint + test + type-check)
- Dockerfile and docker-compose.yml
- .env.example, pyproject.toml, Makefile, .pre-commit-config.yaml
- CONTRIBUTING.md, CHANGELOG.md

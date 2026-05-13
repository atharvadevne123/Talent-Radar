# Changelog

All notable changes to Talent-Radar are documented here.

## [1.0.0] - 2026-05-13

### Added

#### Core ML
- LightGBM + XGBoost + RandomForest ensemble salary prediction model (5-fold CV)
- 7-stage sklearn feature engineering pipeline (SkillScore, Seniority, Location, Industry, ExpPoly, Remote)
- 5-fold cross-validation with R2 and MAE metrics
- Synthetic training data generator for bootstrapping

#### Monitoring
- KS-test drift detection per feature column with configurable alpha
- Salary anomaly detection (Z-score + IQR combined)
- Skill demand spike detection with shifted rolling baseline
- Prediction logging to PostgreSQL via SQLAlchemy

#### Intelligence
- RAG + FAISS bag-of-skills job similarity search
- Time-series salary trend forecasting (SMA and linear trend)
- MLflow experiment tracking with metrics, params, and artifact logging

#### API (FastAPI /api/v1/)
- POST /predict — single job salary prediction
- POST /predict/batch — batch prediction (max 100)
- POST /similar — FAISS job similarity search
- POST /forecast — salary trend forecasting
- POST /drift/detect — KS-test drift detection
- GET /drift/history — recent drift reports
- POST /anomaly/salary — salary anomaly detection
- POST /train — on-demand model retraining
- GET /health, /version, /metrics — operational endpoints

#### Infrastructure
- SQLAlchemy ORM models: JobPosting, PredictionLog, DriftReport
- PostgreSQL support via docker-compose
- Rate limiting middleware (200 req/min per IP)
- Correlation ID middleware for distributed tracing
- CORS middleware
- Input validation utilities
- Automated Airflow retraining DAG (weekly)
- GitHub Actions CI: lint + test + coverage + type-check
- Dockerfile with non-root user and healthcheck
- docker-compose.yml with API and PostgreSQL

#### Developer Experience
- .env.example, pyproject.toml, Makefile, .pre-commit-config.yaml
- Alembic migration configuration
- Database seed script
- Model training CLI script
- Architecture diagram
- CONTRIBUTING.md, CHANGELOG.md, SECURITY.md, CODE_OF_CONDUCT.md
- MIT License

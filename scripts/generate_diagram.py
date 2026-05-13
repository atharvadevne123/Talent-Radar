"""Generate a text-based architecture diagram for Talent-Radar."""
from __future__ import annotations

DIAGRAM = """
Talent-Radar — Architecture
============================

  [Client / API Consumer]
          |
          v
  +-----------------+
  |   FastAPI App   |  /api/v1/predict  /api/v1/health  /api/v1/metrics
  |  (Uvicorn x2)  |  /api/v1/drift/detect  /api/v1/train
  +-----------------+
       |        |
       |        +-------> [Monitoring] KS-test drift detection
       |                       |
       v                       v
  +-----------+         [DriftReport DB table]
  |  ML Model |
  | Ensemble  |   LightGBM (40%) + XGBoost (40%) + RandomForest (20%)
  +-----------+
       |
       v
  +-----------------+
  | Feature Pipeline|   SkillScore | SeniorityEncoder | LocationEncoder
  |  (sklearn)      |   IndustryEncoder | ExpPoly | RemoteFlag
  +-----------------+
       |
       v
  +-----------------+
  |  PostgreSQL DB  |   JobPosting | PredictionLog | DriftReport
  |  (SQLAlchemy)   |
  +-----------------+

  [FAISS Index]  RAG-powered job similarity search (bag-of-skills vectors)

  [Airflow DAG]  Weekly retraining: extract -> drift check -> retrain -> validate

  [CI/CD]  GitHub Actions: lint (ruff) + test (pytest) + type-check (mypy)
"""


def main() -> None:
    import os
    os.makedirs("screenshots", exist_ok=True)
    print(DIAGRAM)
    with open("screenshots/architecture.txt", "w") as f:
        f.write(DIAGRAM)
    print("Architecture diagram written to screenshots/architecture.txt")


if __name__ == "__main__":
    main()

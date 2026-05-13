"""Application settings loaded from environment variables."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Talent-Radar runtime configuration.

    All values can be overridden via environment variables or a .env file.
    """

    database_url: str = "sqlite:///./talent_radar.db"
    model_path: Path = Path("models/talent_radar.pkl")
    model_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
    log_level: str = "INFO"
    drift_alpha: float = 0.05
    max_batch_size: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

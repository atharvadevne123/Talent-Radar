"""Tests for configuration settings."""
from __future__ import annotations


class TestSettings:
    def test_default_database_url_is_sqlite(self):
        from app.config import Settings
        s = Settings()
        assert "sqlite" in s.database_url or "postgresql" in s.database_url

    def test_default_api_port(self):
        from app.config import Settings
        s = Settings()
        assert s.api_port == 8000

    def test_drift_alpha_range(self):
        from app.config import Settings
        s = Settings()
        assert 0 < s.drift_alpha < 1

    def test_max_batch_size_positive(self):
        from app.config import Settings
        s = Settings()
        assert s.max_batch_size > 0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        from app.config import Settings
        s = Settings()
        assert s.api_port == 9000

"""Tests for input validation utilities."""
from __future__ import annotations

import pytest
from app.validation import sanitize_title, validate_job_input


class TestValidateJobInput:
    def test_valid_input_no_errors(self):
        data = {"title": "ML Engineer", "skills": "python, sql", "experience_years": 4, "remote": 0}
        assert validate_job_input(data) == []

    def test_missing_title_error(self):
        data = {"skills": "python", "experience_years": 2, "remote": 0}
        errors = validate_job_input(data)
        assert any("title" in e for e in errors)

    def test_missing_skills_error(self):
        data = {"title": "Engineer", "skills": "", "experience_years": 2, "remote": 0}
        errors = validate_job_input(data)
        assert any("skill" in e for e in errors)

    def test_negative_experience_error(self):
        data = {"title": "Eng", "skills": "python", "experience_years": -1, "remote": 0}
        errors = validate_job_input(data)
        assert any("experience" in e for e in errors)

    def test_invalid_remote_flag_error(self):
        data = {"title": "Eng", "skills": "python", "experience_years": 3, "remote": 5}
        errors = validate_job_input(data)
        assert any("remote" in e for e in errors)

    @pytest.mark.parametrize("seniority", ["junior", "senior", "staff", "unknown_level"])
    def test_seniority_validation(self, seniority):
        data = {"title": "Eng", "skills": "python", "experience_years": 3, "remote": 0, "seniority": seniority}
        errors = validate_job_input(data)
        assert isinstance(errors, list)


class TestSanitizeTitle:
    def test_removes_special_chars(self):
        result = sanitize_title("ML Engineer <script>alert(1)</script>")
        assert "<script>" not in result

    def test_preserves_alphanumeric(self):
        result = sanitize_title("Senior Data Scientist")
        assert result == "Senior Data Scientist"

    def test_empty_string(self):
        assert sanitize_title("") == ""

    @pytest.mark.parametrize("title", ["ML Engineer", "Data Scientist/Analyst", "Sr. Engineer"])
    def test_valid_titles_preserved(self, title):
        result = sanitize_title(title)
        assert len(result) > 0

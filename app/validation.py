"""Input validation utilities for Talent-Radar API."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

VALID_SENIORITY_LEVELS = {"intern", "junior", "mid", "senior", "staff", "principal", "director", "vp"}
VALID_INDUSTRIES = {"tech", "finance", "healthcare", "retail", "education", "government", "consulting", "startup", "media", "energy"}
MAX_SKILLS_COUNT = 20
MAX_TITLE_LENGTH = 200


def validate_job_input(data: dict[str, Any]) -> list[str]:
    """Validate a raw job input dictionary and return a list of error messages.

    Args:
        data: Raw job input fields as a dictionary.

    Returns:
        List of validation error strings (empty if valid).
    """
    errors: list[str] = []

    title = str(data.get("title", "")).strip()
    if not title:
        errors.append("title is required")
    elif len(title) > MAX_TITLE_LENGTH:
        errors.append(f"title exceeds {MAX_TITLE_LENGTH} characters")

    skills_str = str(data.get("skills", "")).strip()
    if not skills_str:
        errors.append("skills are required")
    else:
        skills = [s.strip() for s in skills_str.split(",") if s.strip()]
        if len(skills) > MAX_SKILLS_COUNT:
            errors.append(f"too many skills (max {MAX_SKILLS_COUNT})")

    seniority = str(data.get("seniority", "mid")).lower().strip()
    if seniority not in VALID_SENIORITY_LEVELS:
        logger.warning("Unknown seniority level: %s (will use default)", seniority)

    experience = data.get("experience_years", 0)
    try:
        exp_float = float(experience)
        if exp_float < 0 or exp_float > 60:
            errors.append("experience_years must be between 0 and 60")
    except (TypeError, ValueError):
        errors.append("experience_years must be a number")

    remote = data.get("remote", 0)
    if remote not in (0, 1):
        errors.append("remote must be 0 or 1")

    return errors


def sanitize_title(title: str) -> str:
    """Remove special characters from a job title for safe storage.

    Args:
        title: Raw job title string.

    Returns:
        Sanitized title with only alphanumeric chars, spaces, hyphens, and slashes.
    """
    return re.sub(r"[^a-zA-Z0-9 \-/&.,()]", "", title).strip()

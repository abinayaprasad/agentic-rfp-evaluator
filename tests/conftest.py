"""Shared fixtures: the five CTMS criteria exactly as seeded in db/seed.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def criteria() -> list[dict]:
    return [
        {"criterion_id": 1, "name": "Technical Capability", "weight": 30, "max_score": 10, "is_active": 1,
         "description": "Data model, named eTMF/EDC/SAP integrations, reporting, scalability."},
        {"criterion_id": 2, "name": "Implementation Plan", "weight": 20, "max_score": 10, "is_active": 1,
         "description": "Migration waves, validation phases, training, hypercare, risks."},
        {"criterion_id": 3, "name": "Commercial Value", "weight": 20, "max_score": 10, "is_active": 1,
         "description": "Licence model, implementation fee, five-year TCO, assumptions."},
        {"criterion_id": 4, "name": "Security & Compliance", "weight": 20, "max_score": 10, "is_active": 1,
         "description": "21 CFR Part 11, GxP validation, GDPR, ISO 27001, access control."},
        {"criterion_id": 5, "name": "Support & Experience", "weight": 10, "max_score": 10, "is_active": 1,
         "description": "SLAs, hypercare, pharma references, CTMS go-lives."},
    ]


@pytest.fixture
def sample_rfp_dir() -> Path:
    return PROJECT_ROOT / "data" / "sample_rfps"

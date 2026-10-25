"""
Create the SQLite database and seed the CTMS evaluation criteria.

Usage (from the project root):
    py -3.10 -m db.seed            # create tables + seed if empty
    py -3.10 -m db.seed --reset    # drop everything and reseed

The Streamlit app also calls ``seed_if_empty()`` on startup so a fresh
deployment on Streamlit Community Cloud always has criteria available.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from db.database import DB_PATH, SCHEMA_PATH, get_connection, init_db, resolve_db_path

# ---------------------------------------------------------------------------
# CTMS (Clinical Trial Management System) evaluation criteria.
# Active weights must total 100.
# ---------------------------------------------------------------------------
SEED_CRITERIA: list[dict] = [
    {
        "name": "Technical Capability",
        "weight": 30,
        "max_score": 10,
        "description": (
            "Study, site and country data model; named integrations with the eTMF, "
            "EDC and SAP finance systems; reporting dashboards; multi-country "
            "configuration; scalability to 40+ concurrent studies."
        ),
    },
    {
        "name": "Implementation Plan",
        "weight": 20,
        "max_score": 10,
        "description": (
            "Migration waves and milestones; legacy data migration strategy; "
            "computer system validation (CSV) phases; user training; hypercare; "
            "risk register; realism of the 12-month timeline."
        ),
    },
    {
        "name": "Commercial Value",
        "weight": 20,
        "max_score": 10,
        "description": (
            "Licence model (per study or per user); implementation fee; validation "
            "services fee; five-year total cost of ownership; clarity and "
            "completeness of pricing assumptions."
        ),
    },
    {
        "name": "Security & Compliance",
        "weight": 20,
        "max_score": 10,
        "description": (
            "21 CFR Part 11 audit trail and electronic signatures; GxP validation "
            "package (IQ/OQ/PQ); GDPR and data residency; ISO 27001 or SOC 2; HIPAA; "
            "access control and auditability."
        ),
    },
    {
        "name": "Support & Experience",
        "weight": 10,
        "max_score": 10,
        "description": (
            "Support model and SLA tiers; 24/7 coverage; hypercare duration; named "
            "pharma or CRO references; number of CTMS go-lives; regulatory "
            "inspection history."
        ),
    },
]


def _validate_seed() -> None:
    total = sum(c["weight"] for c in SEED_CRITERIA)
    if abs(total - 100) > 0.01:
        raise ValueError(f"Seed criteria weights must total 100, got {total}")


def seed_criteria(conn: sqlite3.Connection) -> int:
    """Insert seed criteria. Returns number of rows inserted."""
    _validate_seed()
    cur = conn.executemany(
        """
        INSERT INTO evaluation_criteria (name, description, weight, max_score, is_active)
        VALUES (:name, :description, :weight, :max_score, 1)
        """,
        SEED_CRITERIA,
    )
    conn.commit()
    return cur.rowcount


def seed_if_empty(db_path: Path | None = None) -> bool:
    """Create tables if needed and seed criteria only when the table is empty.

    Returns True if seeding happened.
    """
    db_path = resolve_db_path(db_path)
    init_db(db_path)
    with get_connection(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0]
        if count == 0:
            seed_criteria(conn)
            return True
    return False


def reset_database(db_path: Path | None = None) -> None:
    """Delete the database file and recreate it with seed data."""
    db_path = resolve_db_path(db_path)
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)
    with get_connection(db_path) as conn:
        seed_criteria(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and seed the RFP evaluator database.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the database.")
    args = parser.parse_args()

    if args.reset:
        reset_database()
        print(f"Database reset and seeded at {DB_PATH}")
    else:
        seeded = seed_if_empty()
        msg = "seeded" if seeded else "already had criteria; nothing inserted"
        print(f"Database at {DB_PATH} {msg}")

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT criterion_id, name, weight, max_score, is_active "
            "FROM evaluation_criteria ORDER BY criterion_id"
        ).fetchall()
    total = sum(r["weight"] for r in rows if r["is_active"])
    print(f"\n{'ID':<4}{'Criterion':<26}{'Weight':>8}{'Max':>6}{'Active':>8}")
    for r in rows:
        print(f"{r['criterion_id']:<4}{r['name']:<26}{r['weight']:>8.1f}{r['max_score']:>6}{r['is_active']:>8}")
    print(f"\nActive weight total: {total:.1f}  (schema: {SCHEMA_PATH.name})")


if __name__ == "__main__":
    main()

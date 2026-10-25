"""
SQLite access layer for the Agentic RFP Evaluator.

Every function opens its own short-lived connection (via ``get_connection``),
which keeps Streamlit reruns simple and avoids sharing a connection across
threads. All rows are returned as plain dicts so the rest of the code never
depends on sqlite3.Row.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "rfp_evaluator.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

WEIGHT_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Connection and setup
# ---------------------------------------------------------------------------
def resolve_db_path(db_path: Path | None = None) -> Path:
    """Return the explicit path, else the module-level DB_PATH (read at call time
    so tests and the app can repoint it)."""
    return Path(db_path) if db_path is not None else DB_PATH


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection with Row factory and foreign keys enabled."""
    db_path = resolve_db_path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    """Create tables from schema.sql if they do not exist."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(resolve_db_path(db_path)) as conn:
        conn.executescript(schema_sql)
        conn.commit()


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def get_all_criteria() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_criteria ORDER BY criterion_id"
        ).fetchall()
    return _rows_to_dicts(rows)


def get_active_criteria() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_criteria WHERE is_active = 1 ORDER BY criterion_id"
        ).fetchall()
    return _rows_to_dicts(rows)


def active_weight_total(criteria: list[dict[str, Any]] | None = None) -> float:
    """Sum of weights of active criteria (pass a list to avoid a DB hit)."""
    if criteria is None:
        criteria = get_active_criteria()
    return float(sum(c["weight"] for c in criteria if c.get("is_active", 1)))


def weights_are_valid(criteria: list[dict[str, Any]] | None = None) -> bool:
    return abs(active_weight_total(criteria) - 100.0) <= WEIGHT_TOLERANCE


def update_criterion(criterion_id: int, *, weight: float | None = None,
                     is_active: bool | None = None,
                     description: str | None = None) -> None:
    """Update weight, active flag and/or description for one criterion."""
    fields: list[str] = []
    params: list[Any] = []
    if weight is not None:
        fields.append("weight = ?")
        params.append(float(weight))
    if is_active is not None:
        fields.append("is_active = ?")
        params.append(1 if is_active else 0)
    if description is not None:
        fields.append("description = ?")
        params.append(description)
    if not fields:
        return
    fields.append("updated_at = datetime('now')")
    params.append(criterion_id)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE evaluation_criteria SET {', '.join(fields)} WHERE criterion_id = ?",
            params,
        )
        conn.commit()


def bulk_update_criteria(updates: list[dict[str, Any]]) -> None:
    """Apply many updates in one transaction.

    Each item: {"criterion_id": int, "weight": float, "is_active": bool}
    """
    with get_connection() as conn:
        conn.executemany(
            """
            UPDATE evaluation_criteria
               SET weight = :weight,
                   is_active = :is_active,
                   updated_at = datetime('now')
             WHERE criterion_id = :criterion_id
            """,
            [
                {
                    "criterion_id": u["criterion_id"],
                    "weight": float(u["weight"]),
                    "is_active": 1 if u["is_active"] else 0,
                }
                for u in updates
            ],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def create_run(rfp_run_id: str, criteria_snapshot: list[dict[str, Any]],
               llm_model: str | None) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rfp_runs (rfp_run_id, status, criteria_json, llm_model)
            VALUES (?, 'RUNNING', ?, ?)
            """,
            (rfp_run_id, json.dumps(criteria_snapshot), llm_model),
        )
        conn.commit()


def complete_run(rfp_run_id: str, warnings: list[str]) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE rfp_runs SET status = 'COMPLETED', warnings_json = ?
             WHERE rfp_run_id = ?
            """,
            (json.dumps(warnings), rfp_run_id),
        )
        conn.commit()


def fail_run(rfp_run_id: str, error_message: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE rfp_runs SET status = 'FAILED', error_message = ?
             WHERE rfp_run_id = ?
            """,
            (error_message[:2000], rfp_run_id),
        )
        conn.commit()


def save_supplier_results(rfp_run_id: str, results: list[dict[str, Any]]) -> None:
    """Insert one row per supplier.

    Each result dict must contain: supplier_name, submission_date,
    experience_rating, absolute_score, ppi, final_rank, and the full
    per-supplier detail which is stored as result_json.
    """
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO supplier_results
                (rfp_run_id, supplier_name, submission_date, experience_rating,
                 absolute_score, ppi, final_rank, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    rfp_run_id,
                    r["supplier_name"],
                    r["submission_date"],
                    float(r["experience_rating"]),
                    float(r["absolute_score"]),
                    float(r["ppi"]),
                    int(r["final_rank"]),
                    json.dumps(r, default=str),
                )
                for r in results
            ],
        )
        conn.commit()


def get_run(rfp_run_id: str) -> dict[str, Any] | None:
    """Return run header plus ordered supplier results, with JSON fields parsed."""
    with get_connection() as conn:
        run = conn.execute(
            "SELECT * FROM rfp_runs WHERE rfp_run_id = ?", (rfp_run_id,)
        ).fetchone()
        if run is None:
            return None
        suppliers = conn.execute(
            """
            SELECT * FROM supplier_results
             WHERE rfp_run_id = ?
             ORDER BY final_rank
            """,
            (rfp_run_id,),
        ).fetchall()

    run_d = dict(run)
    run_d["criteria"] = json.loads(run_d.pop("criteria_json") or "[]")
    run_d["warnings"] = json.loads(run_d.pop("warnings_json") or "[]")
    run_d["suppliers"] = [json.loads(s["result_json"]) for s in suppliers]
    return run_d


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT r.rfp_run_id, r.created_at, r.status, r.llm_model,
                   COUNT(s.result_id) AS supplier_count
              FROM rfp_runs r
              LEFT JOIN supplier_results s ON s.rfp_run_id = r.rfp_run_id
             GROUP BY r.rfp_run_id
             ORDER BY r.created_at DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Proposal chunks (retrieval-augmented Q&A)
# ---------------------------------------------------------------------------
def save_chunks(rfp_run_id: str, chunks: list[dict[str, Any]]) -> None:
    """Store text chunks for a run. Each chunk: supplier_name, chunk_id, text, char_start."""
    if not chunks:
        return
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO proposal_chunks
                (rfp_run_id, supplier_name, chunk_id, text, char_start)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (rfp_run_id, c["supplier_name"], int(c["chunk_id"]), c["text"], int(c.get("char_start", 0)))
                for c in chunks
            ],
        )
        conn.commit()


def get_chunks(rfp_run_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT supplier_name, chunk_id, text, char_start
              FROM proposal_chunks
             WHERE rfp_run_id = ?
             ORDER BY supplier_name, chunk_id
            """,
            (rfp_run_id,),
        ).fetchall()
    return _rows_to_dicts(rows)


def count_chunks(rfp_run_id: str) -> int:
    with get_connection() as conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM proposal_chunks WHERE rfp_run_id = ?", (rfp_run_id,)
        ).fetchone()[0])


# ---------------------------------------------------------------------------
# Human-in-the-loop: score adjustments
# ---------------------------------------------------------------------------
def save_adjustment(rfp_run_id: str, supplier_name: str, criterion_id: int,
                    original_score: int, adjusted_score: int, reason: str, reviewer: str) -> None:
    """Insert or replace the current adjustment for one (run, supplier, criterion)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO score_adjustments
                (rfp_run_id, supplier_name, criterion_id, original_score, adjusted_score, reason, reviewer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rfp_run_id, supplier_name, criterion_id) DO UPDATE SET
                original_score = excluded.original_score,
                adjusted_score = excluded.adjusted_score,
                reason         = excluded.reason,
                reviewer       = excluded.reviewer,
                created_at     = datetime('now')
            """,
            (rfp_run_id, supplier_name, int(criterion_id), int(original_score), int(adjusted_score),
             reason.strip(), reviewer.strip()),
        )
        conn.commit()


def delete_adjustment(rfp_run_id: str, supplier_name: str, criterion_id: int) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM score_adjustments WHERE rfp_run_id = ? AND supplier_name = ? AND criterion_id = ?",
            (rfp_run_id, supplier_name, int(criterion_id)),
        )
        conn.commit()
        return cur.rowcount


def get_adjustments(rfp_run_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT supplier_name, criterion_id, original_score, adjusted_score, reason, reviewer, created_at
              FROM score_adjustments
             WHERE rfp_run_id = ?
             ORDER BY supplier_name, criterion_id
            """,
            (rfp_run_id,),
        ).fetchall()
    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Human-in-the-loop: award decisions (versioned)
# ---------------------------------------------------------------------------
def save_decision(rfp_run_id: str, *, basis: str, recommended_supplier: str, selected_supplier: str,
                  reviewer: str, reason_category: str | None, reason_text: str | None,
                  adjustments: list[dict[str, Any]]) -> dict[str, Any]:
    """Append a new decision version and return it."""
    is_override = 1 if selected_supplier.strip().lower() != recommended_supplier.strip().lower() else 0
    with get_connection() as conn:
        version = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM award_decisions WHERE rfp_run_id = ?",
            (rfp_run_id,),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO award_decisions
                (rfp_run_id, version, basis, recommended_supplier, selected_supplier, is_override,
                 reason_category, reason_text, reviewer, adjustments_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (rfp_run_id, int(version), basis, recommended_supplier, selected_supplier, is_override,
             (reason_category or None), (reason_text.strip() if reason_text else None), reviewer.strip(),
             json.dumps(adjustments, default=str)),
        )
        conn.commit()
    return get_latest_decision(rfp_run_id)  # type: ignore[return-value]


def _decision_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["is_override"] = bool(d["is_override"])
    d["adjustments"] = json.loads(d.pop("adjustments_json") or "[]")
    return d


def get_latest_decision(rfp_run_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM award_decisions WHERE rfp_run_id = ? ORDER BY version DESC LIMIT 1",
            (rfp_run_id,),
        ).fetchone()
    return _decision_row_to_dict(row) if row else None


def get_decision_history(rfp_run_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM award_decisions WHERE rfp_run_id = ? ORDER BY version DESC",
            (rfp_run_id,),
        ).fetchall()
    return [_decision_row_to_dict(r) for r in rows]

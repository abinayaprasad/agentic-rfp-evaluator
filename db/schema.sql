-- Agentic RFP Evaluation - SQLite schema
-- Three tables as required by the project brief (section 6), with a few
-- extra columns that make runs reproducible and explainable.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Evaluation criteria. Editable from the Streamlit "Criteria" screen.
-- Weights of ACTIVE criteria must total 100 (enforced in Python, not SQL,
-- because the check spans multiple rows).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    description    TEXT    NOT NULL,
    weight         REAL    NOT NULL CHECK (weight >= 0 AND weight <= 100),
    max_score      INTEGER NOT NULL DEFAULT 10 CHECK (max_score > 0),
    is_active      INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- One row per evaluation batch. criteria_json stores a snapshot of the
-- active criteria used for that run so later weight changes never alter
-- a stored result.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id     TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    status         TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    criteria_json  TEXT NOT NULL,
    warnings_json  TEXT,
    llm_model      TEXT,
    error_message  TEXT
);

-- ---------------------------------------------------------------------------
-- One row per supplier per run. result_json holds the complete per-criterion
-- detail (score, benchmark, gap, relative %, evidence, justification, warnings).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS supplier_results (
    result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id        TEXT    NOT NULL REFERENCES rfp_runs(rfp_run_id) ON DELETE CASCADE,
    supplier_name     TEXT    NOT NULL,
    submission_date   TEXT    NOT NULL,
    experience_rating REAL    NOT NULL CHECK (experience_rating >= 0 AND experience_rating <= 10),
    absolute_score    REAL    NOT NULL,
    ppi               REAL    NOT NULL,
    final_rank        INTEGER NOT NULL,
    result_json       TEXT    NOT NULL,
    UNIQUE (rfp_run_id, supplier_name)
);

CREATE INDEX IF NOT EXISTS idx_supplier_results_run
    ON supplier_results (rfp_run_id, final_rank);

-- ---------------------------------------------------------------------------
-- Proposal text chunks for retrieval-augmented Q&A ("Ask the Proposals").
-- One row per chunk per supplier per run; removed with the run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS proposal_chunks (
    chunk_pk       INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id     TEXT    NOT NULL REFERENCES rfp_runs(rfp_run_id) ON DELETE CASCADE,
    supplier_name  TEXT    NOT NULL,
    chunk_id       INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    char_start     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (rfp_run_id, supplier_name, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_proposal_chunks_run
    ON proposal_chunks (rfp_run_id, supplier_name, chunk_id);

-- ---------------------------------------------------------------------------
-- Human-in-the-loop review.
-- score_adjustments: a reviewer may change one LLM criterion score, with a
-- reason. The original score is kept on the row; the LLM result_json is never
-- modified. One current adjustment per (run, supplier, criterion).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS score_adjustments (
    adjustment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id      TEXT    NOT NULL REFERENCES rfp_runs(rfp_run_id) ON DELETE CASCADE,
    supplier_name   TEXT    NOT NULL,
    criterion_id    INTEGER NOT NULL,
    original_score  INTEGER NOT NULL,
    adjusted_score  INTEGER NOT NULL,
    reason          TEXT    NOT NULL,
    reviewer        TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (rfp_run_id, supplier_name, criterion_id)
);

-- award_decisions: the final human selection for a run. Every save creates a
-- new version so earlier decisions remain visible. is_override = 1 when the
-- selected supplier is not the recommended (rank 1) supplier.
CREATE TABLE IF NOT EXISTS award_decisions (
    decision_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id           TEXT    NOT NULL REFERENCES rfp_runs(rfp_run_id) ON DELETE CASCADE,
    version              INTEGER NOT NULL,
    basis                TEXT    NOT NULL CHECK (basis IN ('llm', 'adjusted')),
    recommended_supplier TEXT    NOT NULL,
    selected_supplier    TEXT    NOT NULL,
    is_override          INTEGER NOT NULL CHECK (is_override IN (0, 1)),
    reason_category      TEXT,
    reason_text          TEXT,
    reviewer             TEXT    NOT NULL,
    decided_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    adjustments_json     TEXT    NOT NULL DEFAULT '[]',
    UNIQUE (rfp_run_id, version)
);

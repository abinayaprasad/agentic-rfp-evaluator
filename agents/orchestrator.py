"""
Orchestrator Agent - runs the ten-step evaluation workflow and calls each
tool in order. Contains no formula logic and no prompt text of its own.

    1  Setup       (criteria loaded by caller / UI)
    2  Input       uploads + metadata arrive as ``uploads``
    3  Batch       generate RFP_RUN_ID, snapshot criteria, create run row
    4  Evaluate    Document Tool -> Evaluation Agent, per supplier
    5  Validate    Validation Tool, per supplier
    6-8 Score/Benchmark/Rank   Ranking Tool (score_and_rank)
    9  Persist     supplier_results + run completion
    10 Present     return the result dict used by the UI and JSON download

Per-supplier failures (unreadable PDF, LLM error, bad JSON) never abort the
batch: that supplier receives an all-zero evaluation plus warnings and the
others are still ranked.
"""
from __future__ import annotations

import json
import secrets
from datetime import date, datetime, timezone
from typing import Any, Callable

from agents.evaluator import LLMClient, evaluate_supplier, make_client
from db import database as db
from tools.document_tool import (
    DocumentExtractionError,
    extract_text,
    truncate_for_prompt,
)
from tools.ranking_tool import explain_tie_breaks, score_and_rank
from tools.retrieval_tool import chunk_text
from tools.validation_tool import empty_evaluation, validate_and_normalise

ProgressCallback = Callable[[int, int, str], None]


class BatchValidationError(ValueError):
    """Raised before any LLM call when the inputs cannot be evaluated."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"RFP-{now.strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}"


def _iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)[:10]).isoformat()


def validate_uploads(uploads: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable input problems (empty when valid)."""
    problems: list[str] = []
    if not uploads:
        problems.append("Upload at least one supplier PDF.")
        return problems
    names_seen: set[str] = set()
    for i, u in enumerate(uploads, start=1):
        name = str(u.get("supplier_name") or "").strip()
        if not name:
            problems.append(f"Supplier #{i}: supplier name is required.")
        elif name.lower() in names_seen:
            problems.append(f"Supplier #{i}: duplicate supplier name '{name}'.")
        else:
            names_seen.add(name.lower())
        try:
            _iso_date(u.get("submission_date"))
        except (TypeError, ValueError):
            problems.append(f"Supplier #{i} ({name or 'unnamed'}): submission date is missing or invalid.")
        try:
            rating = float(u.get("experience_rating"))
            if not 0 <= rating <= 10:
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"Supplier #{i} ({name or 'unnamed'}): experience rating must be a number from 0 to 10.")
        if not u.get("file_bytes"):
            problems.append(f"Supplier #{i} ({name or 'unnamed'}): PDF file is empty or missing.")
    return problems


def _noop(_step: int, _total: int, _msg: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_evaluation_batch(uploads: list[dict[str, Any]],
                         progress_callback: ProgressCallback | None = None,
                         llm_client: LLMClient | None = None) -> dict[str, Any]:
    """Evaluate every upload against the active criteria and persist the run.

    ``uploads`` items: {file_bytes, filename, supplier_name, submission_date, experience_rating}

    Returns the complete result dict (also what the UI offers as JSON download).
    Raises BatchValidationError for bad inputs or invalid criteria weights.
    """
    progress = progress_callback or _noop
    log: list[str] = []
    run_warnings: list[str] = []

    def step(n: int, total: int, msg: str) -> None:
        log.append(msg)
        try:
            progress(n, total, msg)
        except Exception:  # a UI progress callback must never abort the workflow
            pass

    # --- validate inputs ------------------------------------------------------
    problems = validate_uploads(uploads)
    if problems:
        raise BatchValidationError(" ".join(problems))

    criteria = db.get_active_criteria()
    if not criteria:
        raise BatchValidationError("No active evaluation criteria. Activate at least one criterion.")
    if not db.weights_are_valid(criteria):
        raise BatchValidationError(
            f"Active criteria weights total {db.active_weight_total(criteria):.2f}; they must total 100.")

    client = llm_client or make_client()
    n_suppliers = len(uploads)
    total_steps = n_suppliers + 3  # batch + per-supplier + rank + persist

    # --- step 3: batch --------------------------------------------------------
    run_id = generate_run_id()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.create_run(run_id, criteria, client.model_name)

    # --- steps 4-5: evaluate + validate per supplier --------------------------
    entries: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []   # for retrieval-augmented Q&A
    try:
        step(1, total_steps, f"Created run {run_id} with {len(criteria)} active criteria")
        for i, u in enumerate(uploads, start=1):
            name = str(u["supplier_name"]).strip()
            warnings: list[str] = []
            doc_meta: dict[str, Any] = {}
            evaluation: dict[str, Any]

            # Document Tool
            try:
                doc = extract_text(u["file_bytes"], u.get("filename") or f"{name}.pdf")
                doc_meta = {"page_count": doc["page_count"], "char_count": doc["char_count"],
                            "filename": doc["filename"]}
                readable = doc["is_readable"]
                if not readable:
                    warnings.append("PDF_TEXT_TOO_SHORT")
            except DocumentExtractionError as exc:
                doc = {"text": ""}
                readable = False
                warnings.append("PDF_EXTRACTION_FAILED")
                doc_meta = {"filename": u.get("filename"), "error": str(exc)}

            if readable:
                supplier_chunks = chunk_text(doc["text"], name)
                all_chunks.extend(supplier_chunks)
                doc_meta["chunk_count"] = len(supplier_chunks)

            if not readable:
                reason = ("No readable text could be extracted from the PDF; "
                          "scored 0 on every criterion.")
                evaluation = empty_evaluation(name, criteria, reason)
                llm_info = {"attempts": 0, "model": client.model_name, "skipped": True}
            else:
                # Evaluation Agent
                text, truncated = truncate_for_prompt(doc["text"])
                if truncated:
                    warnings.append("PDF_TEXT_TRUNCATED")
                call = evaluate_supplier(name, text, criteria, client=client)
                warnings.extend(call.warnings)
                llm_info = {"attempts": call.attempts, "model": call.model, "error": call.error}
                if call.raw is None:
                    reason = call.error or "LLM evaluation failed"
                    evaluation = empty_evaluation(name, criteria, reason)
                else:
                    # Validation Tool
                    evaluation, v_warnings = validate_and_normalise(call.raw, name, criteria)
                    warnings.extend(v_warnings)

            entries.append({
                "supplier_name": name,
                "submission_date": _iso_date(u["submission_date"]),
                "experience_rating": float(u["experience_rating"]),
                "evaluation": evaluation,
                "warnings": warnings,
                "document": doc_meta,
                "llm": llm_info,
            })
            step(1 + i, total_steps,
                 f"Evaluated {name} ({len(warnings)} warning{'s' if len(warnings) != 1 else ''})")

        # --- steps 6-8: score, benchmark, rank ---------------------------------
        ranked = score_and_rank(entries, criteria)
        extras = {e["supplier_name"]: e for e in entries}
        for s in ranked["suppliers"]:
            s["document"] = extras[s["supplier_name"]]["document"]
            s["llm"] = extras[s["supplier_name"]]["llm"]
        tie_break_explanation = explain_tie_breaks(ranked["suppliers"])
        step(n_suppliers + 2, total_steps, "Computed scores, benchmarks, PPI and ranks")

        for s in ranked["suppliers"]:
            for w in s["warnings"]:
                run_warnings.append(f"{s['supplier_name']}: {w}")

        # --- step 9: persist ----------------------------------------------------
        db.save_supplier_results(run_id, ranked["suppliers"])
        db.save_chunks(run_id, all_chunks)
        db.complete_run(run_id, run_warnings)
        step(n_suppliers + 3, total_steps,
             f"Saved results and {len(all_chunks)} text chunks to SQLite")

    except Exception as exc:  # unexpected failure: mark run and re-raise
        db.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise

    # --- step 10: present -----------------------------------------------------
    return {
        "rfp_run_id": run_id,
        "created_at": created_at,
        "status": "COMPLETED",
        "llm_model": client.model_name,
        "criteria": [
            {"criterion_id": int(c["criterion_id"]), "name": c["name"], "weight": float(c["weight"]),
             "max_score": int(c["max_score"]), "description": c["description"]}
            for c in criteria
        ],
        "benchmarks": {str(k): v for k, v in ranked["benchmarks"].items()},
        "suppliers": ranked["suppliers"],
        "tie_break_explanation": tie_break_explanation,
        "run_warnings": run_warnings,
        "workflow_log": log,
    }


def result_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)


def load_run_result(rfp_run_id: str) -> dict[str, Any] | None:
    """Rebuild a result dict (same shape as run_evaluation_batch) from SQLite."""
    stored = db.get_run(rfp_run_id)
    if stored is None:
        return None
    suppliers = stored["suppliers"]
    benchmarks: dict[str, int] = {}
    for s in suppliers:
        for c in s.get("criteria", []):
            benchmarks[str(c["criterion_id"])] = int(c.get("benchmark", 0))
    return {
        "rfp_run_id": stored["rfp_run_id"],
        "created_at": stored["created_at"],
        "status": stored["status"],
        "llm_model": stored.get("llm_model"),
        "criteria": stored["criteria"],
        "benchmarks": benchmarks,
        "suppliers": suppliers,
        "tie_break_explanation": explain_tie_breaks(suppliers) if suppliers else [],
        "run_warnings": stored["warnings"],
        "workflow_log": [],
        "error_message": stored.get("error_message"),
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop: attach adjustments and the award decision to a result
# ---------------------------------------------------------------------------
def attach_human_review(result: dict[str, Any]) -> dict[str, Any]:
    """Load adjustments and the latest decision for ``result`` and return the
    result with a ``human_review`` section:

        human_review = {
          "reviewed": bool,
          "adjustments": [...],
          "adjusted_ranking": [ {supplier_name, final_rank, llm_rank, rank_change, absolute_score, ppi,
                                 llm_absolute_score, llm_ppi, criteria:[...]} ],
          "recommended_supplier": str,          # rank 1 after adjustments
          "decision": {...} | None,
          "decision_history": [...],
          "summary": [str, ...],
        }

    The original LLM ranking in ``result["suppliers"]`` is left untouched.
    """
    from tools.review_tool import apply_adjustments, recommended_supplier, summarize_review

    run_id = result["rfp_run_id"]
    adjustments = db.get_adjustments(run_id)
    adjusted = apply_adjustments(result, adjustments)
    decision = db.get_latest_decision(run_id)
    history = db.get_decision_history(run_id)
    out = dict(result)
    out["human_review"] = {
        "reviewed": bool(adjustments or decision),
        "adjustments": adjustments,
        "adjusted_ranking": adjusted["suppliers"],
        "adjusted_benchmarks": adjusted["benchmarks"],
        "adjusted_tie_break_explanation": adjusted["tie_break_explanation"],
        "recommended_supplier": recommended_supplier(adjusted),
        "decision": decision,
        "decision_history": history,
        "summary": summarize_review(adjusted, decision),
    }
    return out

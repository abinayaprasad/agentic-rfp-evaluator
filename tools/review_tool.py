"""
Review Tool - human-in-the-loop logic. Deterministic Python, no LLM, no I/O.

Two capabilities:

1. Score adjustments. A reviewer may replace one LLM criterion score with a
   reason. ``apply_adjustments`` returns a NEW ranked result computed from the
   adjusted scores using the same Ranking Tool; the original LLM result is
   untouched and every adjusted criterion carries its original score, the
   reason and the reviewer.

2. Award decision. ``validate_decision`` enforces the governance rules: a
   reviewer name is always required; selecting any supplier other than the
   recommended (rank 1) supplier is an override and requires a reason
   category and free-text justification.
"""
from __future__ import annotations

import copy
from typing import Any

from tools.ranking_tool import explain_tie_breaks, score_and_rank

OVERRIDE_CATEGORIES: list[str] = [
    "Strategic relationship / incumbent advantage",
    "Risk appetite (delivery or compliance risk judged differently)",
    "Budget constraint",
    "Reference check outcome",
    "Timeline pressure",
    "Scope or requirement clarification after scoring",
    "Other (explain)",
]

MIN_REASON_CHARS = 10


class ReviewValidationError(ValueError):
    """Raised for an invalid adjustment or decision."""


# ---------------------------------------------------------------------------
# Score adjustments
# ---------------------------------------------------------------------------
def validate_adjustment(result: dict[str, Any], supplier_name: str, criterion_id: int,
                        adjusted_score: Any, reason: str, reviewer: str) -> dict[str, Any]:
    """Check an adjustment against the run and return the matching criterion row."""
    if not (reviewer or "").strip():
        raise ReviewValidationError("Reviewer name is required.")
    if len((reason or "").strip()) < MIN_REASON_CHARS:
        raise ReviewValidationError(f"A reason of at least {MIN_REASON_CHARS} characters is required.")
    supplier = next((s for s in result["suppliers"] if s["supplier_name"] == supplier_name), None)
    if supplier is None:
        raise ReviewValidationError(f"Unknown supplier '{supplier_name}' for this run.")
    crit = next((c for c in supplier["criteria"] if int(c["criterion_id"]) == int(criterion_id)), None)
    if crit is None:
        raise ReviewValidationError(f"Criterion {criterion_id} is not part of this run.")
    try:
        score = int(adjusted_score)
    except (TypeError, ValueError):
        raise ReviewValidationError("Adjusted score must be a whole number.") from None
    if not 0 <= score <= int(crit["max_score"]):
        raise ReviewValidationError(f"Adjusted score must be between 0 and {crit['max_score']}.")
    if score == int(crit["score"]):
        raise ReviewValidationError("Adjusted score equals the LLM score; nothing to change.")
    return crit


def _entries_from_result(result: dict[str, Any],
                         adjustments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild Ranking Tool entries from a stored result, substituting adjusted scores."""
    adj_map = {(a["supplier_name"], int(a["criterion_id"])): a for a in adjustments}
    entries = []
    for s in result["suppliers"]:
        crit_rows = []
        for c in s["criteria"]:
            key = (s["supplier_name"], int(c["criterion_id"]))
            a = adj_map.get(key)
            crit_rows.append({
                "criterion_id": int(c["criterion_id"]),
                "score": int(a["adjusted_score"]) if a else int(c["score"]),
                "max_score": int(c["max_score"]),
                "justification": c.get("justification", ""),
                "evidence": c.get("evidence", ""),
            })
        entries.append({
            "supplier_name": s["supplier_name"],
            "submission_date": s["submission_date"],
            "experience_rating": s["experience_rating"],
            "evaluation": {"criteria": crit_rows, "risks": s.get("risks", []),
                           "overall_summary": s.get("overall_summary", "")},
            "warnings": list(s.get("warnings", [])),
        })
    return entries


def apply_adjustments(result: dict[str, Any],
                      adjustments: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a new result dict re-scored and re-ranked with the adjustments applied.

    The input ``result`` is not mutated. Each adjusted criterion in the output
    carries ``original_score``, ``adjusted`` = True, ``adjustment_reason`` and
    ``adjusted_by``. Each supplier carries ``llm_rank`` (its rank before
    adjustments) and ``rank_change``.
    """
    criteria = result["criteria"]
    if not adjustments:
        out = copy.deepcopy(result)
        for s in out["suppliers"]:
            s["llm_rank"] = s["final_rank"]
            s["rank_change"] = 0
            for c in s["criteria"]:
                c["original_score"] = c["score"]
                c["adjusted"] = False
        out["adjustments"] = []
        return out

    ranked = score_and_rank(_entries_from_result(result, adjustments), criteria)
    adj_map = {(a["supplier_name"], int(a["criterion_id"])): a for a in adjustments}
    original = {s["supplier_name"]: s for s in result["suppliers"]}

    for s in ranked["suppliers"]:
        o = original[s["supplier_name"]]
        s["llm_rank"] = o["final_rank"]
        s["rank_change"] = o["final_rank"] - s["final_rank"]   # positive = moved up
        s["llm_absolute_score"] = o["absolute_score"]
        s["llm_ppi"] = o["ppi"]
        s["document"] = o.get("document", {})
        s["llm"] = o.get("llm", {})
        for c in s["criteria"]:
            a = adj_map.get((s["supplier_name"], int(c["criterion_id"])))
            if a:
                c["original_score"] = int(a["original_score"])
                c["adjusted"] = True
                c["adjustment_reason"] = a["reason"]
                c["adjusted_by"] = a["reviewer"]
            else:
                c["original_score"] = c["score"]
                c["adjusted"] = False

    out = copy.deepcopy(result)
    out["suppliers"] = ranked["suppliers"]
    out["benchmarks"] = {str(k): v for k, v in ranked["benchmarks"].items()}
    out["tie_break_explanation"] = explain_tie_breaks(ranked["suppliers"])
    out["adjustments"] = copy.deepcopy(adjustments)
    return out


def recommended_supplier(result: dict[str, Any]) -> str:
    """Rank-1 supplier of a (possibly adjusted) result."""
    return min(result["suppliers"], key=lambda s: s["final_rank"])["supplier_name"]


# ---------------------------------------------------------------------------
# Award decision
# ---------------------------------------------------------------------------
def validate_decision(result: dict[str, Any], selected_supplier: str, reviewer: str,
                      reason_category: str | None, reason_text: str | None) -> dict[str, Any]:
    """Validate a decision and return a normalised dict ready to persist.

    Returns {recommended_supplier, selected_supplier, is_override, reason_category, reason_text, reviewer}.
    """
    if not (reviewer or "").strip():
        raise ReviewValidationError("Reviewer name is required.")
    names = [s["supplier_name"] for s in result["suppliers"]]
    if selected_supplier not in names:
        raise ReviewValidationError(f"'{selected_supplier}' is not a supplier in this run.")
    rec = recommended_supplier(result)
    is_override = selected_supplier != rec
    category = (reason_category or "").strip() or None
    text = (reason_text or "").strip() or None
    if is_override:
        if category not in OVERRIDE_CATEGORIES:
            raise ReviewValidationError("Select an override reason category.")
        if not text or len(text) < MIN_REASON_CHARS:
            raise ReviewValidationError(
                f"An override justification of at least {MIN_REASON_CHARS} characters is required.")
    return {
        "recommended_supplier": rec,
        "selected_supplier": selected_supplier,
        "is_override": is_override,
        "reason_category": category if is_override else None,
        "reason_text": text,
        "reviewer": reviewer.strip(),
    }


def summarize_review(adjusted: dict[str, Any], decision: dict[str, Any] | None) -> list[str]:
    """Plain-language lines for the Run Details screen and the export."""
    lines: list[str] = []
    adjustments = adjusted.get("adjustments", [])
    if adjustments:
        lines.append(f"{len(adjustments)} score adjustment(s) applied by a reviewer.")
        moved = [s for s in adjusted["suppliers"] if s.get("rank_change")]
        if moved:
            lines.append("Rank changes after adjustments: " + "; ".join(
                f"{s['supplier_name']} #{s['llm_rank']} -> #{s['final_rank']}" for s in moved) + ".")
        else:
            lines.append("Adjustments did not change the ranking order.")
    else:
        lines.append("No score adjustments; ranking is the LLM-based ranking.")
    if decision:
        if decision["is_override"]:
            lines.append(
                f"OVERRIDE: {decision['reviewer']} selected {decision['selected_supplier']} instead of the "
                f"recommended {decision['recommended_supplier']} ({decision['reason_category']}): "
                f"{decision['reason_text']}")
        else:
            lines.append(f"{decision['reviewer']} accepted the recommended supplier "
                         f"{decision['selected_supplier']} (decision v{decision['version']}).")
    else:
        lines.append("No award decision recorded yet.")
    return lines

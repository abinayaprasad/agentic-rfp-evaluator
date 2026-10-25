"""
Ranking Tool - deterministic scoring, peer benchmarking and ranking.

Pure Python. No LLM, no I/O, no randomness: the same validated inputs always
produce the same numbers and the same order.

Formulas (BUILD_PLAN.md section 7):
    absolute_score  = sum((score / max_score) * weight)             range 0-100
    benchmark[c]    = max(score for that criterion across suppliers)
    gap             = score - benchmark                               0 or negative
    relative_pct    = score / benchmark * 100   (benchmark 0 -> 100 if score 0 else 0)
    ppi             = sum(relative_pct * weight) / sum(weight)       range 0-100

Tie-break order (mandatory):
    1. higher PPI  2. earlier submission date  3. higher experience rating  4. name A-Z
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

PPI_DECIMALS = 4          # precision used for comparisons and storage
DISPLAY_DECIMALS = 2


# ---------------------------------------------------------------------------
# Elementary formulas
# ---------------------------------------------------------------------------
def relative_performance_pct(score: float, benchmark: float) -> float:
    if benchmark == 0:
        return 100.0 if score == 0 else 0.0
    return round(score / benchmark * 100.0, PPI_DECIMALS)


def compute_absolute_score(criteria_results: list[dict[str, Any]],
                           criteria: list[dict[str, Any]]) -> float:
    """Sum of (score / max_score) * weight over the active criteria.

    Weights and max_score come from ``criteria`` (the DB snapshot), not from
    the model output.
    """
    meta = {int(c["criterion_id"]): c for c in criteria}
    total = 0.0
    for r in criteria_results:
        c = meta.get(int(r["criterion_id"]))
        if c is None:
            continue
        max_score = float(c["max_score"])
        if max_score <= 0:
            continue
        total += (float(r["score"]) / max_score) * float(c["weight"])
    return round(total, PPI_DECIMALS)


def compute_benchmarks(all_evaluations: list[dict[str, Any]],
                       criteria: list[dict[str, Any]]) -> dict[int, int]:
    """Highest validated score per criterion across all suppliers."""
    benchmarks = {int(c["criterion_id"]): 0 for c in criteria}
    for ev in all_evaluations:
        for r in ev["criteria"]:
            cid = int(r["criterion_id"])
            if cid in benchmarks:
                benchmarks[cid] = max(benchmarks[cid], int(r["score"]))
    return benchmarks


def compute_ppi(criteria_results: list[dict[str, Any]],
                criteria: list[dict[str, Any]]) -> float:
    """Weighted average of relative_pct values already present on the results."""
    meta = {int(c["criterion_id"]): float(c["weight"]) for c in criteria}
    weighted = 0.0
    weight_sum = 0.0
    for r in criteria_results:
        w = meta.get(int(r["criterion_id"]))
        if w is None:
            continue
        weighted += float(r["relative_pct"]) * w
        weight_sum += w
    if weight_sum == 0:
        return 0.0
    return round(weighted / weight_sum, PPI_DECIMALS)


# ---------------------------------------------------------------------------
# Sorting / ranking
# ---------------------------------------------------------------------------
def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def sort_key(result: dict[str, Any]) -> tuple:
    return (
        -round(float(result["ppi"]), PPI_DECIMALS),
        _parse_date(result["submission_date"]),
        -float(result["experience_rating"]),
        str(result["supplier_name"]).strip().lower(),
    )


def _separating_rule(upper: dict[str, Any], lower: dict[str, Any]) -> str:
    """Explain which rule placed ``upper`` above ``lower``."""
    u_ppi, l_ppi = round(float(upper["ppi"]), PPI_DECIMALS), round(float(lower["ppi"]), PPI_DECIMALS)
    if u_ppi != l_ppi:
        return (f"Rule 1 (PPI): {upper['supplier_name']} {u_ppi:.{DISPLAY_DECIMALS}f} "
                f"vs {lower['supplier_name']} {l_ppi:.{DISPLAY_DECIMALS}f}")
    u_date, l_date = _parse_date(upper["submission_date"]), _parse_date(lower["submission_date"])
    if u_date != l_date:
        return (f"Rule 2 (earlier submission): {upper['supplier_name']} {u_date.isoformat()} "
                f"vs {lower['supplier_name']} {l_date.isoformat()} (PPI tied at {u_ppi:.{DISPLAY_DECIMALS}f})")
    u_exp, l_exp = float(upper["experience_rating"]), float(lower["experience_rating"])
    if u_exp != l_exp:
        return (f"Rule 3 (experience rating): {upper['supplier_name']} {u_exp:g} "
                f"vs {lower['supplier_name']} {l_exp:g} (PPI and date tied)")
    return (f"Rule 4 (name A-Z): '{upper['supplier_name']}' before '{lower['supplier_name']}' "
            f"(PPI, date and rating tied)")


def rank_suppliers(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new list sorted by the tie-break order with final_rank and
    tie_break_reason added. Input dicts are not mutated."""
    ranked = [dict(r) for r in sorted(results, key=sort_key)]
    for i, r in enumerate(ranked):
        r["final_rank"] = i + 1
        if i == 0:
            r["tie_break_reason"] = f"Ranked first on PPI {float(r['ppi']):.{DISPLAY_DECIMALS}f}"
        else:
            r["tie_break_reason"] = _separating_rule(ranked[i - 1], r)
    return ranked


def explain_tie_breaks(ranked: list[dict[str, Any]]) -> list[str]:
    return [f"#{r['final_rank']} {r['supplier_name']}: {r['tie_break_reason']}" for r in ranked]


# ---------------------------------------------------------------------------
# Full pipeline: validated evaluations -> ranked results
# ---------------------------------------------------------------------------
def score_and_rank(entries: list[dict[str, Any]],
                   criteria: list[dict[str, Any]]) -> dict[str, Any]:
    """Run steps 6-8 of the workflow.

    ``entries`` items must contain:
        supplier_name, submission_date, experience_rating,
        evaluation (normalised dict from validation_tool), warnings (list)

    Returns {"benchmarks": {...}, "suppliers": [ranked result dicts]}.
    """
    meta = {int(c["criterion_id"]): c for c in criteria}
    evaluations = [e["evaluation"] for e in entries]
    benchmarks = compute_benchmarks(evaluations, criteria)

    results: list[dict[str, Any]] = []
    for e in entries:
        ev = e["evaluation"]
        detailed: list[dict[str, Any]] = []
        for r in ev["criteria"]:
            cid = int(r["criterion_id"])
            c = meta.get(cid)
            if c is None:
                continue
            bench = benchmarks[cid]
            score = int(r["score"])
            detailed.append({
                "criterion_id": cid,
                "name": c["name"],
                "weight": float(c["weight"]),
                "max_score": int(c["max_score"]),
                "score": score,
                "benchmark": bench,
                "gap": score - bench,
                "relative_pct": relative_performance_pct(score, bench),
                "justification": r["justification"],
                "evidence": r["evidence"],
            })
        results.append({
            "supplier_name": e["supplier_name"],
            "submission_date": _parse_date(e["submission_date"]).isoformat(),
            "experience_rating": float(e["experience_rating"]),
            "absolute_score": compute_absolute_score(detailed, criteria),
            "ppi": compute_ppi(detailed, criteria),
            "criteria": detailed,
            "risks": list(ev.get("risks", [])),
            "overall_summary": ev.get("overall_summary", ""),
            "warnings": list(e.get("warnings", [])),
        })

    ranked = rank_suppliers(results)
    return {"benchmarks": benchmarks, "suppliers": ranked}

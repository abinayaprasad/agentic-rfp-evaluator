"""
Validation Tool - checks and normalises the raw LLM evaluation.

Guarantees: ``validate_and_normalise`` never raises and always returns a
``SupplierEvaluation``-shaped dict containing exactly one entry for every
active criterion, with integer scores inside [0, max_score]. Every repair is
recorded as a warning code so the UI can show what was changed.

Warning codes (see BUILD_PLAN.md section 9):
    LLM_INVALID_JSON            raw payload missing or not an object
    MISSING_CRITERION:<id>      active criterion absent -> filled with score 0
    UNKNOWN_CRITERION:<id>      id not in active set -> dropped
    DUPLICATE_CRITERION:<id>    repeated id -> first kept
    SCORE_CLIPPED_HIGH:<id>     score > max_score -> clipped
    SCORE_CLIPPED_LOW:<id>      score < 0 -> clipped
    SCORE_NON_NUMERIC:<id>      score not a number -> 0
    SCORE_ROUNDED:<id>          non-integer score -> rounded half-up
    MAX_SCORE_OVERRIDDEN:<id>   model max_score != DB -> DB value used
    EMPTY_TEXT_FIELD:<id>       empty justification/evidence -> placeholder
    SUPPLIER_NAME_MISMATCH      model name != user input -> user input used
    RISKS_NOT_LIST              risks field malformed -> []
    SUMMARY_MISSING             overall_summary missing -> placeholder
    PDF_TEXT_TOO_SHORT          raised by orchestrator when extraction fails
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

PLACEHOLDER_TEXT = "Not provided"
MISSING_JUSTIFICATION = "Not returned by model"


# ---------------------------------------------------------------------------
# Strict output models - what the rest of the pipeline can rely on
# ---------------------------------------------------------------------------
class CriterionResult(BaseModel):
    criterion_id: int
    score: int = Field(ge=0)
    max_score: int = Field(gt=0)
    justification: str
    evidence: str

    @field_validator("justification", "evidence")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return v if v and v.strip() else PLACEHOLDER_TEXT


class SupplierEvaluation(BaseModel):
    supplier_name: str
    criteria: list[CriterionResult]
    risks: list[str] = Field(default_factory=list)
    overall_summary: str = PLACEHOLDER_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_number(value: Any) -> float | None:
    """Return a float for ints, floats and numeric strings; None otherwise."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            f = float(value.strip())
            return f if math.isfinite(f) else None
        except ValueError:
            return None
    return None


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _normalise_score(raw_score: Any, max_score: int, cid: int, warnings: list[str]) -> int:
    num = _to_number(raw_score)
    if num is None:
        warnings.append(f"SCORE_NON_NUMERIC:{cid}")
        return 0
    if num != math.floor(num):
        warnings.append(f"SCORE_ROUNDED:{cid}")
        num = _round_half_up(num)
    score = int(num)
    if score > max_score:
        warnings.append(f"SCORE_CLIPPED_HIGH:{cid}")
        score = max_score
    elif score < 0:
        warnings.append(f"SCORE_CLIPPED_LOW:{cid}")
        score = 0
    return score


def empty_evaluation(supplier_name: str, criteria: list[dict[str, Any]],
                     reason: str) -> dict[str, Any]:
    """An all-zero evaluation used when the document or LLM call failed."""
    return SupplierEvaluation(
        supplier_name=supplier_name,
        criteria=[
            CriterionResult(
                criterion_id=int(c["criterion_id"]),
                score=0,
                max_score=int(c["max_score"]),
                justification=reason,
                evidence=PLACEHOLDER_TEXT,
            )
            for c in criteria
        ],
        risks=[reason],
        overall_summary=reason,
    ).model_dump()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def validate_and_normalise(raw: Any, supplier_name: str,
                           criteria: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Normalise ``raw`` (parsed LLM JSON) against the active ``criteria``.

    Returns (evaluation_dict, warnings). Never raises.
    """
    warnings: list[str] = []
    active: dict[int, dict[str, Any]] = {int(c["criterion_id"]): c for c in criteria}

    if not isinstance(raw, dict):
        warnings.append("LLM_INVALID_JSON")
        return empty_evaluation(supplier_name, criteria, "Model response was not valid JSON"), warnings

    # --- supplier name ------------------------------------------------------
    model_name = _to_text(raw.get("supplier_name"))
    if model_name.lower() != supplier_name.strip().lower():
        warnings.append("SUPPLIER_NAME_MISMATCH")

    # --- criteria -----------------------------------------------------------
    raw_criteria = raw.get("criteria")
    if not isinstance(raw_criteria, list):
        raw_criteria = []
        warnings.append("LLM_INVALID_JSON")

    seen: dict[int, dict[str, Any]] = {}
    for item in raw_criteria:
        if not isinstance(item, dict):
            continue
        cid_num = _to_number(item.get("criterion_id"))
        if cid_num is None:
            continue
        cid = int(cid_num)
        if cid not in active:
            warnings.append(f"UNKNOWN_CRITERION:{cid}")
            continue
        if cid in seen:
            warnings.append(f"DUPLICATE_CRITERION:{cid}")
            continue

        db_max = int(active[cid]["max_score"])
        model_max = _to_number(item.get("max_score"))
        if model_max is not None and int(model_max) != db_max:
            warnings.append(f"MAX_SCORE_OVERRIDDEN:{cid}")

        score = _normalise_score(item.get("score"), db_max, cid, warnings)

        justification = _to_text(item.get("justification"))
        evidence = _to_text(item.get("evidence"))
        if not justification or not evidence:
            warnings.append(f"EMPTY_TEXT_FIELD:{cid}")

        seen[cid] = {
            "criterion_id": cid,
            "score": score,
            "max_score": db_max,
            "justification": justification or PLACEHOLDER_TEXT,
            "evidence": evidence or PLACEHOLDER_TEXT,
        }

    ordered: list[dict[str, Any]] = []
    for cid, c in active.items():
        if cid in seen:
            ordered.append(seen[cid])
        else:
            warnings.append(f"MISSING_CRITERION:{cid}")
            ordered.append({
                "criterion_id": cid,
                "score": 0,
                "max_score": int(c["max_score"]),
                "justification": MISSING_JUSTIFICATION,
                "evidence": PLACEHOLDER_TEXT,
            })

    # --- risks and summary --------------------------------------------------
    risks_raw = raw.get("risks", [])
    if isinstance(risks_raw, str):
        risks = [risks_raw.strip()] if risks_raw.strip() else []
    elif isinstance(risks_raw, list):
        risks = [_to_text(r) for r in risks_raw if _to_text(r)]
    else:
        risks = []
        warnings.append("RISKS_NOT_LIST")

    summary = _to_text(raw.get("overall_summary"))
    if not summary:
        warnings.append("SUMMARY_MISSING")
        summary = PLACEHOLDER_TEXT

    evaluation = SupplierEvaluation(
        supplier_name=supplier_name.strip(),
        criteria=[CriterionResult(**c) for c in ordered],
        risks=risks,
        overall_summary=summary,
    )
    return evaluation.model_dump(), warnings

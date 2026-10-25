from __future__ import annotations

import pytest

from agents import orchestrator
from agents.evaluator import MockLLMClient
from db import database as db
from db.seed import seed_if_empty
from tools.review_tool import (
    MIN_REASON_CHARS,
    OVERRIDE_CATEGORIES,
    ReviewValidationError,
    apply_adjustments,
    recommended_supplier,
    summarize_review,
    validate_adjustment,
    validate_decision,
)


# ---------------------------------------------------------------------------
# Fixtures: a small ranked result built through the real ranking tool
# ---------------------------------------------------------------------------
def _result(criteria) -> dict:
    from tools.ranking_tool import explain_tie_breaks, score_and_rank

    def entry(name, scores, date, rating):
        return {"supplier_name": name, "submission_date": date, "experience_rating": rating,
                "evaluation": {"criteria": [{"criterion_id": i + 1, "score": s, "max_score": 10,
                                             "justification": "j", "evidence": "e"} for i, s in enumerate(scores)],
                               "risks": [], "overall_summary": ""},
                "warnings": []}

    ranked = score_and_rank([
        entry("Apex Systems", [9, 7, 6, 9, 7], "2026-10-01", 8.5),     # abs 78, PPI 90.48 -> #1
        entry("NexaWorks", [7, 9, 7, 7, 9], "2026-10-03", 7.0),         # abs 76, PPI 88.89 -> #2
        entry("BrightPath Tech", [5, 5, 7, 4, 4], "2026-09-28", 5.0),   # -> #3
    ], criteria)
    return {"rfp_run_id": "RFP-TEST", "created_at": "2026-10-25T00:00:00", "status": "COMPLETED",
            "llm_model": "mock", "criteria": criteria,
            "benchmarks": {str(k): v for k, v in ranked["benchmarks"].items()},
            "suppliers": ranked["suppliers"],
            "tie_break_explanation": explain_tie_breaks(ranked["suppliers"]),
            "run_warnings": [], "workflow_log": []}


@pytest.fixture
def result(criteria):
    return _result(criteria)


# ---------------------------------------------------------------------------
# validate_adjustment
# ---------------------------------------------------------------------------
def test_validate_adjustment_ok(result):
    crit = validate_adjustment(result, "NexaWorks", 1, 9, "Integration detail stronger than scored", "Abi")
    assert crit["criterion_id"] == 1 and crit["score"] == 7


@pytest.mark.parametrize("kwargs,msg", [
    (dict(supplier_name="NexaWorks", criterion_id=1, adjusted_score=9, reason="x" * 20, reviewer=""), "Reviewer"),
    (dict(supplier_name="NexaWorks", criterion_id=1, adjusted_score=9, reason="short", reviewer="A"), "reason"),
    (dict(supplier_name="Ghost", criterion_id=1, adjusted_score=9, reason="x" * 20, reviewer="A"), "Unknown supplier"),
    (dict(supplier_name="NexaWorks", criterion_id=99, adjusted_score=9, reason="x" * 20, reviewer="A"), "not part"),
    (dict(supplier_name="NexaWorks", criterion_id=1, adjusted_score=11, reason="x" * 20, reviewer="A"), "between 0"),
    (dict(supplier_name="NexaWorks", criterion_id=1, adjusted_score="high", reason="x" * 20, reviewer="A"), "whole number"),
    (dict(supplier_name="NexaWorks", criterion_id=1, adjusted_score=7, reason="x" * 20, reviewer="A"), "equals"),
])
def test_validate_adjustment_rejects(result, kwargs, msg):
    with pytest.raises(ReviewValidationError, match=msg):
        validate_adjustment(result, **kwargs)


# ---------------------------------------------------------------------------
# apply_adjustments
# ---------------------------------------------------------------------------
def test_no_adjustments_returns_equivalent_ranking_with_flags(result):
    out = apply_adjustments(result, [])
    assert [s["supplier_name"] for s in out["suppliers"]] == [s["supplier_name"] for s in result["suppliers"]]
    assert all(s["rank_change"] == 0 and s["llm_rank"] == s["final_rank"] for s in out["suppliers"])
    assert all(c["adjusted"] is False and c["original_score"] == c["score"]
               for s in out["suppliers"] for c in s["criteria"])
    assert out["adjustments"] == []


def test_adjustment_rerank_and_traceability(result):
    adj = [{"supplier_name": "NexaWorks", "criterion_id": 1, "original_score": 7, "adjusted_score": 10,
            "reason": "Named integrations verified in appendix", "reviewer": "Abi"}]
    out = apply_adjustments(result, adj)

    # NexaWorks: 1.0*30 + 0.9*20 + 0.7*20 + 0.7*20 + 0.9*10 = 85 -> now rank 1
    nexa = next(s for s in out["suppliers"] if s["supplier_name"] == "NexaWorks")
    apex = next(s for s in out["suppliers"] if s["supplier_name"] == "Apex Systems")
    assert nexa["final_rank"] == 1 and nexa["llm_rank"] == 2 and nexa["rank_change"] == 1
    assert apex["final_rank"] == 2 and apex["llm_rank"] == 1 and apex["rank_change"] == -1
    assert nexa["absolute_score"] == pytest.approx(85.0)
    assert nexa["llm_absolute_score"] == pytest.approx(76.0)

    c1 = next(c for c in nexa["criteria"] if c["criterion_id"] == 1)
    assert c1["score"] == 10 and c1["original_score"] == 7 and c1["adjusted"] is True
    assert c1["adjustment_reason"].startswith("Named integrations") and c1["adjusted_by"] == "Abi"
    assert all(c["adjusted"] is False for c in apex["criteria"])

    # benchmarks recomputed from adjusted scores
    assert out["benchmarks"]["1"] == 10
    assert len(out["tie_break_explanation"]) == 3
    assert out["adjustments"] == adj


def test_apply_adjustments_does_not_mutate_input(result):
    before = [(s["supplier_name"], s["final_rank"], [c["score"] for c in s["criteria"]]) for s in result["suppliers"]]
    apply_adjustments(result, [{"supplier_name": "Apex Systems", "criterion_id": 1, "original_score": 9,
                                "adjusted_score": 2, "reason": "x" * 12, "reviewer": "A"}])
    after = [(s["supplier_name"], s["final_rank"], [c["score"] for c in s["criteria"]]) for s in result["suppliers"]]
    assert before == after
    assert "adjustments" not in result


def test_recommended_supplier(result):
    assert recommended_supplier(result) == "Apex Systems"
    out = apply_adjustments(result, [{"supplier_name": "NexaWorks", "criterion_id": 1, "original_score": 7,
                                      "adjusted_score": 10, "reason": "x" * 12, "reviewer": "A"}])
    assert recommended_supplier(out) == "NexaWorks"


# ---------------------------------------------------------------------------
# validate_decision
# ---------------------------------------------------------------------------
def test_accepting_recommendation_needs_no_reason(result):
    d = validate_decision(result, "Apex Systems", "Abi", None, None)
    assert d["is_override"] is False and d["reason_category"] is None
    assert d["recommended_supplier"] == "Apex Systems"


def test_override_requires_category_and_text(result):
    with pytest.raises(ReviewValidationError, match="category"):
        validate_decision(result, "NexaWorks", "Abi", None, "x" * 20)
    with pytest.raises(ReviewValidationError, match="justification"):
        validate_decision(result, "NexaWorks", "Abi", OVERRIDE_CATEGORIES[0], "short")
    d = validate_decision(result, "NexaWorks", "Abi", OVERRIDE_CATEGORIES[0],
                          "Existing SAP integration partner; lower integration risk.")
    assert d["is_override"] is True and d["reason_category"] == OVERRIDE_CATEGORIES[0]


def test_decision_requires_reviewer_and_known_supplier(result):
    with pytest.raises(ReviewValidationError, match="Reviewer"):
        validate_decision(result, "Apex Systems", " ", None, None)
    with pytest.raises(ReviewValidationError, match="not a supplier"):
        validate_decision(result, "Ghost", "Abi", None, None)


def test_summarize_review_lines(result):
    out = apply_adjustments(result, [])
    lines = summarize_review(out, None)
    assert lines[0].startswith("No score adjustments")
    assert lines[-1].startswith("No award decision")
    decision = {"is_override": True, "reviewer": "Abi", "selected_supplier": "NexaWorks",
                "recommended_supplier": "Apex Systems", "reason_category": "Budget constraint",
                "reason_text": "Cheaper over five years.", "version": 1}
    assert any(l.startswith("OVERRIDE: Abi selected NexaWorks") for l in summarize_review(out, decision))


# ---------------------------------------------------------------------------
# Persistence + attach_human_review (temp DB, mock LLM)
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "review.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    seed_if_empty(path)
    return path


def _run(sample_rfp_dir):
    files = {"Apex Systems": "apex_systems.pdf", "NexaWorks": "nexaworks.pdf", "Orbit Digital": "orbit_digital.pdf"}
    uploads = []
    for name, f in files.items():
        p = sample_rfp_dir / f
        if not p.exists():
            pytest.skip("run scripts/generate_rfps.py first")
        uploads.append({"file_bytes": p.read_bytes(), "filename": f, "supplier_name": name,
                        "submission_date": "2026-10-01", "experience_rating": 7.0})
    return orchestrator.run_evaluation_batch(uploads, llm_client=MockLLMClient())


def test_adjustment_persistence_roundtrip(temp_db, sample_rfp_dir):
    res = _run(sample_rfp_dir)
    rid = res["rfp_run_id"]
    db.save_adjustment(rid, "Orbit Digital", 1, 5, 9, "Verified integration list with vendor", "Abi")
    db.save_adjustment(rid, "Orbit Digital", 1, 5, 8, "Revised after second look", "Abi")  # replace
    adjs = db.get_adjustments(rid)
    assert len(adjs) == 1 and adjs[0]["adjusted_score"] == 8 and adjs[0]["reason"].startswith("Revised")
    assert db.delete_adjustment(rid, "Orbit Digital", 1) == 1
    assert db.get_adjustments(rid) == []


def test_decision_versions_and_attach(temp_db, sample_rfp_dir):
    res = _run(sample_rfp_dir)
    rid = res["rfp_run_id"]

    view = orchestrator.attach_human_review(res)
    hr = view["human_review"]
    assert hr["reviewed"] is False and hr["decision"] is None
    assert hr["recommended_supplier"] == res["suppliers"][0]["supplier_name"]
    assert [s["supplier_name"] for s in hr["adjusted_ranking"]] == [s["supplier_name"] for s in res["suppliers"]]

    rec = hr["recommended_supplier"]
    other = next(s["supplier_name"] for s in res["suppliers"] if s["supplier_name"] != rec)
    d1 = db.save_decision(rid, basis="llm", recommended_supplier=rec, selected_supplier=rec,
                          reviewer="Abi", reason_category=None, reason_text=None, adjustments=[])
    assert d1["version"] == 1 and d1["is_override"] is False
    d2 = db.save_decision(rid, basis="llm", recommended_supplier=rec, selected_supplier=other,
                          reviewer="Abi", reason_category=OVERRIDE_CATEGORIES[1],
                          reason_text="Lower delivery risk after reference calls.", adjustments=[])
    assert d2["version"] == 2 and d2["is_override"] is True

    view = orchestrator.attach_human_review(res)
    hr = view["human_review"]
    assert hr["reviewed"] is True
    assert hr["decision"]["version"] == 2 and hr["decision"]["selected_supplier"] == other
    assert [d["version"] for d in hr["decision_history"]] == [2, 1]
    assert any(line.startswith("OVERRIDE:") for line in hr["summary"])

    # export carries the review and the original LLM ranking is untouched
    text = orchestrator.result_to_json(view)
    assert '"human_review"' in text and '"decision_history"' in text
    assert [s["final_rank"] for s in view["suppliers"]] == [1, 2, 3]

    # cascade delete removes review rows with the run
    with db.get_connection() as conn:
        conn.execute("DELETE FROM rfp_runs WHERE rfp_run_id = ?", (rid,))
        conn.commit()
    assert db.get_decision_history(rid) == [] and db.get_adjustments(rid) == []


def test_min_reason_constant_is_reasonable():
    assert 5 <= MIN_REASON_CHARS <= 30

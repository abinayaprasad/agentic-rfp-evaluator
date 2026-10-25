from __future__ import annotations

import pytest

from tools.ranking_tool import (
    compute_absolute_score,
    compute_benchmarks,
    compute_ppi,
    explain_tie_breaks,
    rank_suppliers,
    relative_performance_pct,
    score_and_rank,
)


def _evaluation(name: str, scores: list[int]) -> dict:
    return {
        "supplier_name": name,
        "criteria": [
            {"criterion_id": i + 1, "score": s, "max_score": 10,
             "justification": "j", "evidence": "e"}
            for i, s in enumerate(scores)
        ],
        "risks": [],
        "overall_summary": "",
    }


def _entry(name: str, scores: list[int], date: str = "2026-10-01", rating: float = 8.0) -> dict:
    return {
        "supplier_name": name,
        "submission_date": date,
        "experience_rating": rating,
        "evaluation": _evaluation(name, scores),
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Worked example from BUILD_PLAN.md section 7.3
# ---------------------------------------------------------------------------
APEX = [9, 7, 6, 9, 7]
NEXA = [7, 9, 7, 7, 9]


def test_worked_example_absolute_scores(criteria):
    assert compute_absolute_score(_evaluation("Apex", APEX)["criteria"], criteria) == pytest.approx(78.0)
    assert compute_absolute_score(_evaluation("Nexa", NEXA)["criteria"], criteria) == pytest.approx(76.0)


def test_worked_example_benchmarks(criteria):
    bench = compute_benchmarks([_evaluation("Apex", APEX), _evaluation("Nexa", NEXA)], criteria)
    assert bench == {1: 9, 2: 9, 3: 7, 4: 9, 5: 9}


def test_worked_example_full_pipeline(criteria):
    out = score_and_rank([_entry("Apex Systems", APEX), _entry("NexaWorks", NEXA)], criteria)
    apex, nexa = out["suppliers"]

    assert apex["supplier_name"] == "Apex Systems" and apex["final_rank"] == 1
    assert nexa["supplier_name"] == "NexaWorks" and nexa["final_rank"] == 2

    assert apex["absolute_score"] == pytest.approx(78.0)
    assert nexa["absolute_score"] == pytest.approx(76.0)
    assert apex["ppi"] == pytest.approx(90.48, abs=0.01)
    assert nexa["ppi"] == pytest.approx(88.89, abs=0.01)

    assert [c["gap"] for c in apex["criteria"]] == [0, -2, -1, 0, -2]
    assert [c["gap"] for c in nexa["criteria"]] == [-2, 0, 0, -2, 0]

    apex_rel = [c["relative_pct"] for c in apex["criteria"]]
    assert apex_rel == pytest.approx([100.0, 77.78, 85.71, 100.0, 77.78], abs=0.01)

    assert apex["tie_break_reason"].startswith("Ranked first on PPI 90.48")
    assert nexa["tie_break_reason"].startswith("Rule 1 (PPI)")


# ---------------------------------------------------------------------------
# Elementary formulas
# ---------------------------------------------------------------------------
def test_relative_pct_normal():
    assert relative_performance_pct(7, 9) == pytest.approx(77.7778, abs=1e-4)
    assert relative_performance_pct(9, 9) == 100.0


def test_relative_pct_benchmark_zero():
    assert relative_performance_pct(0, 0) == 100.0
    assert relative_performance_pct(3, 0) == 0.0  # defensive; cannot occur when benchmark is a max


def test_all_zero_criterion_gives_everyone_100_on_it(criteria):
    out = score_and_rank([_entry("A", [0, 5, 5, 5, 5]), _entry("B", [0, 5, 5, 5, 5])], criteria)
    for s in out["suppliers"]:
        assert s["criteria"][0]["benchmark"] == 0
        assert s["criteria"][0]["relative_pct"] == 100.0
        assert s["ppi"] == pytest.approx(100.0)


def test_absolute_score_ignores_unknown_criteria(criteria):
    results = _evaluation("X", [10, 10, 10, 10, 10])["criteria"]
    results.append({"criterion_id": 99, "score": 10, "max_score": 10})
    assert compute_absolute_score(results, criteria) == pytest.approx(100.0)


def test_absolute_score_uses_db_weights_not_model_max(criteria):
    results = _evaluation("X", [5, 5, 5, 5, 5])["criteria"]
    for r in results:
        r["max_score"] = 5  # model claims a different max; DB says 10
    assert compute_absolute_score(results, criteria) == pytest.approx(50.0)


def test_ppi_is_weighted_average(criteria):
    results = [
        {"criterion_id": 1, "relative_pct": 100.0},
        {"criterion_id": 2, "relative_pct": 50.0},
        {"criterion_id": 3, "relative_pct": 50.0},
        {"criterion_id": 4, "relative_pct": 50.0},
        {"criterion_id": 5, "relative_pct": 0.0},
    ]
    # (100*30 + 50*20 + 50*20 + 50*20 + 0*10) / 100 = 60
    assert compute_ppi(results, criteria) == pytest.approx(60.0)


def test_ppi_respects_inactive_criteria(criteria):
    active = criteria[:2]  # weights 30 and 20
    results = [{"criterion_id": 1, "relative_pct": 100.0}, {"criterion_id": 2, "relative_pct": 50.0}]
    assert compute_ppi(results, active) == pytest.approx((100 * 30 + 50 * 20) / 50)


# ---------------------------------------------------------------------------
# Tie-break rules
# ---------------------------------------------------------------------------
def test_tie_on_ppi_broken_by_earlier_date(criteria):
    same = [8, 8, 8, 8, 8]
    out = score_and_rank([
        _entry("Zeta", same, date="2026-10-01", rating=5.0),
        _entry("Alpha", same, date="2026-10-05", rating=9.0),
    ], criteria)
    assert [s["supplier_name"] for s in out["suppliers"]] == ["Zeta", "Alpha"]
    assert out["suppliers"][1]["tie_break_reason"].startswith("Rule 2 (earlier submission)")


def test_tie_on_ppi_and_date_broken_by_higher_rating(criteria):
    same = [8, 8, 8, 8, 8]
    out = score_and_rank([
        _entry("Zeta", same, date="2026-10-01", rating=6.0),
        _entry("Alpha", same, date="2026-10-01", rating=9.5),
    ], criteria)
    assert [s["supplier_name"] for s in out["suppliers"]] == ["Alpha", "Zeta"]
    assert out["suppliers"][1]["tie_break_reason"].startswith("Rule 3 (experience rating)")


def test_full_tie_broken_by_name_ascending(criteria):
    same = [8, 8, 8, 8, 8]
    out = score_and_rank([
        _entry("Zeta", same, date="2026-10-01", rating=7.0),
        _entry("alpha", same, date="2026-10-01", rating=7.0),   # lower-case on purpose
        _entry("Mid", same, date="2026-10-01", rating=7.0),
    ], criteria)
    assert [s["supplier_name"] for s in out["suppliers"]] == ["alpha", "Mid", "Zeta"]
    assert out["suppliers"][1]["tie_break_reason"].startswith("Rule 4 (name A-Z)")


def test_ppi_beats_all_other_rules(criteria):
    out = score_and_rank([
        _entry("Late Strong", [9, 9, 9, 9, 9], date="2026-12-31", rating=0.0),
        _entry("Early Weak", [5, 5, 5, 5, 5], date="2026-01-01", rating=10.0),
    ], criteria)
    assert out["suppliers"][0]["supplier_name"] == "Late Strong"


def test_ranks_are_sequential_without_gaps(criteria):
    entries = [_entry(f"S{i}", [i % 10 + 1] * 5) for i in range(7)]
    out = score_and_rank(entries, criteria)
    assert [s["final_rank"] for s in out["suppliers"]] == [1, 2, 3, 4, 5, 6, 7]


def test_ranking_is_deterministic_regardless_of_input_order(criteria):
    entries = [
        _entry("Apex Systems", APEX, "2026-10-01", 8.5),
        _entry("BrightPath Tech", [5, 6, 9, 3, 4], "2026-09-28", 5.0),
        _entry("NexaWorks", NEXA, "2026-10-03", 7.0),
        _entry("Orbit Digital", [6, 6, 7, 6, 9], "2026-10-03", 9.0),
    ]
    first = [s["supplier_name"] for s in score_and_rank(entries, criteria)["suppliers"]]
    second = [s["supplier_name"] for s in score_and_rank(list(reversed(entries)), criteria)["suppliers"]]
    assert first == second


def test_rank_suppliers_does_not_mutate_input():
    results = [
        {"supplier_name": "B", "ppi": 90.0, "submission_date": "2026-10-01", "experience_rating": 5},
        {"supplier_name": "A", "ppi": 95.0, "submission_date": "2026-10-01", "experience_rating": 5},
    ]
    ranked = rank_suppliers(results)
    assert "final_rank" not in results[0]
    assert [r["supplier_name"] for r in ranked] == ["A", "B"]


def test_explain_tie_breaks_lists_every_supplier(criteria):
    out = score_and_rank([_entry("Apex Systems", APEX), _entry("NexaWorks", NEXA)], criteria)
    lines = explain_tie_breaks(out["suppliers"])
    assert len(lines) == 2
    assert lines[0].startswith("#1 Apex Systems:")
    assert lines[1].startswith("#2 NexaWorks:")


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------
def test_result_shape_contains_required_fields(criteria):
    out = score_and_rank([_entry("Apex Systems", APEX, "2026-10-01", 8.5)], criteria)
    s = out["suppliers"][0]
    for key in ("supplier_name", "submission_date", "experience_rating", "absolute_score",
                "ppi", "final_rank", "tie_break_reason", "criteria", "risks",
                "overall_summary", "warnings"):
        assert key in s
    c = s["criteria"][0]
    for key in ("criterion_id", "name", "weight", "max_score", "score", "benchmark",
                "gap", "relative_pct", "justification", "evidence"):
        assert key in c
    assert s["submission_date"] == "2026-10-01"
    assert set(out["benchmarks"]) == {1, 2, 3, 4, 5}

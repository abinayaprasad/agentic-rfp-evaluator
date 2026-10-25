from __future__ import annotations

from tools.validation_tool import (
    MISSING_JUSTIFICATION,
    PLACEHOLDER_TEXT,
    empty_evaluation,
    validate_and_normalise,
)


def _good_payload(name: str = "Apex Systems") -> dict:
    return {
        "supplier_name": name,
        "criteria": [
            {"criterion_id": i, "score": 7, "max_score": 10,
             "justification": f"Justification {i}", "evidence": f"Evidence {i}"}
            for i in range(1, 6)
        ],
        "risks": ["Risk A", "Risk B"],
        "overall_summary": "A solid proposal.",
    }


def _scores(ev: dict) -> dict[int, int]:
    return {c["criterion_id"]: c["score"] for c in ev["criteria"]}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_clean_payload_passes_without_warnings(criteria):
    ev, warnings = validate_and_normalise(_good_payload(), "Apex Systems", criteria)
    assert warnings == []
    assert ev["supplier_name"] == "Apex Systems"
    assert [c["criterion_id"] for c in ev["criteria"]] == [1, 2, 3, 4, 5]
    assert all(c["score"] == 7 for c in ev["criteria"])
    assert ev["risks"] == ["Risk A", "Risk B"]
    assert ev["overall_summary"] == "A solid proposal."


def test_output_is_ordered_by_active_criteria_even_if_input_is_shuffled(criteria):
    payload = _good_payload()
    payload["criteria"].reverse()
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert warnings == []
    assert [c["criterion_id"] for c in ev["criteria"]] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Structural failures
# ---------------------------------------------------------------------------
def test_non_dict_payload_gives_zero_evaluation(criteria):
    ev, warnings = validate_and_normalise(None, "Apex Systems", criteria)
    assert "LLM_INVALID_JSON" in warnings
    assert len(ev["criteria"]) == 5
    assert all(c["score"] == 0 for c in ev["criteria"])


def test_criteria_not_a_list_fills_everything(criteria):
    ev, warnings = validate_and_normalise(
        {"supplier_name": "Apex Systems", "criteria": "oops"}, "Apex Systems", criteria)
    assert "LLM_INVALID_JSON" in warnings
    assert sum(1 for w in warnings if w.startswith("MISSING_CRITERION:")) == 5
    assert all(c["score"] == 0 for c in ev["criteria"])


# ---------------------------------------------------------------------------
# Criterion-level rules
# ---------------------------------------------------------------------------
def test_missing_criterion_is_filled_with_zero(criteria):
    payload = _good_payload()
    payload["criteria"] = [c for c in payload["criteria"] if c["criterion_id"] != 3]
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "MISSING_CRITERION:3" in warnings
    filled = next(c for c in ev["criteria"] if c["criterion_id"] == 3)
    assert filled["score"] == 0
    assert filled["justification"] == MISSING_JUSTIFICATION
    assert filled["evidence"] == PLACEHOLDER_TEXT
    assert len(ev["criteria"]) == 5


def test_unknown_criterion_is_dropped(criteria):
    payload = _good_payload()
    payload["criteria"].append({"criterion_id": 99, "score": 10, "max_score": 10,
                                "justification": "x", "evidence": "y"})
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "UNKNOWN_CRITERION:99" in warnings
    assert 99 not in _scores(ev)
    assert len(ev["criteria"]) == 5


def test_duplicate_criterion_keeps_first(criteria):
    payload = _good_payload()
    payload["criteria"].append({"criterion_id": 1, "score": 2, "max_score": 10,
                                "justification": "dup", "evidence": "dup"})
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "DUPLICATE_CRITERION:1" in warnings
    assert _scores(ev)[1] == 7


def test_score_above_max_is_clipped(criteria):
    payload = _good_payload()
    payload["criteria"][0]["score"] = 15
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "SCORE_CLIPPED_HIGH:1" in warnings
    assert _scores(ev)[1] == 10


def test_negative_score_is_clipped_to_zero(criteria):
    payload = _good_payload()
    payload["criteria"][1]["score"] = -3
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "SCORE_CLIPPED_LOW:2" in warnings
    assert _scores(ev)[2] == 0


def test_non_numeric_score_becomes_zero(criteria):
    payload = _good_payload()
    payload["criteria"][2]["score"] = "high"
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "SCORE_NON_NUMERIC:3" in warnings
    assert _scores(ev)[3] == 0


def test_numeric_string_score_is_accepted(criteria):
    payload = _good_payload()
    payload["criteria"][2]["score"] = "8"
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert warnings == []
    assert _scores(ev)[3] == 8


def test_boolean_score_is_non_numeric(criteria):
    payload = _good_payload()
    payload["criteria"][2]["score"] = True
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "SCORE_NON_NUMERIC:3" in warnings
    assert _scores(ev)[3] == 0


def test_float_score_is_rounded_half_up(criteria):
    payload = _good_payload()
    payload["criteria"][3]["score"] = 7.5
    payload["criteria"][4]["score"] = 6.4
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "SCORE_ROUNDED:4" in warnings and "SCORE_ROUNDED:5" in warnings
    assert _scores(ev)[4] == 8
    assert _scores(ev)[5] == 6


def test_float_above_max_is_rounded_then_clipped(criteria):
    payload = _good_payload()
    payload["criteria"][0]["score"] = 10.7
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "SCORE_ROUNDED:1" in warnings and "SCORE_CLIPPED_HIGH:1" in warnings
    assert _scores(ev)[1] == 10


def test_model_max_score_is_overridden_by_db(criteria):
    payload = _good_payload()
    payload["criteria"][0]["max_score"] = 5
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "MAX_SCORE_OVERRIDDEN:1" in warnings
    assert ev["criteria"][0]["max_score"] == 10


def test_empty_text_fields_get_placeholder(criteria):
    payload = _good_payload()
    payload["criteria"][0]["justification"] = ""
    payload["criteria"][1]["evidence"] = None
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "EMPTY_TEXT_FIELD:1" in warnings and "EMPTY_TEXT_FIELD:2" in warnings
    assert ev["criteria"][0]["justification"] == PLACEHOLDER_TEXT
    assert ev["criteria"][1]["evidence"] == PLACEHOLDER_TEXT


# ---------------------------------------------------------------------------
# Supplier-level rules
# ---------------------------------------------------------------------------
def test_supplier_name_mismatch_uses_user_input(criteria):
    ev, warnings = validate_and_normalise(_good_payload("Apex Sys"), "Apex Systems", criteria)
    assert "SUPPLIER_NAME_MISMATCH" in warnings
    assert ev["supplier_name"] == "Apex Systems"


def test_supplier_name_match_is_case_insensitive(criteria):
    ev, warnings = validate_and_normalise(_good_payload("apex systems"), "Apex Systems", criteria)
    assert "SUPPLIER_NAME_MISMATCH" not in warnings


def test_risks_as_string_is_wrapped(criteria):
    payload = _good_payload()
    payload["risks"] = "Single risk"
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert ev["risks"] == ["Single risk"]


def test_risks_wrong_type_is_emptied(criteria):
    payload = _good_payload()
    payload["risks"] = 42
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "RISKS_NOT_LIST" in warnings
    assert ev["risks"] == []


def test_missing_summary_gets_placeholder(criteria):
    payload = _good_payload()
    del payload["overall_summary"]
    ev, warnings = validate_and_normalise(payload, "Apex Systems", criteria)
    assert "SUMMARY_MISSING" in warnings
    assert ev["overall_summary"] == PLACEHOLDER_TEXT


# ---------------------------------------------------------------------------
# Inactive criteria are ignored entirely
# ---------------------------------------------------------------------------
def test_only_active_criteria_are_considered(criteria):
    active = criteria[:4]  # criterion 5 deactivated
    ev, warnings = validate_and_normalise(_good_payload(), "Apex Systems", active)
    assert "UNKNOWN_CRITERION:5" in warnings
    assert [c["criterion_id"] for c in ev["criteria"]] == [1, 2, 3, 4]


def test_empty_evaluation_helper(criteria):
    ev = empty_evaluation("BrightPath Tech", criteria, "PDF text too short")
    assert ev["supplier_name"] == "BrightPath Tech"
    assert len(ev["criteria"]) == 5
    assert all(c["score"] == 0 and c["justification"] == "PDF text too short" for c in ev["criteria"])
    assert ev["risks"] == ["PDF text too short"]

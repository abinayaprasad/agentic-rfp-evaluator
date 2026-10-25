from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents import orchestrator
from agents.evaluator import MockLLMClient
from db import database as db
from db.seed import seed_if_empty

SAMPLES = {
    "Apex Systems": ("apex_systems.pdf", "2026-10-01", 8.5),
    "BrightPath Tech": ("brightpath_tech.pdf", "2026-09-28", 5.0),
    "NexaWorks": ("nexaworks.pdf", "2026-10-03", 7.0),
    "Orbit Digital": ("orbit_digital.pdf", "2026-10-03", 9.0),
}


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the database module at a fresh, seeded SQLite file."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    seed_if_empty(path)
    return path


def _uploads(sample_rfp_dir: Path, names=None) -> list[dict]:
    uploads = []
    for name, (fname, date, rating) in SAMPLES.items():
        if names and name not in names:
            continue
        p = sample_rfp_dir / fname
        if not p.exists():
            pytest.skip("run scripts/generate_rfps.py first")
        uploads.append({"file_bytes": p.read_bytes(), "filename": fname, "supplier_name": name,
                        "submission_date": date, "experience_rating": rating})
    return uploads


# ---------------------------------------------------------------------------
# Input validation (before any LLM call)
# ---------------------------------------------------------------------------
def test_validate_uploads_catches_problems():
    problems = orchestrator.validate_uploads([
        {"file_bytes": b"x", "supplier_name": "", "submission_date": "2026-01-01", "experience_rating": 5},
        {"file_bytes": b"x", "supplier_name": "A", "submission_date": "not-a-date", "experience_rating": 5},
        {"file_bytes": b"x", "supplier_name": "a", "submission_date": "2026-01-01", "experience_rating": 11},
        {"file_bytes": b"", "supplier_name": "B", "submission_date": "2026-01-01", "experience_rating": 3},
    ])
    joined = " ".join(problems)
    assert "supplier name is required" in joined
    assert "submission date is missing or invalid" in joined
    assert "duplicate supplier name 'a'" in joined
    assert "experience rating must be a number from 0 to 10" in joined
    assert "PDF file is empty or missing" in joined


def test_validate_uploads_empty_list():
    assert orchestrator.validate_uploads([]) == ["Upload at least one supplier PDF."]


def test_batch_rejects_bad_inputs_without_creating_run(temp_db, sample_rfp_dir):
    with pytest.raises(orchestrator.BatchValidationError):
        orchestrator.run_evaluation_batch([], llm_client=MockLLMClient())
    assert db.list_runs() == []


def test_batch_rejects_invalid_weights(temp_db, sample_rfp_dir):
    db.update_criterion(5, is_active=False)  # total becomes 90
    with pytest.raises(orchestrator.BatchValidationError, match="must total 100"):
        orchestrator.run_evaluation_batch(_uploads(sample_rfp_dir, ["Apex Systems"]),
                                          llm_client=MockLLMClient())
    assert db.list_runs() == []


# ---------------------------------------------------------------------------
# Full pipeline with the mock scorer
# ---------------------------------------------------------------------------
def test_full_batch_on_sample_pdfs(temp_db, sample_rfp_dir):
    progress: list[tuple[int, int, str]] = []
    result = orchestrator.run_evaluation_batch(
        _uploads(sample_rfp_dir), progress_callback=lambda n, t, m: progress.append((n, t, m)),
        llm_client=MockLLMClient())

    # shape
    assert result["rfp_run_id"].startswith("RFP-")
    assert result["status"] == "COMPLETED"
    assert result["llm_model"] == "mock-keyword-scorer"
    assert len(result["criteria"]) == 5
    assert set(result["benchmarks"]) == {"1", "2", "3", "4", "5"}
    assert len(result["tie_break_explanation"]) == 4

    # ranks 1..4, sorted
    ranks = [s["final_rank"] for s in result["suppliers"]]
    assert ranks == [1, 2, 3, 4]
    ppis = [s["ppi"] for s in result["suppliers"]]
    assert ppis == sorted(ppis, reverse=True)

    # every supplier has all criteria with peer metrics
    for s in result["suppliers"]:
        assert len(s["criteria"]) == 5
        for c in s["criteria"]:
            assert c["gap"] <= 0
            assert 0 <= c["relative_pct"] <= 100
        assert s["document"]["page_count"] >= 2
        assert s["llm"]["attempts"] == 1

    # the mock scorer should rank the compliance-poor BrightPath last
    assert result["suppliers"][-1]["supplier_name"] == "BrightPath Tech"

    # progress callback covered every step
    assert progress[0][0] == 1 and progress[-1][0] == progress[-1][1] == 4 + 3
    assert progress[-1][2].startswith("Saved results")

    # JSON serialisable
    text = orchestrator.result_to_json(result)
    assert json.loads(text)["rfp_run_id"] == result["rfp_run_id"]


def test_batch_is_persisted(temp_db, sample_rfp_dir):
    result = orchestrator.run_evaluation_batch(_uploads(sample_rfp_dir), llm_client=MockLLMClient())
    stored = db.get_run(result["rfp_run_id"])
    assert stored is not None
    assert stored["status"] == "COMPLETED"
    assert stored["llm_model"] == "mock-keyword-scorer"
    assert len(stored["criteria"]) == 5                      # snapshot
    assert [s["final_rank"] for s in stored["suppliers"]] == [1, 2, 3, 4]
    assert stored["suppliers"][0]["supplier_name"] == result["suppliers"][0]["supplier_name"]
    runs = db.list_runs()
    assert len(runs) == 1 and runs[0]["supplier_count"] == 4


def test_criteria_snapshot_is_independent_of_later_changes(temp_db, sample_rfp_dir):
    result = orchestrator.run_evaluation_batch(_uploads(sample_rfp_dir, ["Apex Systems"]),
                                               llm_client=MockLLMClient())
    db.update_criterion(1, weight=50)
    stored = db.get_run(result["rfp_run_id"])
    assert stored["criteria"][0]["weight"] == 30


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------
def test_unreadable_pdf_gets_zero_scores_and_warning(temp_db, sample_rfp_dir):
    uploads = _uploads(sample_rfp_dir, ["Apex Systems"])
    broken = sample_rfp_dir / "broken_empty.pdf"
    uploads.append({"file_bytes": broken.read_bytes(), "filename": "broken_empty.pdf",
                    "supplier_name": "Ghost Corp", "submission_date": "2026-10-02",
                    "experience_rating": 6.0})
    result = orchestrator.run_evaluation_batch(uploads, llm_client=MockLLMClient())

    ghost = next(s for s in result["suppliers"] if s["supplier_name"] == "Ghost Corp")
    assert ghost["final_rank"] == 2
    assert ghost["absolute_score"] == 0
    assert all(c["score"] == 0 for c in ghost["criteria"])
    assert "PDF_TEXT_TOO_SHORT" in ghost["warnings"]
    assert ghost["llm"]["skipped"] is True
    assert any("Ghost Corp: PDF_TEXT_TOO_SHORT" in w for w in result["run_warnings"])


def test_non_pdf_bytes_are_isolated(temp_db, sample_rfp_dir):
    uploads = _uploads(sample_rfp_dir, ["Apex Systems"])
    uploads.append({"file_bytes": b"not a pdf", "filename": "notes.txt", "supplier_name": "TextCo",
                    "submission_date": "2026-10-02", "experience_rating": 6.0})
    result = orchestrator.run_evaluation_batch(uploads, llm_client=MockLLMClient())
    textco = next(s for s in result["suppliers"] if s["supplier_name"] == "TextCo")
    assert "PDF_EXTRACTION_FAILED" in textco["warnings"]
    assert textco["absolute_score"] == 0


def test_llm_failure_for_one_supplier_does_not_abort_batch(temp_db, sample_rfp_dir):
    client = MockLLMClient(fail_first_n=1)   # first supplier's call fails
    result = orchestrator.run_evaluation_batch(_uploads(sample_rfp_dir, ["Apex Systems", "NexaWorks"]),
                                               llm_client=client)
    assert result["status"] == "COMPLETED"
    failed = next(s for s in result["suppliers"] if "LLM_CALL_FAILED" in s["warnings"])
    assert failed["absolute_score"] == 0
    ok = next(s for s in result["suppliers"] if "LLM_CALL_FAILED" not in s["warnings"])
    assert ok["absolute_score"] > 0
    assert ok["final_rank"] == 1


def test_malformed_json_is_retried_and_recorded(temp_db, sample_rfp_dir):
    client = MockLLMClient(malformed_first_n=1)
    result = orchestrator.run_evaluation_batch(_uploads(sample_rfp_dir, ["Apex Systems"]), llm_client=client)
    s = result["suppliers"][0]
    assert s["llm"]["attempts"] == 2
    assert "LLM_RETRY_INVALID_JSON:1" in s["warnings"]
    assert s["absolute_score"] > 0


def test_unexpected_error_marks_run_failed(temp_db, sample_rfp_dir, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(orchestrator.db, "save_supplier_results", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        orchestrator.run_evaluation_batch(_uploads(sample_rfp_dir, ["Apex Systems"]),
                                          llm_client=MockLLMClient())
    runs = db.list_runs()
    assert len(runs) == 1 and runs[0]["status"] == "FAILED"


def test_run_id_format():
    rid = orchestrator.generate_run_id()
    assert rid.startswith("RFP-") and len(rid) == len("RFP-20261024-A1B2C3")

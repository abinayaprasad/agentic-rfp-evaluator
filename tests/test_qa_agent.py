from __future__ import annotations

import pytest

from agents import orchestrator
from agents.evaluator import LLMCallError, MockLLMClient
from agents.qa_agent import (
    answer_question,
    answer_question_for_run,
    build_qa_user_prompt,
    retrieve_passages,
)
from db import database as db
from db.seed import seed_if_empty

CHUNKS = [
    {"supplier_name": "Apex Systems", "chunk_id": 0,
     "text": "EU customer data hosted in Frankfurt with a Dublin replica; GDPR data residency selected per study."},
    {"supplier_name": "Apex Systems", "chunk_id": 1,
     "text": "Hypercare of 10 weeks after each wave with a named support lead."},
    {"supplier_name": "BrightPath Tech", "chunk_id": 0,
     "text": "Support is provided by email during business hours. Two weeks of hypercare follow go-live."},
    {"supplier_name": "NexaWorks", "chunk_id": 0,
     "text": "Hosting available in the EU (Netherlands) or US (Ohio); GDPR data processing agreement included."},
]


class _ScriptedClient:
    model_name = "scripted"

    def __init__(self, response="Answer text [Apex Systems].", fail=False):
        self._response = response
        self._fail = fail
        self.sent: list[tuple[str, list[str]]] = []

    def complete(self, system, user_messages):
        self.sent.append((system, list(user_messages)))
        if self._fail:
            raise LLMCallError("boom")
        return self._response


# ---------------------------------------------------------------------------
# Retrieval + prompt
# ---------------------------------------------------------------------------
def test_retrieve_passages_ranks_gdpr_chunks_first():
    hits = retrieve_passages("GDPR data residency EU hosting", CHUNKS, k=2)
    assert {h["supplier_name"] for h in hits} == {"Apex Systems", "NexaWorks"}


def test_retrieve_passages_clamps_k():
    assert len(retrieve_passages("hypercare", CHUNKS, k=0)) == 1
    assert len(retrieve_passages("hypercare GDPR support hosting", CHUNKS, k=999)) <= len(CHUNKS)


def test_prompt_numbers_passages_and_names_suppliers():
    user = build_qa_user_prompt("Who offers EU hosting?", CHUNKS[:2])
    assert user.startswith("QUESTION: Who offers EU hosting?")
    assert "[1] (Apex Systems)" in user and "[2] (Apex Systems)" in user
    assert "Frankfurt" in user


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------
def test_answer_question_uses_retrieved_passages_and_returns_them():
    client = _ScriptedClient("Apex and NexaWorks offer EU hosting [Apex Systems] [NexaWorks].")
    res = answer_question("Which suppliers offer EU hosting?", CHUNKS, k=3, client=client)
    assert res.error is None
    assert "EU hosting" in res.answer
    assert res.model == "scripted"
    assert {p["supplier_name"] for p in res.passages} >= {"Apex Systems", "NexaWorks"}
    system, users = client.sent[0]
    assert "ONLY the numbered passages" in system
    assert users[0].startswith("QUESTION:")
    assert set(res.cited_suppliers) == {p["supplier_name"] for p in res.passages}


def test_answer_question_supplier_filter():
    client = _ScriptedClient()
    res = answer_question("hypercare", CHUNKS, k=5, suppliers=["BrightPath Tech"], client=client)
    assert [p["supplier_name"] for p in res.passages] == ["BrightPath Tech"]


def test_answer_question_empty_question():
    res = answer_question("   ", CHUNKS, client=_ScriptedClient())
    assert res.error == "EMPTY_QUESTION"


def test_answer_question_no_chunks():
    res = answer_question("anything", [], client=_ScriptedClient())
    assert res.error == "NO_CHUNKS"


def test_answer_question_no_matches_does_not_call_model():
    client = _ScriptedClient()
    res = answer_question("quantum teleportation", CHUNKS, client=client)
    assert res.error == "NO_MATCHES"
    assert client.sent == []


def test_answer_question_model_failure_still_returns_passages():
    res = answer_question("GDPR hosting", CHUNKS, client=_ScriptedClient(fail=True))
    assert res.error == "boom"
    assert res.passages
    assert "retrieved passages are shown below" in res.answer


def test_mock_client_answers_extractively():
    res = answer_question("GDPR hosting", CHUNKS, client=MockLLMClient())
    assert res.error is None
    assert "[Apex Systems]" in res.answer or "[NexaWorks]" in res.answer


# ---------------------------------------------------------------------------
# Chunks persisted by the orchestrator and used by answer_question_for_run
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "qa.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    seed_if_empty(path)
    return path


def test_orchestrator_persists_chunks_and_qa_uses_them(temp_db, sample_rfp_dir):
    files = {"Apex Systems": "apex_systems.pdf", "NexaWorks": "nexaworks.pdf"}
    uploads = []
    for name, f in files.items():
        p = sample_rfp_dir / f
        if not p.exists():
            pytest.skip("run scripts/generate_rfps.py first")
        uploads.append({"file_bytes": p.read_bytes(), "filename": f, "supplier_name": name,
                        "submission_date": "2026-10-01", "experience_rating": 7.0})
    result = orchestrator.run_evaluation_batch(uploads, llm_client=MockLLMClient())
    run_id = result["rfp_run_id"]

    assert db.count_chunks(run_id) > 10
    stored = db.get_chunks(run_id)
    assert {c["supplier_name"] for c in stored} == {"Apex Systems", "NexaWorks"}
    assert all(c["text"] for c in stored)
    doc_meta = {s["supplier_name"]: s["document"] for s in result["suppliers"]}
    assert doc_meta["Apex Systems"]["chunk_count"] == sum(1 for c in stored if c["supplier_name"] == "Apex Systems")

    res = answer_question_for_run(run_id, "Which supplier mentions 21 CFR Part 11?", client=MockLLMClient())
    assert res.error is None
    assert res.passages

    # chunks are removed with the run
    with db.get_connection() as conn:
        conn.execute("DELETE FROM rfp_runs WHERE rfp_run_id = ?", (run_id,))
        conn.commit()
    assert db.count_chunks(run_id) == 0


def test_unreadable_pdf_produces_no_chunks(temp_db, sample_rfp_dir):
    broken = sample_rfp_dir / "broken_empty.pdf"
    if not broken.exists():
        pytest.skip("run scripts/generate_rfps.py first")
    result = orchestrator.run_evaluation_batch(
        [{"file_bytes": broken.read_bytes(), "filename": "broken_empty.pdf", "supplier_name": "Ghost",
          "submission_date": "2026-10-01", "experience_rating": 5.0}],
        llm_client=MockLLMClient())
    assert db.count_chunks(result["rfp_run_id"]) == 0
    assert "chunk_count" not in result["suppliers"][0]["document"]

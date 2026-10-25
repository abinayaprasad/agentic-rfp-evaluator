"""
Q&A Agent - retrieval-augmented answers over the proposals of one RFP run.

Flow:  question -> BM25 retrieval over stored chunks -> LLM answers using only
those passages, citing supplier names -> answer + the passages shown to the user.

This feature is separate from scoring: it never changes scores, benchmarks or
ranks. It reuses the same LLMClient as the Evaluation Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.evaluator import LLMCallError, LLMClient, make_client
from db import database as db
from tools.retrieval_tool import BM25Index

DEFAULT_TOP_K = 6
MAX_TOP_K = 12


@dataclass
class QAResult:
    question: str
    answer: str
    passages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    error: str | None = None

    @property
    def cited_suppliers(self) -> list[str]:
        seen: list[str] = []
        for p in self.passages:
            if p["supplier_name"] not in seen:
                seen.append(p["supplier_name"])
        return seen


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def build_qa_system_prompt() -> str:
    return (
        "You are a procurement analyst assistant. You answer questions about supplier RFP "
        "proposals for a Clinical Trial Management System (CTMS) selection.\n\n"
        "Rules:\n"
        "1. Use ONLY the numbered passages provided. Do not use outside knowledge.\n"
        "2. Cite the supplier for every claim in square brackets, e.g. [Apex Systems]. "
        "You may also cite the passage number, e.g. [Apex Systems, 2].\n"
        "3. If the passages do not contain the answer, say so plainly and name which "
        "suppliers were silent on the topic.\n"
        "4. When the question compares suppliers, answer supplier by supplier.\n"
        "5. Be concise: a short paragraph or a bulleted list. Plain text, no markdown headers."
    )


def build_qa_user_prompt(question: str, passages: list[dict[str, Any]]) -> str:
    lines = [f"QUESTION: {question.strip()}", "", "RETRIEVED PASSAGES"]
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] ({p['supplier_name']})")
        lines.append(p["text"].strip())
        lines.append("")
    lines.append("Answer the question using only the passages above, citing suppliers in square brackets.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_passages(question: str, chunks: list[dict[str, Any]], k: int = DEFAULT_TOP_K,
                      suppliers: list[str] | None = None) -> list[dict[str, Any]]:
    k = max(1, min(int(k), MAX_TOP_K))
    index = BM25Index(chunks)
    return index.search(question, k=k, suppliers=suppliers)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def answer_question(question: str, chunks: list[dict[str, Any]], *,
                    k: int = DEFAULT_TOP_K, suppliers: list[str] | None = None,
                    client: LLMClient | None = None) -> QAResult:
    """Retrieve passages and ask the LLM. Never raises for model problems."""
    question = (question or "").strip()
    if not question:
        return QAResult(question=question, answer="Please enter a question.", error="EMPTY_QUESTION")
    if not chunks:
        return QAResult(question=question, answer="No proposal text is stored for this run.",
                        error="NO_CHUNKS")

    passages = retrieve_passages(question, chunks, k=k, suppliers=suppliers)
    if not passages:
        return QAResult(question=question,
                        answer="None of the proposals contain passages matching that question. "
                               "Try different wording or key terms used in the documents.",
                        passages=[], error="NO_MATCHES")

    client = client or make_client()
    system = build_qa_system_prompt()
    user = build_qa_user_prompt(question, passages)
    try:
        text = client.complete(system, [user]).strip()
    except LLMCallError as exc:
        return QAResult(question=question, answer="The model call failed; the retrieved passages are "
                        "shown below so you can read them directly.", passages=passages,
                        model=client.model_name, error=str(exc))
    return QAResult(question=question, answer=text or "(empty answer)", passages=passages,
                    model=client.model_name)


def answer_question_for_run(rfp_run_id: str, question: str, *, k: int = DEFAULT_TOP_K,
                            suppliers: list[str] | None = None,
                            client: LLMClient | None = None) -> QAResult:
    """Convenience wrapper: load the run's chunks from SQLite, then answer."""
    chunks = db.get_chunks(rfp_run_id)
    return answer_question(question, chunks, k=k, suppliers=suppliers, client=client)

from __future__ import annotations

from pathlib import Path

import pytest

from tools.document_tool import extract_text
from tools.retrieval_tool import (
    MAX_CHUNK_CHARS,
    BM25Index,
    chunk_text,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Supplier IS certified") == ["supplier", "certified"]


def test_tokenize_keeps_regulatory_identifiers():
    toks = tokenize("ISO 27001:2022 and 21 CFR Part 11 e-signatures")
    assert "27001" in toks and "cfr" in toks and "11" in toks and "e-signatures" in toks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _para(label: str, n: int = 25) -> str:
    return " ".join(f"{label}{i}" for i in range(n))


def test_chunk_text_covers_all_paragraphs_and_is_ordered():
    paras = [_para(f"p{i}_") for i in range(12)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, "Acme", target_chars=400)
    assert len(chunks) > 1
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    joined = "\n".join(c["text"] for c in chunks)
    for p in paras:
        assert p in joined
    assert all(c["supplier_name"] == "Acme" for c in chunks)
    assert all(c["text"].strip() for c in chunks)


def test_chunk_text_respects_target_size_for_normal_paragraphs():
    text = "\n\n".join(_para(f"q{i}_") for i in range(20))
    chunks = chunk_text(text, "Acme", target_chars=500)
    # each chunk is at most target + one paragraph (overlap), never wildly larger
    assert all(len(c["text"]) <= 500 + 300 for c in chunks)


def test_chunk_text_overlap_repeats_short_trailing_paragraph():
    # a + b + short fill the first chunk; short is the trailing paragraph and is
    # carried over as overlap into the next chunk together with c
    short = "Short tail paragraph."
    text = "\n\n".join([_para("a", 40), _para("b", 40), short, _para("c", 40)])
    chunks = chunk_text(text, "Acme", target_chars=350)
    assert len(chunks) == 2
    assert chunks[0]["text"].endswith(short)
    assert chunks[1]["text"].startswith(short)


def test_chunk_text_splits_very_long_paragraph_on_sentences():
    long_para = " ".join(f"Sentence number {i} is here." for i in range(120))
    chunks = chunk_text(long_para, "Acme")
    assert len(chunks) > 1
    assert all(len(c["text"]) <= MAX_CHUNK_CHARS + 50 for c in chunks)


def test_chunk_text_empty_and_whitespace():
    assert chunk_text("", "Acme") == []
    assert chunk_text("   \n\n  \n", "Acme") == []


def test_chunk_text_is_deterministic():
    text = "\n\n".join(_para(f"d{i}_") for i in range(15))
    assert chunk_text(text, "A") == chunk_text(text, "A")


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
@pytest.fixture
def small_index():
    chunks = [
        {"supplier_name": "Apex", "chunk_id": 0, "text": "Data hosted in Frankfurt with GDPR residency selected per study."},
        {"supplier_name": "Apex", "chunk_id": 1, "text": "Pricing is fixed for five years; licence per active study."},
        {"supplier_name": "Bright", "chunk_id": 0, "text": "Lowest price and a seven month timeline with a single go-live."},
        {"supplier_name": "Nexa", "chunk_id": 0, "text": "Hosting available in the EU or US; GDPR data processing agreement included."},
        {"supplier_name": "Orbit", "chunk_id": 0, "text": "Fifteen pharma references over ten years in market."},
    ]
    return BM25Index(chunks)


def test_search_returns_relevant_chunks_first(small_index):
    hits = small_index.search("GDPR residency", k=3)
    assert hits[0]["supplier_name"] == "Apex" and hits[0]["chunk_id"] == 0
    assert {h["supplier_name"] for h in hits[:2]} == {"Apex", "Nexa"}
    assert all(h["score"] > 0 for h in hits)


def test_search_respects_k(small_index):
    assert len(small_index.search("GDPR price timeline references", k=2)) == 2


def test_search_supplier_filter(small_index):
    hits = small_index.search("GDPR", k=5, suppliers=["Nexa"])
    assert [h["supplier_name"] for h in hits] == ["Nexa"]


def test_search_no_match_and_empty_query(small_index):
    assert small_index.search("quantum teleportation", k=5) == []
    assert small_index.search("", k=5) == []
    assert small_index.search("the and of", k=5) == []  # all stopwords


def test_search_is_deterministic_on_ties():
    chunks = [{"supplier_name": s, "chunk_id": 0, "text": "identical passage about hypercare"} for s in "ABC"]
    idx = BM25Index(chunks)
    hits = idx.search("hypercare", k=3)
    assert [h["supplier_name"] for h in hits] == ["A", "B", "C"]


def test_empty_index():
    assert BM25Index([]).search("anything") == []
    assert len(BM25Index([])) == 0


# ---------------------------------------------------------------------------
# End to end on the sample PDFs
# ---------------------------------------------------------------------------
def test_sample_pdfs_retrieve_compliance_from_apex(sample_rfp_dir: Path):
    files = {"Apex Systems": "apex_systems.pdf", "BrightPath Tech": "brightpath_tech.pdf",
             "NexaWorks": "nexaworks.pdf", "Orbit Digital": "orbit_digital.pdf"}
    chunks = []
    for name, f in files.items():
        p = sample_rfp_dir / f
        if not p.exists():
            pytest.skip("run scripts/generate_rfps.py first")
        chunks += chunk_text(extract_text(p.read_bytes())["text"], name)
    assert 20 <= len(chunks) <= 60
    idx = BM25Index(chunks)
    hits = idx.search("21 CFR Part 11 audit trail electronic signatures", k=4)
    assert hits[0]["supplier_name"] in {"Apex Systems", "NexaWorks"}
    assert "BrightPath Tech" not in {h["supplier_name"] for h in hits[:2]}

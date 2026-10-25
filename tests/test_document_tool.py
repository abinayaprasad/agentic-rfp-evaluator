from __future__ import annotations

from pathlib import Path

import pytest

from tools.document_tool import (
    MIN_TEXT_CHARS,
    DocumentExtractionError,
    extract_text,
    truncate_for_prompt,
)

SUPPLIER_FILES = ["apex_systems.pdf", "brightpath_tech.pdf", "nexaworks.pdf", "orbit_digital.pdf"]


@pytest.mark.parametrize("filename", SUPPLIER_FILES)
def test_sample_proposals_are_readable(sample_rfp_dir: Path, filename: str):
    path = sample_rfp_dir / filename
    if not path.exists():
        pytest.skip("run scripts/generate_rfps.py first")
    result = extract_text(path.read_bytes(), filename)
    assert result["is_readable"] is True
    assert result["char_count"] > MIN_TEXT_CHARS
    assert 2 <= result["page_count"] <= 4
    assert "RFP-CTMS-2026-014" in result["text"]


def test_broken_pdf_is_not_readable(sample_rfp_dir: Path):
    path = sample_rfp_dir / "broken_empty.pdf"
    if not path.exists():
        pytest.skip("run scripts/generate_rfps.py first")
    result = extract_text(path.read_bytes(), "broken_empty.pdf")
    assert result["page_count"] == 1
    assert result["char_count"] == 0
    assert result["is_readable"] is False


def test_non_pdf_bytes_raise():
    with pytest.raises(DocumentExtractionError):
        extract_text(b"this is definitely not a pdf", "notes.txt")


def test_empty_bytes_raise():
    with pytest.raises(DocumentExtractionError):
        extract_text(b"", "empty.pdf")


def test_whitespace_is_normalised(sample_rfp_dir: Path):
    path = sample_rfp_dir / "apex_systems.pdf"
    if not path.exists():
        pytest.skip("run scripts/generate_rfps.py first")
    text = extract_text(path.read_bytes())["text"]
    assert "  " not in text          # no double spaces
    assert "\n\n\n" not in text      # no triple blank lines
    assert text == text.strip()


def test_truncate_short_text_unchanged():
    text, truncated = truncate_for_prompt("short text", limit=100)
    assert text == "short text"
    assert truncated is False


def test_truncate_long_text_cuts_at_paragraph():
    para = "word " * 50
    text = "\n\n".join([para] * 10)
    out, truncated = truncate_for_prompt(text, limit=600)
    assert truncated is True
    assert len(out) < len(text)
    assert out.endswith("[... document truncated for evaluation ...]")

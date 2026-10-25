"""
Document Tool - extracts clean text from an uploaded supplier PDF.

This is a plain tool: no LLM, no database. It is called by the orchestrator
once per uploaded file.
"""
from __future__ import annotations

import io
import re
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Below this many characters the document is treated as unreadable
# (for example a scanned PDF with no text layer).
MIN_TEXT_CHARS = 200

# Proposal text longer than this is truncated before prompting, keeping the
# beginning of the document. All synthetic proposals are far below this.
MAX_TEXT_CHARS = 12_000

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


class DocumentExtractionError(Exception):
    """Raised when the bytes cannot be parsed as a PDF."""


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def extract_text(file_bytes: bytes, filename: str = "upload.pdf") -> dict[str, Any]:
    """Extract text from PDF bytes.

    Returns a dict with keys:
        text         cleaned full text (may be empty)
        page_count   number of pages
        char_count   length of ``text``
        filename     the supplied filename
        is_readable  True when char_count >= MIN_TEXT_CHARS

    Raises DocumentExtractionError if the bytes are not a readable PDF.
    """
    if not file_bytes:
        raise DocumentExtractionError(f"{filename}: file is empty")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except PdfReadError as exc:
        raise DocumentExtractionError(f"{filename}: not a valid PDF ({exc})") from exc
    except Exception as exc:  # pypdf can raise a range of low-level errors
        raise DocumentExtractionError(f"{filename}: could not read PDF ({exc})") from exc

    text = _clean("\n\n".join(pages))
    return {
        "text": text,
        "page_count": len(pages),
        "char_count": len(text),
        "filename": filename,
        "is_readable": len(text) >= MIN_TEXT_CHARS,
    }


def truncate_for_prompt(text: str, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    """Return (text, was_truncated). Cuts at the last paragraph break before the limit."""
    if len(text) <= limit:
        return text, False
    cut = text.rfind("\n\n", 0, limit)
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip() + "\n\n[... document truncated for evaluation ...]", True

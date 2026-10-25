"""
Retrieval Tool - chunking and BM25 lexical search over proposal text.

Used by the "Ask the Proposals" feature (retrieval-augmented generation).
Pure Python, deterministic, no network and no extra dependencies, so it is
fully testable offline.

    chunks = chunk_text(text, supplier_name)
    index  = BM25Index(chunks)
    hits   = index.search("EU data residency", k=6)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TARGET_CHUNK_CHARS = 700     # aim for chunks around this size
MAX_CHUNK_CHARS = 1000       # hard cap unless a single sentence is longer
OVERLAP_MAX_CHARS = 220      # a trailing paragraph this short is repeated in the next chunk

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the this to was were will with
we our you your they their which who whom what when where how not no yes than then there these those
""".split())


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with a small stopword list removed.

    Keeps hyphenated and dotted tokens ("part-11", "iso 27001:2022" -> "iso", "27001")
    so regulatory identifiers remain searchable.
    """
    tokens = []
    for raw in _WORD_RE.findall(text.lower()):
        tok = raw.strip(".-")
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Split a paragraph longer than ``limit`` on sentence boundaries."""
    sentences = _SENTENCE_RE.split(paragraph)
    pieces: list[str] = []
    current = ""
    for s in sentences:
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= limit:
            current = f"{current} {s}"
        else:
            pieces.append(current)
            current = s
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, supplier_name: str,
               target_chars: int = TARGET_CHUNK_CHARS,
               max_chars: int = MAX_CHUNK_CHARS) -> list[dict[str, Any]]:
    """Split ``text`` into overlapping, paragraph-aligned chunks.

    Returns a list of dicts: {"supplier_name", "chunk_id", "text", "char_start"}.
    chunk_id is 0-based per supplier. Every non-blank paragraph appears in at
    least one chunk; a short trailing paragraph is repeated at the start of the
    next chunk to give retrieval some context overlap.
    """
    # paragraphs with their character offsets in the original text
    units: list[tuple[str, int]] = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        para = " ".join(m.group(0).split())
        if not para:
            continue
        if len(para) > max_chars:
            units.extend((piece, m.start()) for piece in _split_long_paragraph(para, max_chars))
        else:
            units.append((para, m.start()))

    chunks: list[dict[str, Any]] = []
    current: list[tuple[str, int]] = []
    current_len = 0
    only_overlap = False   # True when `current` holds just the carried-over paragraph

    def emit() -> None:
        chunks.append({
            "supplier_name": supplier_name,
            "chunk_id": len(chunks),
            "text": "\n".join(u[0] for u in current),
            "char_start": current[0][1],
        })

    for unit in units:
        if current and not only_overlap and current_len + len(unit[0]) + 1 > target_chars:
            emit()
            last = current[-1]
            if len(last[0]) <= OVERLAP_MAX_CHARS:
                current, current_len, only_overlap = [last], len(last[0]), True
            else:
                current, current_len, only_overlap = [], 0, False
        current.append(unit)
        current_len += len(unit[0]) + 1
        only_overlap = False

    if current and not only_overlap:
        emit()
    return chunks


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 over a list of chunks (dicts with a "text" key)."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._doc_tokens = [tokenize(c["text"]) for c in self.chunks]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._tf = [Counter(t) for t in self._doc_tokens]
        df: Counter[str] = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        n = len(self.chunks)
        self._idf = {term: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for term, d in df.items()}

    def __len__(self) -> int:
        return len(self.chunks)

    def score(self, query_tokens: list[str], idx: int) -> float:
        tf = self._tf[idx]
        dl = self._doc_len[idx]
        if dl == 0 or self._avg_len == 0:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * dl / self._avg_len)
        s = 0.0
        for q in query_tokens:
            f = tf.get(q)
            if not f:
                continue
            s += self._idf.get(q, 0.0) * (f * (self.k1 + 1)) / (f + norm)
        return s

    def search(self, query: str, k: int = 6,
               suppliers: list[str] | None = None) -> list[dict[str, Any]]:
        """Top-k chunks with a positive BM25 score, highest first.

        Ties are broken by original chunk order so results are deterministic.
        ``suppliers`` optionally restricts results to those supplier names.
        """
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        allowed = {s.lower() for s in suppliers} if suppliers else None
        scored: list[tuple[float, int]] = []
        for i, c in enumerate(self.chunks):
            if allowed is not None and c.get("supplier_name", "").lower() not in allowed:
                continue
            s = self.score(q, i)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [{**self.chunks[i], "score": round(s, 4)} for s, i in scored[:k]]

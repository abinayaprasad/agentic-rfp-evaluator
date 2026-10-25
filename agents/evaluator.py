"""
Evaluation Agent - builds the scoring prompt from the active criteria and
asks an LLM for a criterion-wise JSON scorecard.

Responsibilities (and nothing more):
    * build_prompt()        assemble system + user prompt from DB criteria
    * parse_json_response() tolerant JSON extraction from model text
    * evaluate_supplier()   call the LLM, retry once on malformed JSON

The agent returns the RAW parsed dict. Normalisation is the Validation
Tool's job; arithmetic and ranking belong to the Ranking Tool.

Provider selection (secret / env var LLM_PROVIDER):
    anthropic   Anthropic Claude via the official SDK      (default)
    openai      OpenAI chat completions with JSON mode
    openrouter  any OpenRouter model via the OpenAI-compatible endpoint
    mock        deterministic keyword scorer, no network   (tests, offline demo)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

RFP_CONTEXT = (
    "The buyer is a mid-size pharmaceutical sponsor running about 40 active clinical "
    "studies across 12 countries on a legacy on-premise Clinical Trial Management System "
    "(CTMS) and spreadsheets. It is procuring a validated, cloud-based CTMS that integrates "
    "with its eTMF, EDC and SAP finance systems, automates investigator site payments, "
    "migrates all studies within 12 months, and satisfies 21 CFR Part 11, GxP validation, "
    "GDPR and ISO 27001 requirements."
)

# Ask servers for gzip only: avoids brotli/zstd decoder incompatibilities seen
# on some Python installs (httpx2 + old Brotli). Purely a transport safety net.
SAFE_HTTP_HEADERS = {"Accept-Encoding": "gzip, deflate"}

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 4096
OPENAI_SEED = 20261024   # fixed seed for OpenAI-compatible providers (best-effort determinism)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None   # OpenAI-compatible endpoints (OpenRouter, proxies)

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "openrouter":
            return DEFAULT_OPENROUTER_MODEL
        if self.provider == "openai":
            return DEFAULT_OPENAI_MODEL
        return DEFAULT_ANTHROPIC_MODEL

    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        return OPENROUTER_BASE_URL if self.provider == "openrouter" else None


def _secret(name: str) -> str | None:
    """Read from Streamlit secrets when available, else environment."""
    try:  # pragma: no cover - depends on runtime
        import streamlit as st  # noqa: WPS433
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def get_llm_config() -> LLMConfig:
    provider = (_secret("LLM_PROVIDER") or "anthropic").strip().lower()
    api_key = _secret("LLM_API_KEY")
    if not api_key:
        api_key = _secret("ANTHROPIC_API_KEY") if provider == "anthropic" else _secret("OPENAI_API_KEY")
    if not api_key and provider == "openrouter":
        api_key = _secret("OPENROUTER_API_KEY")
    model = _secret("LLM_MODEL")
    base_url = _secret("LLM_BASE_URL")
    return LLMConfig(provider=provider, api_key=api_key, model=model or None, base_url=base_url or None)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
OUTPUT_SCHEMA_EXAMPLE = """{
  "supplier_name": "<supplier name exactly as given>",
  "criteria": [
    {
      "criterion_id": <integer id from the criteria list>,
      "score": <integer 0..max_score>,
      "max_score": <integer, copy from the criteria list>,
      "justification": "<1-3 sentences explaining the score>",
      "evidence": "<short quote or close paraphrase from the proposal, or 'No evidence found in document'>"
    }
  ],
  "risks": ["<risk statement>", "..."],
  "overall_summary": "<2-3 sentence summary of the proposal's fit>"
}"""


def build_system_prompt() -> str:
    return (
        "You are a procurement evaluation assistant for a pharmaceutical sponsor selecting a "
        "Clinical Trial Management System (CTMS). You score supplier proposals strictly against "
        "the criteria provided.\n\n"
        "Rules:\n"
        "1. Use ONLY evidence present in the supplier document. If the document is silent on a "
        "criterion, give a low score and say that no evidence was found.\n"
        "2. Return exactly one result for EVERY criterion listed, using the given criterion_id. "
        "Do not add, merge or skip criteria.\n"
        "3. Every score is an integer between 0 and that criterion's max_score.\n"
        "4. Scores reflect how fully and credibly the proposal addresses what the criterion inspects. "
        "Vague or unsupported claims score lower than specific, verifiable statements.\n"
        "5. Do not compare this supplier with any other supplier. Do not compute totals, weights, "
        "benchmarks or rankings; those are calculated separately.\n"
        "6. Output valid JSON only. No markdown fences, no commentary before or after the JSON.\n\n"
        "Scoring anchors (apply the same scale to every criterion, max_score 10):\n"
        "- 10: every inspected item is addressed with specific, verifiable detail (named systems, "
        "numbers, dates, certificates, named roles); no material gap.\n"
        "- 8-9: all inspected items addressed; one minor item is brief or generic.\n"
        "- 6-7: most inspected items addressed, but at least one is missing or only asserted without detail.\n"
        "- 4-5: roughly half the inspected items addressed; statements are mostly generic.\n"
        "- 2-3: the topic is mentioned only in passing; claims are vague or unsupported.\n"
        "- 0-1: the document does not address the criterion at all.\n"
        "Count the inspected items from the criterion description, judge each as specific / generic / "
        "absent, then pick the band. Do not round up for tone or confidence of the writing. "
        "Scale proportionally when max_score is not 10."
    )


def build_user_prompt(supplier_name: str, text: str, criteria: list[dict[str, Any]]) -> str:
    lines = [
        "RFP CONTEXT",
        RFP_CONTEXT,
        "",
        "EVALUATION CRITERIA (score each one)",
    ]
    for c in criteria:
        lines.append(
            f"- criterion_id: {int(c['criterion_id'])} | name: {c['name']} | "
            f"max_score: {int(c['max_score'])} | inspect: {c.get('description') or c['name']}"
        )
    lines += [
        "",
        f"SUPPLIER NAME: {supplier_name}",
        "",
        "SUPPLIER PROPOSAL TEXT",
        "<<<",
        text,
        ">>>",
        "",
        "Return JSON matching this schema exactly:",
        OUTPUT_SCHEMA_EXAMPLE,
    ]
    return "\n".join(lines)


def build_prompt(supplier_name: str, text: str, criteria: list[dict[str, Any]]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    if not criteria:
        raise ValueError("At least one active criterion is required to build the prompt")
    return build_system_prompt(), build_user_prompt(supplier_name, text, criteria)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model text. Returns None on failure."""
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------
class LLMClient(Protocol):
    model_name: str

    def complete(self, system: str, user_messages: list[str]) -> str:
        """Send one system prompt and an alternating list of user turns
        (retries are appended as further user messages). Returns model text."""
        ...


class LLMCallError(Exception):
    """Raised when the provider call fails or the model refuses."""


class AnthropicClient:
    def __init__(self, api_key: str | None, model: str):
        import anthropic  # local import so the mock provider needs no SDK
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {"default_headers": SAFE_HTTP_HEADERS}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)
        self.model_name = model

    def complete(self, system: str, user_messages: list[str]) -> str:
        messages: list[dict[str, Any]] = []
        for i, u in enumerate(user_messages):
            if i > 0:
                # keep strict user/assistant alternation for retries
                messages.append({"role": "assistant", "content": "(previous response was not valid JSON)"})
            messages.append({"role": "user", "content": u})
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=messages,
            )
        except self._anthropic.APIStatusError as exc:
            raise LLMCallError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMCallError(f"Anthropic connection error: {exc}") from exc
        except Exception as exc:  # SDK / transport bugs must not abort the batch
            raise LLMCallError(f"Anthropic client error: {type(exc).__name__}: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMCallError("Model declined to evaluate this document (refusal)")
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        if response.stop_reason == "max_tokens":
            text += ""  # parse will fail if truncated; retry path handles it
        return text


class OpenAIClient:
    """OpenAI chat completions; also serves any OpenAI-compatible endpoint
    (OpenRouter, local proxies) via ``base_url``."""

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None):
        import openai
        self._openai = openai
        kwargs: dict[str, Any] = {"default_headers": SAFE_HTTP_HEADERS}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self.model_name = model

    def complete(self, system: str, user_messages: list[str]) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for i, u in enumerate(user_messages):
            if i > 0:
                messages.append({"role": "assistant", "content": "(previous response was not valid JSON)"})
            messages.append({"role": "user", "content": u})
        # JSON mode only when the prompt asks for JSON (OpenAI rejects it otherwise).
        # The evaluation prompt does; the Q&A prompt wants plain text.
        wants_json = "json" in (system + " " + " ".join(user_messages)).lower()
        kwargs: dict[str, Any] = {}
        if wants_json:
            kwargs["response_format"] = {"type": "json_object"}
        # Low-variance settings: greedy decoding plus a fixed seed. OpenAI documents
        # the seed as best-effort, so outputs are near-identical rather than guaranteed.
        determinism = {"temperature": 0, "seed": OPENAI_SEED}
        try:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    **determinism,
                    **kwargs,
                )
            except self._openai.BadRequestError as exc:
                # Some models (reasoning models, certain OpenRouter backends) reject
                # temperature/seed; retry once without them rather than failing.
                msg = str(exc).lower()
                if "temperature" in msg or "seed" in msg:
                    response = self._client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        **kwargs,
                    )
                else:
                    raise
        except self._openai.OpenAIError as exc:
            cause = exc.__cause__ or exc.__context__
            detail = f" (cause: {type(cause).__name__}: {cause})" if cause else ""
            raise LLMCallError(f"OpenAI API error: {exc}{detail}") from exc
        except Exception as exc:  # SDK / transport bugs must not abort the batch
            raise LLMCallError(f"OpenAI client error: {type(exc).__name__}: {exc}") from exc
        return response.choices[0].message.content or ""


class MockLLMClient:
    """Deterministic keyword-based scorer for tests and offline demos.

    Scores each criterion from 2 to 9 based on how many of its indicator
    keywords appear in the proposal text. Produces the same JSON shape as a
    real model, including evidence snippets pulled from the document.
    """
    model_name = "mock-keyword-scorer"

    KEYWORDS: dict[str, list[str]] = {
        "technical": ["api", "rest", "integration", "etmf", "edc", "sap", "bapi", "cdisc",
                      "dashboard", "data model", "saml", "scalab"],
        "implementation": ["wave", "milestone", "migration", "hypercare", "training",
                           "risk register", "exit criteria", "iq", "oq", "pq"],
        "commercial": ["licence", "license", "total", "assum", "fixed", "tco", "per study",
                       "per user", "year 1", "5-year"],
        "security": ["part 11", "annex 11", "gdpr", "iso 27001", "soc 2", "hipaa",
                     "audit trail", "validation", "encrypt", "residency", "penetration"],
        "support": ["24x7", "sla", "p1", "hypercare", "reference", "inspection",
                    "customer success", "availability", "go-live"],
    }

    def __init__(self, fail_first_n: int = 0, malformed_first_n: int = 0):
        self._fail_first_n = fail_first_n
        self._malformed_first_n = malformed_first_n
        self.calls = 0

    @staticmethod
    def _bucket(name: str) -> str:
        n = name.lower()
        if "tech" in n:
            return "technical"
        if "implement" in n or "plan" in n:
            return "implementation"
        if "commerc" in n or "price" in n or "cost" in n:
            return "commercial"
        if "secur" in n or "compli" in n:
            return "security"
        return "support"

    def complete(self, system: str, user_messages: list[str]) -> str:
        self.calls += 1
        if self.calls <= self._fail_first_n:
            raise LLMCallError("mock failure")
        if self.calls <= self._malformed_first_n:
            return "Sure! Here is the evaluation: {not valid json"

        user = user_messages[0]
        if user.startswith("QUESTION:"):
            # Q&A prompt from agents/qa_agent.py: answer extractively from the passages
            passages = re.findall(r"\[(\d+)\] \(([^)]+)\)\n(.+?)(?=\n\n\[\d+\] \(|\n\nAnswer)", user, re.S)
            if not passages:
                return "I could not find this in the retrieved passages."
            return ("Based on the retrieved passages: " +
                    "; ".join(f"[{sup}] {txt.strip()[:160]}" for _n, sup, txt in passages[:3]))
        text = user.split("<<<", 1)[1].split(">>>", 1)[0] if "<<<" in user else user
        low = text.lower()
        supplier = re.search(r"SUPPLIER NAME: (.+)", user)
        criteria = re.findall(
            r"criterion_id: (\d+) \| name: (.+?) \| max_score: (\d+) \| inspect:", user)

        results = []
        for cid, name, max_score in criteria:
            kws = self.KEYWORDS[self._bucket(name)]
            hits = [k for k in kws if k in low]
            max_s = int(max_score)
            score = max(2, min(max_s - 1, 2 + round(len(hits) / max(len(kws), 1) * (max_s - 2))))
            snippet = ""
            if hits:
                idx = low.find(hits[0])
                snippet = text[max(0, idx - 60): idx + 120].replace("\n", " ").strip()
            results.append({
                "criterion_id": int(cid),
                "score": score,
                "max_score": max_s,
                "justification": (f"Document mentions {len(hits)} of {len(kws)} indicators for {name}: "
                                  f"{', '.join(hits[:5]) or 'none'}."),
                "evidence": snippet or "No evidence found in document",
            })
        payload = {
            "supplier_name": supplier.group(1).strip() if supplier else "",
            "criteria": results,
            "risks": ["Mock evaluation: scores are keyword-based, not a real assessment."],
            "overall_summary": "Mock keyword-based evaluation generated without an LLM.",
        }
        return json.dumps(payload)


def make_client(config: LLMConfig | None = None) -> LLMClient:
    config = config or get_llm_config()
    model = config.resolved_model()
    if config.provider == "mock":
        return MockLLMClient()
    if config.provider in ("openai", "openrouter"):
        return OpenAIClient(config.api_key, model, base_url=config.resolved_base_url())
    if config.provider == "anthropic":
        return AnthropicClient(config.api_key, model)
    raise ValueError(
        f"Unknown LLM_PROVIDER '{config.provider}' (use anthropic, openai, openrouter or mock)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@dataclass
class EvaluationCallResult:
    raw: dict[str, Any] | None
    attempts: int
    model: str
    error: str | None = None
    raw_text_preview: str = ""
    warnings: list[str] = field(default_factory=list)


def evaluate_supplier(supplier_name: str, text: str, criteria: list[dict[str, Any]],
                      client: LLMClient | None = None, max_attempts: int = 2) -> EvaluationCallResult:
    """Call the LLM and return the raw parsed JSON (or None after retries).

    Never raises for model/parse problems; the orchestrator decides what to do
    with a None result. Raises ValueError only for programmer errors (no criteria).
    """
    client = client or make_client()
    system, user = build_prompt(supplier_name, text, criteria)
    user_messages = [user]
    warnings: list[str] = []
    last_text = ""

    for attempt in range(1, max_attempts + 1):
        try:
            last_text = client.complete(system, user_messages)
        except LLMCallError as exc:
            return EvaluationCallResult(raw=None, attempts=attempt, model=client.model_name,
                                        error=str(exc), warnings=warnings + ["LLM_CALL_FAILED"])
        parsed = parse_json_response(last_text)
        if parsed is not None:
            return EvaluationCallResult(raw=parsed, attempts=attempt, model=client.model_name,
                                        raw_text_preview=last_text[:500], warnings=warnings)
        warnings.append(f"LLM_RETRY_INVALID_JSON:{attempt}")
        user_messages.append(
            "Your previous response was not valid JSON. Return ONLY the JSON object matching the "
            "schema, with no markdown fences or commentary."
        )

    return EvaluationCallResult(raw=None, attempts=max_attempts, model=client.model_name,
                                error="Model did not return valid JSON after retries",
                                raw_text_preview=last_text[:500], warnings=warnings)

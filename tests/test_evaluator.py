from __future__ import annotations

import json

import pytest

from agents.evaluator import (
    LLMCallError,
    LLMConfig,
    MockLLMClient,
    build_prompt,
    evaluate_supplier,
    make_client,
    parse_json_response,
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def test_prompt_lists_every_active_criterion_from_input(criteria):
    system, user = build_prompt("Apex Systems", "proposal text", criteria)
    for c in criteria:
        assert f"criterion_id: {c['criterion_id']} | name: {c['name']}" in user
    assert "SUPPLIER NAME: Apex Systems" in user
    assert "<<<\nproposal text\n>>>" in user
    assert "JSON only" in system


def test_prompt_reflects_inactive_criteria(criteria):
    _, user = build_prompt("X", "t", criteria[:3])
    assert "criterion_id: 4" not in user
    assert "criterion_id: 5" not in user


def test_prompt_requires_criteria():
    with pytest.raises(ValueError):
        build_prompt("X", "t", [])


def test_prompt_contains_grounding_rules(criteria):
    system, _ = build_prompt("X", "t", criteria)
    assert "ONLY evidence present in the supplier document" in system
    assert "exactly one result for EVERY criterion" in system
    assert "Do not compute totals" in system


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def test_parse_plain_json():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_in_markdown_fence():
    text = "```json\n{\"a\": 1}\n```"
    assert parse_json_response(text) == {"a": 1}


def test_parse_json_with_surrounding_commentary():
    text = "Here is your result:\n{\"a\": {\"b\": 2}}\nHope this helps."
    assert parse_json_response(text) == {"a": {"b": 2}}


def test_parse_rejects_non_object():
    assert parse_json_response("[1, 2, 3]") is None
    assert parse_json_response("") is None
    assert parse_json_response("not json at all") is None
    assert parse_json_response("{broken") is None


# ---------------------------------------------------------------------------
# Mock client behaviour
# ---------------------------------------------------------------------------
def test_mock_client_returns_schema_shaped_json(criteria):
    client = MockLLMClient()
    result = evaluate_supplier("Apex Systems", "We provide a REST API, 21 CFR Part 11 audit trail, 24x7 SLA.",
                               criteria, client=client)
    assert result.raw is not None
    assert result.attempts == 1
    assert result.raw["supplier_name"] == "Apex Systems"
    ids = sorted(c["criterion_id"] for c in result.raw["criteria"])
    assert ids == [1, 2, 3, 4, 5]
    for c in result.raw["criteria"]:
        assert 0 <= c["score"] <= 10
        assert c["justification"] and c["evidence"]


def test_mock_client_is_deterministic(criteria):
    text = "Named integrations, GDPR, ISO 27001, wave plan, hypercare."
    a = evaluate_supplier("S", text, criteria, client=MockLLMClient()).raw
    b = evaluate_supplier("S", text, criteria, client=MockLLMClient()).raw
    assert a == b


def test_mock_scores_rich_text_higher_than_poor_text(criteria):
    rich = ("REST API integration with eTMF and EDC and SAP BAPI, 21 CFR Part 11, GDPR, ISO 27001, "
            "SOC 2, audit trail, encryption, wave milestones, migration, hypercare, 24x7 SLA P1.")
    poor = "We are a company that sells software."
    r = evaluate_supplier("A", rich, criteria, client=MockLLMClient()).raw
    p = evaluate_supplier("B", poor, criteria, client=MockLLMClient()).raw
    assert sum(c["score"] for c in r["criteria"]) > sum(c["score"] for c in p["criteria"])


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------
def test_retries_once_on_malformed_json_then_succeeds(criteria):
    client = MockLLMClient(malformed_first_n=1)
    result = evaluate_supplier("S", "text", criteria, client=client)
    assert result.raw is not None
    assert result.attempts == 2
    assert result.warnings == ["LLM_RETRY_INVALID_JSON:1"]
    assert client.calls == 2


def test_gives_up_after_max_attempts(criteria):
    client = MockLLMClient(malformed_first_n=5)
    result = evaluate_supplier("S", "text", criteria, client=client, max_attempts=2)
    assert result.raw is None
    assert result.attempts == 2
    assert "did not return valid JSON" in result.error
    assert client.calls == 2


def test_call_error_returns_failed_result_without_raising(criteria):
    client = MockLLMClient(fail_first_n=1)
    result = evaluate_supplier("S", "text", criteria, client=client)
    assert result.raw is None
    assert "LLM_CALL_FAILED" in result.warnings
    assert "mock failure" in result.error


class _ScriptedClient:
    """Returns pre-scripted responses; records the messages it was sent."""
    model_name = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent: list[list[str]] = []

    def complete(self, system, user_messages):
        self.sent.append(list(user_messages))
        return self._responses.pop(0)


def test_retry_appends_correction_message(criteria):
    client = _ScriptedClient(["garbage", json.dumps({"supplier_name": "S", "criteria": []})])
    result = evaluate_supplier("S", "text", criteria, client=client)
    assert result.raw == {"supplier_name": "S", "criteria": []}
    assert len(client.sent[1]) == 2
    assert "not valid JSON" in client.sent[1][1]


# ---------------------------------------------------------------------------
# Config / factory
# ---------------------------------------------------------------------------
def test_make_client_mock():
    client = make_client(LLMConfig(provider="mock"))
    assert isinstance(client, MockLLMClient)


def test_make_client_unknown_provider():
    with pytest.raises(ValueError):
        make_client(LLMConfig(provider="banana"))


def test_default_models():
    assert LLMConfig(provider="anthropic").resolved_model() == "claude-opus-5-5"
    assert LLMConfig(provider="openai").resolved_model() == "gpt-4o-mini"
    assert LLMConfig(provider="anthropic", model="x").resolved_model() == "x"


def test_openrouter_config_defaults():
    cfg = LLMConfig(provider="openrouter")
    assert cfg.resolved_model() == "openai/gpt-4o-mini"
    assert cfg.resolved_base_url() == "https://openrouter.ai/api/v1"
    assert LLMConfig(provider="openai").resolved_base_url() is None
    assert LLMConfig(provider="openai", base_url="http://x").resolved_base_url() == "http://x"


def test_make_client_openrouter_uses_openai_client_with_base_url():
    client = make_client(LLMConfig(provider="openrouter", api_key="sk-or-dummy"))
    assert client.model_name == "openai/gpt-4o-mini"
    assert str(client._client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs

        class _Msg:
            content = '{"ok": true}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _openai_client_with_fake():
    from agents.evaluator import OpenAIClient
    client = OpenAIClient(api_key="sk-dummy", model="gpt-4o-mini")
    fake = _FakeCompletions()
    client._client.chat.completions.create = fake.create
    return client, fake


def test_openai_json_mode_enabled_when_prompt_mentions_json():
    client, fake = _openai_client_with_fake()
    client.complete("Return JSON only.", ["score this"])
    assert fake.kwargs["response_format"] == {"type": "json_object"}


def test_openai_json_mode_disabled_for_plain_text_prompts():
    client, fake = _openai_client_with_fake()
    client.complete("Answer in plain text.", ["QUESTION: who offers EU hosting?"])
    assert "response_format" not in fake.kwargs


def test_openai_sends_temperature_zero_and_fixed_seed():
    from agents.evaluator import OPENAI_SEED
    client, fake = _openai_client_with_fake()
    client.complete("Return JSON only.", ["score this"])
    assert fake.kwargs["temperature"] == 0
    assert fake.kwargs["seed"] == OPENAI_SEED


def test_openai_retries_without_determinism_params_when_rejected():
    import openai
    from agents.evaluator import OpenAIClient
    client = OpenAIClient(api_key="sk-dummy", model="o-model")
    calls = []

    class _Msg:
        content = '{"ok": true}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    def create(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise openai.BadRequestError(
                "Unsupported parameter: 'temperature' is not supported with this model.",
                response=__import__("httpx2").Response(400, request=__import__("httpx2").Request("POST", "http://x")),
                body=None,
            )
        return _Resp()

    client._client.chat.completions.create = create
    out = client.complete("Return JSON only.", ["score this"])
    assert out == '{"ok": true}'
    assert len(calls) == 2
    assert "temperature" in calls[0] and "temperature" not in calls[1]


def test_system_prompt_contains_scoring_anchors(criteria):
    system, _ = build_prompt("X", "t", criteria)
    assert "Scoring anchors" in system
    for band in ("- 10:", "- 8-9:", "- 6-7:", "- 4-5:", "- 2-3:", "- 0-1:"):
        assert band in system

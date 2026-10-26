"""GeminiLLM behaviour with a fake client: no network, no API key needed."""

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors

from reviewer.agents import llm as llm_mod
from reviewer.agents.llm import GeminiLLM, LLMError, _strip_code_fence, make_llm
from reviewer.config import Settings


def response(text):
    usage = SimpleNamespace(prompt_token_count=1200, cached_content_token_count=200, candidates_token_count=100,
                            thoughts_token_count=50)
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=[])


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.configs = []

    async def generate_content(self, *, model, contents, config):
        self.configs.append(config)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make(outcomes, monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(llm_mod.asyncio, "sleep", no_sleep)
    g = GeminiLLM(Settings(_env_file=None, gemini_api_key="k", gemini_min_interval_s=0))
    g.client = SimpleNamespace(aio=SimpleNamespace(models=FakeModels(outcomes)))
    return g


def call(g):
    return asyncio.run(g.complete_json(agent="bug", system="s", context="c", instructions="i",
                                       schema={"type": "object"}))


def rate_limited(detail="Quota exceeded per minute. 'retryDelay': '7s'"):
    return errors.ClientError(429, {"error": {"code": 429, "message": detail, "status": "RESOURCE_EXHAUSTED"}})


def test_retries_after_per_minute_rate_limit(monkeypatch):
    g = make([rate_limited(), response('{"findings": []}')], monkeypatch)
    data, usage = call(g)
    assert data == {"findings": []}
    assert usage.input_tokens == 1000 and usage.cache_read_input_tokens == 200
    assert usage.output_tokens == 150  # thinking tokens count as output
    assert usage.cost_usd == 0.0       # free tier / unpriced model


def test_daily_quota_fails_fast_with_clear_message(monkeypatch):
    g = make([rate_limited("Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier")], monkeypatch)
    with pytest.raises(LLMError, match="DAILY quota"):
        call(g)


def test_schema_rejection_falls_back_to_plain_json(monkeypatch):
    bad_schema = errors.ClientError(400, {"error": {"code": 400, "message": "Invalid JSON schema: field x"}})
    g = make([bad_schema, response('```json\n{"findings": []}\n```')], monkeypatch)
    data, _ = call(g)
    assert data == {"findings": []}
    configs = g.client.aio.models.configs
    assert configs[0].response_json_schema is not None and configs[1].response_json_schema is None


def test_unknown_model_points_to_models_command(monkeypatch):
    g = make([errors.ClientError(404, {"error": {"code": 404, "message": "models/x is not found"}})], monkeypatch)
    with pytest.raises(LLMError, match="reviewer.cli models"):
        call(g)


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="aistudio.google.com"):
        GeminiLLM(Settings(_env_file=None, gemini_api_key=""))


def test_strip_code_fence_and_factory():
    assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}\n'
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'
    assert isinstance(make_llm(Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k")), GeminiLLM)
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        make_llm(Settings(_env_file=None, llm_provider="nope"))


def test_invalid_json_retried_once_and_tokens_summed(monkeypatch):
    g = make([response('{"findings": [ {"file": '), response('{"findings": []}')], monkeypatch)
    data, usage = call(g)
    assert data == {"findings": []}
    assert usage.input_tokens == 2000 and usage.output_tokens == 300  # both attempts counted


def test_truncation_reported_clearly(monkeypatch):
    truncated = SimpleNamespace(text='{"findings": [', usage_metadata=None,
                                candidates=[SimpleNamespace(finish_reason="FinishReason.MAX_TOKENS")])
    g = make([truncated, truncated], monkeypatch)
    with pytest.raises(LLMError, match="truncated"):
        call(g)

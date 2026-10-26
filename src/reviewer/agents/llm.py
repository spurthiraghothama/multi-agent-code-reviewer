"""LLM access layer.

Design decisions:

* **Direct Anthropic SDK, not a LangChain chat model.** LangGraph orchestrates the agents, but each
  agent calls Claude through the official SDK. Two reasons: (1) we use *structured outputs*
  (`output_config.format` with a JSON schema), which constrains decoding so the response is
  guaranteed to parse - LangChain's `with_structured_output` relies on forced tool calls, which the
  current Claude models reject; (2) we need the exact `usage` block (including cache read/write
  tokens) for per-review cost tracking.

* **One protocol, swappable implementations.** Agents depend on the `LLM` protocol, so tests use a
  scripted fake, and the backend (Claude or Gemini, via LLM_PROVIDER) can change without touching
  agent code.

* **Prompt caching.** All three specialists receive the same (large) repo context. We send it as
  the first content block with a cache breakpoint and put the agent-specific instructions *after*
  it, so the shared prefix is identical across agents. Caveat: the specialists run in parallel, so
  on a cold cache they may all miss; caching pays off reliably on the lead agent and on re-reviews
  after a new push (same context within the 5-minute TTL).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Protocol

import anthropic

from reviewer.agents.cost import cost_usd
from reviewer.config import Settings
from reviewer.models import TokenUsage

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when an agent call fails. Carries usage when tokens were already spent, so a failed
    call still shows up in the review's cost."""

    def __init__(self, message: str, usage: TokenUsage | None = None):
        super().__init__(message)
        self.usage = usage


class LLM(Protocol):
    async def complete_json(self, *, agent: str, system: str, context: str, instructions: str,
                            schema: dict[str, Any]) -> tuple[dict[str, Any], TokenUsage]: ...


class AnthropicLLM:
    def __init__(self, settings: Settings, model: str | None = None):
        self.s = settings
        self.model = model or settings.llm_model
        if not (settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # Fail at startup with a clear message, not deep inside the first agent call.
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to the .env file in the project folder.")
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None, max_retries=3)

    async def complete_json(self, *, agent: str, system: str, context: str, instructions: str,
                            schema: dict[str, Any]) -> tuple[dict[str, Any], TokenUsage]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.s.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": instructions},
            ]}],
            output_config={"effort": self.s.llm_effort, "format": {"type": "json_schema", "schema": schema}},
        )
        t0 = time.perf_counter()
        try:
            if self.s.llm_refusal_fallback:
                resp = await self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = await self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise LLMError(f"{agent}: rate limited after retries") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"{agent}: API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"{agent}: connection error") from e
        latency = time.perf_counter() - t0

        u = resp.usage
        usage = TokenUsage(
            agent=agent, model=resp.model,
            input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            cache_read_input_tokens=u.cache_read_input_tokens or 0,
            latency_s=latency,
        )
        usage.cost_usd = cost_usd(resp.model, usage.input_tokens, usage.output_tokens,
                                  usage.cache_creation_input_tokens, usage.cache_read_input_tokens)

        if resp.stop_reason == "refusal":
            raise LLMError(f"{agent}: model declined the request", usage)
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"{agent}: output truncated at max_tokens={self.s.llm_max_tokens}", usage)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as e:
            raise LLMError(f"{agent}: response was not valid JSON", usage) from e


class GeminiLLM:
    """Google Gemini backend, usable on the free tier.

    Differences from the Claude backend, and how they're handled:
      * Free-tier rate limits are tight (a handful of requests per minute, plus a daily cap). We
        space calls out with a minimum interval, and on HTTP 429 we wait (using the server's
        suggested retry delay when it gives one) and retry. A *daily* quota can't be waited out
        within a run, so that fails immediately with a clear message.
      * Structured output uses `response_json_schema`. If the API rejects our schema (it supports a
        subset of JSON Schema), we retry once in plain JSON mode; Pydantic validation downstream
        still catches anything malformed.
      * Cost is reported as $0 (free tier); token counts are still tracked for the benchmark.
    """

    MAX_ATTEMPTS = 6

    def __init__(self, settings: Settings, model: str | None = None):
        from google import genai

        self.s = settings
        self.model = model or settings.gemini_model
        key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                           "and add it to the .env file in the project folder.")
        self.client = genai.Client(api_key=key)
        # The SDK logs "AFC is enabled ..." on every call at INFO; we don't use function calling.
        logging.getLogger("google_genai.models").setLevel(logging.WARNING)
        self._gate = asyncio.Lock()
        self._last_call = 0.0
        self._schema_supported = True

    async def _throttle(self) -> None:
        # Serialise call *starts* so parallel agents don't burst past the per-minute limit.
        async with self._gate:
            wait = self.s.gemini_min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    PARSE_ATTEMPTS = 2

    async def complete_json(self, *, agent: str, system: str, context: str, instructions: str,
                            schema: dict[str, Any]) -> tuple[dict[str, Any], TokenUsage]:
        prompt = f"{context}\n\n{instructions}"
        t0 = time.perf_counter()
        total = TokenUsage(agent=agent, model=self.model)
        problem = ""
        # Gemini occasionally returns truncated or malformed JSON (long test code is the usual cause).
        # One retry fixes most of these; tokens from the failed attempt still count.
        for parse_attempt in range(1, self.PARSE_ATTEMPTS + 1):
            resp = await self._generate(agent, system, prompt, schema)
            u = resp.usage_metadata
            if u:
                cached = u.cached_content_token_count or 0
                total.input_tokens += (u.prompt_token_count or 0) - cached
                total.cache_read_input_tokens += cached
                # Thinking tokens are billed as output on paid tiers, so count them as output here too.
                total.output_tokens += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
            total.latency_s = time.perf_counter() - t0
            total.cost_usd = cost_usd(self.model, total.input_tokens, total.output_tokens, 0,
                                      total.cache_read_input_tokens)

            reason = str(resp.candidates[0].finish_reason) if resp.candidates else "no candidates"
            text = resp.text or ""
            if "MAX_TOKENS" in reason:
                problem = f"output truncated at {self.s.gemini_max_tokens} tokens (thinking counts toward it)"
            elif not text:
                problem = f"no text returned (finish reason: {reason})"
            else:
                try:
                    return json.loads(_strip_code_fence(text)), total
                except json.JSONDecodeError:
                    problem = "response was not valid JSON"
            if parse_attempt < self.PARSE_ATTEMPTS:
                log.warning("%s: %s; retrying once", agent, problem)
        raise LLMError(f"{agent}: {problem}", total)

    async def _generate(self, agent: str, system: str, prompt: str, schema: dict[str, Any]):
        """One generate_content call, with throttling and rate-limit / server-error retries."""
        from google.genai import errors, types

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            config = types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=self.s.gemini_max_tokens,
                response_mime_type="application/json",
                response_json_schema=schema if self._schema_supported else None,
            )
            await self._throttle()
            try:
                return await self.client.aio.models.generate_content(model=self.model, contents=prompt,
                                                                    config=config)
            except errors.ClientError as e:
                msg = str(e)
                if e.code == 400 and self._schema_supported and "schema" in msg.lower():
                    log.warning("Gemini rejected the JSON schema; falling back to plain JSON mode")
                    self._schema_supported = False
                    continue
                if e.code == 429:
                    if "PerDay" in msg or "per day" in msg.lower():
                        raise LLMError(f"{agent}: Gemini free-tier DAILY quota used up. Wait until it resets "
                                       "(midnight Pacific time) or set GEMINI_MODEL to a model with a "
                                       "higher daily limit (e.g. gemini-flash-lite-latest).") from e
                    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", msg)
                    delay = int(m.group(1)) + 1 if m else min(15 * attempt, 60)
                    log.info("%s: Gemini rate limit hit, retrying in %ss (attempt %d/%d)", agent, delay,
                             attempt, self.MAX_ATTEMPTS)
                    await asyncio.sleep(delay)
                    continue
                if e.code == 404:
                    raise LLMError(f"{agent}: model '{self.model}' not found. Run "
                                   "`python -m reviewer.cli models` to list the models your key can use.") from e
                raise LLMError(f"{agent}: Gemini API error {e.code}: {msg[:300]}") from e
            except errors.ServerError as e:
                log.info("%s: Gemini server error %s, retrying", agent, e.code)
                await asyncio.sleep(min(10 * attempt, 60))
        raise LLMError(f"{agent}: Gemini still rate limited after {self.MAX_ATTEMPTS} attempts")


def _strip_code_fence(text: str) -> str:
    """Plain JSON mode occasionally wraps output in ```json fences; strip them."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t


def make_llm(settings: Settings) -> LLM:
    """Pick the backend from LLM_PROVIDER. Everything downstream only sees the LLM protocol."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return GeminiLLM(settings)
    if provider == "anthropic":
        return AnthropicLLM(settings)
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use 'anthropic' or 'gemini')")

"""Single-prompt baseline: one LLM call that sees only the diff.

This is what most "AI PR reviewer" bots do. Same model, same effort and same output schema as the
multi-agent system, so the comparison isolates the *architecture* (context retrieval, specialised
agents, sandbox verification, lead filtering), not the model.

All of its findings count as posted - it has no verification step to filter with.
"""

from __future__ import annotations

import time

from reviewer.agents.llm import LLM, LLMError
from reviewer.agents.prompts import FINDINGS_SCHEMA
from reviewer.agents.specialists import parse_findings
from reviewer.models import Finding, ReviewRequest, TokenUsage
from reviewer.pipeline import diff_overview

BASELINE_SYSTEM = """You are an expert code reviewer. Review the pull request diff for bugs, edge cases,
performance problems and missing tests. For each issue give the file, the line number in the new
version of the file, a severity, your confidence (0-1), a description, a suggested fix and concrete
evidence. Set test_code to "". Return an empty list if the PR has no issues."""


async def run_baseline(req: ReviewRequest, llm: LLM) -> tuple[list[Finding], list[TokenUsage], float, str]:
    t0 = time.perf_counter()
    try:
        raw, usage = await llm.complete_json(agent="baseline", system=BASELINE_SYSTEM,
                                             context=diff_overview(req, limit=200_000),
                                             instructions="Review this pull request.", schema=FINDINGS_SCHEMA)
    except LLMError as e:
        return [], [e.usage] if e.usage else [], time.perf_counter() - t0, str(e)
    findings = parse_findings("baseline", raw, {f.path for f in req.changed_files})
    return findings, [usage], time.perf_counter() - t0, ""

"""Lead agent: deduplicate, rank, filter, summarise.

Split between deterministic code and the LLM on purpose:

  * Deterministic: exact/near duplicates (same file, lines within 3, similar text), ranking, and
    the posting policy. These must be predictable and testable - you don't want an LLM deciding
    whether a confirmed critical bug gets posted.
  * LLM: semantic duplicates ("off-by-one in pagination" vs "last page is skipped" from two
    agents) and the human-readable summary. That's judgement and prose, which LLMs are good at.

If the lead LLM call fails we still post: deterministic dedupe + a templated summary.
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher

from reviewer.agents.llm import LLM, LLMError
from reviewer.agents.prompts import LEAD_INSTRUCTIONS, LEAD_SCHEMA, SYSTEM
from reviewer.models import Finding, TokenUsage, VerificationStatus

log = logging.getLogger(__name__)

LINE_WINDOW = 3
TEXT_SIMILARITY = 0.6


def strength(f: Finding) -> tuple:
    """Sort key: confirmed first, then severity, then confidence."""
    return (f.status == VerificationStatus.confirmed, f.severity.rank, f.confidence)


def rank(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=strength, reverse=True)


def _near_duplicate(a: Finding, b: Finding) -> bool:
    if a.file != b.file or abs(a.line - b.line) > LINE_WINDOW:
        return False
    return a.category == b.category or SequenceMatcher(None, a.description.lower(),
                                                       b.description.lower()).ratio() >= TEXT_SIMILARITY


def deterministic_dedupe(findings: list[Finding]) -> list[Finding]:
    kept: list[Finding] = []
    for f in rank(findings):  # strongest first, so the survivor of each cluster is the best one
        if not any(_near_duplicate(f, k) for k in kept):
            kept.append(f)
    return kept


def should_post(f: Finding, min_confidence: float) -> bool:
    """Posting policy: confirmed by a failing test, or high-severity with high confidence.

    A finding whose generated test *passed* (not_reproduced) is evidence against it, so it is not
    posted even if the model was confident.
    """
    if f.category == "missing_test":
        return f.status == VerificationStatus.confirmed  # a failing "missing test" = a found bug
    if f.status == VerificationStatus.confirmed:
        return True
    if f.status == VerificationStatus.not_reproduced:
        return False
    return f.severity.rank >= 3 and f.confidence >= min_confidence


def apply_duplicate_groups(findings: list[Finding], groups: list[list[str]]) -> list[Finding]:
    by_id = {f.id: f for f in findings}
    drop: set[str] = set()
    for group in groups:
        members = [by_id[i] for i in group if i in by_id]
        if len(members) < 2:
            continue
        # Trust the LLM on *what* is a duplicate, but not on which one survives: keep the strongest.
        best = max(members, key=strength)
        drop.update(m.id for m in members if m.id != best.id)
    return [f for f in findings if f.id not in drop]


def fallback_summary(posted: list[Finding]) -> str:
    if not posted:
        return "Automated review: no significant issues found."
    lines = [f"Automated review found {len(posted)} issue(s):"]
    lines += [f"- **{f.severity.value}** `{f.file}:{f.line}` {f.description}"
              + (" (confirmed by failing test)" if f.status == VerificationStatus.confirmed else "")
              for f in posted]
    return "\n".join(lines)


def suggested_tests_section(tests: list[Finding]) -> str:
    if not tests:
        return ""
    parts = ["", "<details><summary>Suggested tests (generated, verified passing in sandbox)</summary>", ""]
    for t in tests:
        parts += [f"**{t.file}** - {t.description}", "```python", t.test_code.strip(), "```", ""]
    parts.append("</details>")
    return "\n".join(parts)


async def run_lead(findings: list[Finding], llm: LLM, min_confidence: float,
                   diff_overview: str) -> tuple[str, list[Finding], list[TokenUsage], list[str]]:
    candidates = deterministic_dedupe(findings)
    # Filter *before* the LLM sees anything, so the summary can only talk about what gets posted.
    postable = [f for f in candidates if should_post(f, min_confidence)]
    passing_tests = [f for f in candidates
                     if f.category == "missing_test" and f.status == VerificationStatus.not_reproduced]
    usage: list[TokenUsage] = []
    errors: list[str] = []
    summary = ""

    if postable:
        payload = [{"id": f.id, **f.model_dump(include={"agent", "category", "file", "line", "severity",
                                                        "confidence", "description", "evidence", "status"},
                                               mode="json")} for f in postable]
        try:
            raw, u = await llm.complete_json(
                agent="lead", system=SYSTEM, context=diff_overview,
                instructions=LEAD_INSTRUCTIONS + json.dumps(payload, indent=1), schema=LEAD_SCHEMA)
            usage.append(u)
            postable = apply_duplicate_groups(postable, raw.get("duplicate_groups", []))
            summary = raw.get("summary_markdown", "")
        except LLMError as e:
            log.error("lead agent failed, using deterministic summary: %s", e)
            errors.append(str(e))
            if e.usage:
                usage.append(e.usage)

    posted = rank(postable)
    summary = (summary or fallback_summary(posted)) + suggested_tests_section(passing_tests)
    return summary, posted, usage, errors

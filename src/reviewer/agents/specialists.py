"""The three specialist agents (bug, performance, test).

They are deliberately thin: an agent = (name, instructions). All three share the system prompt,
the context and the output schema, and differ only in their instructions. Specialisation helps
because a single "find everything" prompt tends to fixate on one kind of issue; separate agents
each spend their full attention (and reasoning budget) on one lens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError

from reviewer.agents.llm import LLM, LLMError
from reviewer.agents.prompts import BUG_INSTRUCTIONS, FINDINGS_SCHEMA, PERF_INSTRUCTIONS, SYSTEM, TEST_INSTRUCTIONS
from reviewer.models import Finding, TokenUsage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Specialist:
    name: str
    instructions: str


BUG = Specialist("bug", BUG_INSTRUCTIONS)
PERF = Specialist("performance", PERF_INSTRUCTIONS)
TEST = Specialist("test", TEST_INSTRUCTIONS)
SPECIALISTS = [BUG, PERF, TEST]


def parse_findings(agent: str, raw: dict, valid_files: set[str] | None = None) -> list[Finding]:
    """Validate raw LLM JSON into Findings. Invalid items are dropped individually, not the whole batch."""
    out = []
    for item in raw.get("findings", []):
        try:
            item = {**item, "confidence": min(max(float(item.get("confidence", 0)), 0.0), 1.0)}
            f = Finding(agent=agent, **item)
        except (ValidationError, TypeError, ValueError) as e:
            log.warning("%s: dropping malformed finding: %s", agent, e)
            continue
        if valid_files is not None and f.file not in valid_files:
            # Hallucinated path - nothing we could comment on or test.
            log.warning("%s: dropping finding for unknown file %s", agent, f.file)
            continue
        out.append(f)
    return out


async def run_specialist(spec: Specialist, llm: LLM, context: str,
                         valid_files: set[str] | None) -> tuple[list[Finding], list[TokenUsage], list[str]]:
    try:
        raw, usage = await llm.complete_json(agent=spec.name, system=SYSTEM, context=context,
                                             instructions=spec.instructions, schema=FINDINGS_SCHEMA)
    except LLMError as e:
        # One failing agent must not sink the whole review: return what we know and record the error.
        log.error("%s agent failed: %s", spec.name, e)
        return [], [e.usage] if e.usage else [], [str(e)]
    return parse_findings(spec.name, raw, valid_files), [usage], []

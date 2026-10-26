"""LangGraph orchestration.

                 ┌──► bug ────────┐
        START ───┼──► performance ┼──► verify (sandbox) ──► lead ──► END
                 └──► test ───────┘

Why LangGraph rather than a hand-written `asyncio.gather`?
  * The graph *is* the architecture: fan-out/fan-in, ordering and state are declared, not buried
    in control flow, which makes it easy to add a node (e.g. a security agent) or a conditional
    edge (e.g. skip the sandbox when no finding has a test).
  * State merging is explicit: `findings` and `usage` use an `operator.add` reducer, so the three
    parallel branches append to the same lists without racing or overwriting each other.
  * Nodes in the same "superstep" run concurrently, and the join waits for all of them.
For a pipeline this small, gather() would also work - the graph earns its keep as the system grows.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from reviewer.agents.lead import run_lead
from reviewer.agents.llm import LLM
from reviewer.agents.specialists import SPECIALISTS, Specialist, run_specialist
from reviewer.models import Finding, ReviewRequest, TokenUsage
from reviewer.sandbox.verify import Verifier


class ReviewState(TypedDict, total=False):
    request: ReviewRequest
    context: str
    # Files shown to the agents. Agents may flag a *caller* in another file, so this is wider than
    # the diff; anything outside it is a hallucinated path and gets dropped.
    valid_files: set[str]
    diff_overview: str
    findings: Annotated[list[Finding], operator.add]      # appended to by parallel specialists
    usage: Annotated[list[TokenUsage], operator.add]
    errors: Annotated[list[str], operator.add]
    verified: list[Finding]
    posted: list[Finding]
    summary: str


def build_review_graph(llm: LLM, verifier: Verifier, min_confidence: float,
                       specialists: list[Specialist] = SPECIALISTS):
    g = StateGraph(ReviewState)

    def make_specialist_node(spec: Specialist):
        async def node(state: ReviewState) -> dict:
            findings, usage, errors = await run_specialist(spec, llm, state["context"], state.get("valid_files"))
            return {"findings": findings, "usage": usage, "errors": errors}
        return node

    async def verify_node(state: ReviewState) -> dict:
        findings = state.get("findings", [])
        return {"verified": await verifier.verify(state["request"], findings)}

    async def lead_node(state: ReviewState) -> dict:
        summary, posted, usage, errors = await run_lead(state.get("verified", []), llm, min_confidence,
                                                        state.get("diff_overview", ""))
        return {"summary": summary, "posted": posted, "usage": usage, "errors": errors}

    for spec in specialists:
        g.add_node(spec.name, make_specialist_node(spec))
        g.add_edge(START, spec.name)
    g.add_node("verify", verify_node)
    g.add_node("lead", lead_node)
    # A list of sources = a join: "verify" runs once, after *all* specialists finish.
    g.add_edge([s.name for s in specialists], "verify")
    g.add_edge("verify", "lead")
    g.add_edge("lead", END)
    return g.compile()

"""End-to-end review pipeline: ReviewRequest -> ReviewResult.

This is source-agnostic: the webhook builds a ReviewRequest from a GitHub PR, the benchmark
builds one from local git branches, and both call `ReviewPipeline.review`. Posting to GitHub is
a separate step so that dry runs and the benchmark exercise the exact same code path.
"""

from __future__ import annotations

import asyncio
import logging
import time

from reviewer.agents.graph import build_review_graph
from reviewer.agents.llm import LLM
from reviewer.config import Settings
from reviewer.indexing.call_graph import CallGraph
from reviewer.indexing.context import build_context
from reviewer.indexing.parser import parse_repo
from reviewer.indexing.store import CodeIndex
from reviewer.models import ReviewRequest, ReviewResult
from reviewer.sandbox.verify import Verifier

log = logging.getLogger(__name__)


def diff_overview(req: ReviewRequest, limit: int = 20_000) -> str:
    text = f"# Pull request: {req.title}\n\n" + "".join(
        f"### {f.path} ({f.status})\n```diff\n{f.patch}\n```\n" for f in req.changed_files)
    return text[:limit]


class ReviewPipeline:
    def __init__(self, settings: Settings, llm: LLM, verifier: Verifier, use_vector_index: bool = True):
        self.s = settings
        self.use_vector_index = use_vector_index
        self.graph = build_review_graph(llm, verifier, settings.min_confidence_unverified)

    async def review(self, req: ReviewRequest) -> ReviewResult:
        t0 = time.perf_counter()

        # 1. Index: parse every file (cheap, ~ms per file) for the call graph; re-embed only files
        #    whose content hash changed (expensive part).
        modules = await asyncio.to_thread(parse_repo, req.repo_dir)
        call_graph = CallGraph(modules)
        index = None
        if self.use_vector_index:
            index = CodeIndex(self.s.chroma_path, req.repo_full_name)
            await asyncio.to_thread(index.sync, modules)

        # 2. Context: changed functions + callers + similar code, within a budget.
        ctx = await asyncio.to_thread(
            build_context, req, call_graph, index, max_chars=self.s.max_context_chars,
            max_callers=self.s.max_callers_per_function, k_similar=self.s.retrieval_k)
        log.info("Context: %d changed functions, %d callers, %d related chunks, %d chars%s",
                 len(ctx.changed_functions), ctx.caller_count, ctx.similar_count, len(ctx.text),
                 " (truncated)" if ctx.truncated else "")

        # 3. Agents -> sandbox -> lead.
        state = await self.graph.ainvoke({
            "request": req, "context": ctx.text, "valid_files": ctx.files,
            "diff_overview": diff_overview(req), "findings": [], "usage": [], "errors": [],
        })
        for err in state.get("errors", []):
            log.warning("Agent error: %s", err)

        return ReviewResult(
            summary_markdown=state.get("summary", ""),
            posted_findings=state.get("posted", []),
            all_findings=state.get("verified", []),
            usage=state.get("usage", []),
            latency_s=time.perf_counter() - t0,
            errors=state.get("errors", []),
        )

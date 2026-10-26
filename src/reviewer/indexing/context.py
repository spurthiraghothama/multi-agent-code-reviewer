"""Builds the review context for a PR: changed functions + their callers + related code.

Why not just send the diff? Most real bugs in a PR are *interactions*: the function changed its
return type and a caller three files away still expects the old one. A diff-only reviewer cannot
see that caller. The context we build, in priority order:

  1. the diff itself                                   (what changed)
  2. full source of each changed function              (the diff hunk is often partial)
  3. callers of each changed function (call graph)     (who might break)
  4. semantically similar code (ChromaDB)              (conventions, sibling implementations,
                                                         existing tests for this code)

Everything is emitted with real line numbers so agents can cite file:line precisely. We enforce
a character budget and fill it in priority order, so for a huge PR we drop "similar code" before
we drop callers, and callers before the diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reviewer.indexing.call_graph import CallGraph
from reviewer.indexing.parser import CodeChunk, module_name_for
from reviewer.indexing.store import CodeIndex
from reviewer.models import ReviewRequest


@dataclass
class ReviewContext:
    text: str
    changed_functions: list[str] = field(default_factory=list)
    caller_count: int = 0
    similar_count: int = 0
    truncated: bool = False
    files: set[str] = field(default_factory=set)  # every file shown to the agents


class _Budget:
    def __init__(self, limit: int):
        self.limit, self.used, self.parts, self.truncated = limit, 0, [], False

    def add(self, text: str) -> bool:
        if self.used + len(text) > self.limit:
            self.truncated = True
            return False
        self.parts.append(text)
        self.used += len(text)
        return True


def _block(title: str, chunk: CodeChunk) -> str:
    return f"### {title}: `{chunk.qualname}` ({chunk.file}:{chunk.start_line}-{chunk.end_line})\n" \
           f"```python\n{chunk.numbered_source()}\n```\n"


def build_context(req: ReviewRequest, graph: CallGraph, index: CodeIndex | None, *,
                  max_chars: int, max_callers: int, k_similar: int) -> ReviewContext:
    budget = _Budget(max_chars)
    budget.add(f"# Pull request: {req.title}\n{(req.body or '')[:2000]}\n\n## Diff\n")
    for f in req.changed_files:
        if not budget.add(f"### {f.path} ({f.status})\n```diff\n{f.patch}\n```\n"):
            budget.add(f"### {f.path} ({f.status}) - diff omitted (context budget)\n")

    py_files = [f.path for f in req.changed_files if f.path.endswith(".py") and f.status != "removed"]
    if py_files:
        budget.add("\n## Import paths (use these in test_code)\n" + "".join(
            f"- `{p}` -> `import {module_name_for(p)}`\n" for p in py_files))

    changed: list[CodeChunk] = []
    for f in req.changed_files:
        if f.path.endswith(".py") and f.status != "removed":
            changed.extend(graph.enclosing_chunks(f.path, f.added_lines))

    included = {c.chunk_id for c in changed}
    budget.add("\n## Changed functions (full source at PR head)\n")
    for c in changed:
        budget.add(_block("Changed", c))

    caller_count = 0
    budget.add("\n## Callers of changed functions (from static call graph)\n")
    for c in changed:
        for edge in graph.callers(c.fqname)[:max_callers]:
            caller = graph.chunks[edge.caller]
            if caller.chunk_id in included:
                continue
            note = "" if edge.kind == "exact" else " [name-based match, may be a different function]"
            if budget.add(_block(f"Caller of `{c.qualname}`{note}", caller)):
                included.add(caller.chunk_id)
                caller_count += 1

    similar_count = 0
    if index is not None and k_similar > 0:
        budget.add("\n## Related code (semantic search)\n")
        by_id = {ch.chunk_id: ch for ch in graph.chunks.values()}
        for c in changed:
            for hit in index.similar(c, k_similar, included):
                ch = by_id.get(hit["id"])
                if ch is None or ch.chunk_id in included:
                    continue
                if budget.add(_block(f"Related to `{c.qualname}`", ch)):
                    included.add(ch.chunk_id)
                    similar_count += 1

    by_id = {ch.chunk_id: ch for ch in graph.chunks.values()}
    files = {f.path for f in req.changed_files} | {by_id[i].file for i in included if i in by_id}
    return ReviewContext(text="".join(budget.parts), changed_functions=[c.fqname for c in changed],
                         caller_count=caller_count, similar_count=similar_count, truncated=budget.truncated,
                         files=files)

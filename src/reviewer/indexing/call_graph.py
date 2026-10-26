"""Static call graph: "which functions call which".

Python is dynamic, so a perfect call graph is impossible statically. We resolve calls in order of
confidence and record how each edge was resolved:

  exact  - the callee was found through scope rules we can trust:
             foo()            -> a function `foo` defined in the same module
             self.foo()       -> method `foo` on the enclosing class
             mod.foo() / foo() where `mod`/`foo` came from an import we can follow
             Foo()            -> Foo.__init__
  fuzzy  - only the bare name matched, e.g. `obj.save()` -> every method named `save`. We keep these
           only when the name is rare (<= FUZZY_LIMIT definitions), since `get` or `run` would link
           everything to everything.

Precision vs recall trade-off: for code review, missing a caller means we might not notice a broken
call site; a spurious caller just costs a few hundred tokens of context. So we lean towards recall
but cap the fan-out.
"""

from __future__ import annotations

import builtins
from collections import defaultdict
from dataclasses import dataclass

from reviewer.indexing.parser import CallSite, CodeChunk, ModuleInfo

FUZZY_LIMIT = 3


@dataclass(frozen=True)
class Edge:
    caller: str  # fqname
    callee: str  # fqname
    kind: str    # exact | fuzzy


class CallGraph:
    def __init__(self, modules: dict[str, ModuleInfo]):
        self.chunks: dict[str, CodeChunk] = {}
        self.by_short_name: dict[str, list[str]] = defaultdict(list)
        for mod in modules.values():
            for c in mod.chunks:
                self.chunks[c.fqname] = c
                self.by_short_name[c.qualname.split(".")[-1]].append(c.fqname)

        self._callers: dict[str, set[Edge]] = defaultdict(set)
        self._callees: dict[str, set[Edge]] = defaultdict(set)
        for mod in modules.values():
            for chunk in mod.chunks:
                for site in chunk.calls:
                    for callee, kind in self._resolve(site, chunk, mod):
                        if callee == chunk.fqname:
                            continue  # ignore direct recursion
                        e = Edge(chunk.fqname, callee, kind)
                        self._callers[callee].add(e)
                        self._callees[chunk.fqname].add(e)

    def _lookup(self, fq: str) -> str | None:
        if fq in self.chunks:
            return fq
        # Calling a class means calling its constructor.
        if f"{fq}.__init__" in self.chunks:
            return f"{fq}.__init__"
        return None

    def _resolve(self, site: CallSite, chunk: CodeChunk, mod: ModuleInfo) -> list[tuple[str, str]]:
        prefix = f"{mod.module}." if mod.module else ""
        if site.kind == "self" and chunk.class_name:
            hit = self._lookup(f"{prefix}{chunk.class_name}.{site.name}")
            if hit:
                return [(hit, "exact")]
        elif site.kind == "name":
            for candidate in (f"{prefix}{site.name}", mod.imports.get(site.name, "")):
                hit = candidate and self._lookup(candidate)
                if hit:
                    return [(hit, "exact")]
            if hasattr(builtins, site.name):
                return []  # len(), print(), ... never user code
        elif site.kind == "attr" and site.base:
            if site.base in mod.imports:  # module.func() or ImportedClass.method()
                hit = self._lookup(f"{mod.imports[site.base]}.{site.name}")
                if hit:
                    return [(hit, "exact")]
            hit = self._lookup(f"{prefix}{site.base}.{site.name}")  # LocalClass.staticmethod()
            if hit:
                return [(hit, "exact")]

        candidates = self.by_short_name.get(site.name, [])
        if 0 < len(candidates) <= FUZZY_LIMIT:
            return [(c, "fuzzy") for c in candidates]
        return []

    def callers(self, fqname: str) -> list[Edge]:
        # Exact edges first: they are the most trustworthy context.
        return sorted(self._callers.get(fqname, ()), key=lambda e: (e.kind != "exact", e.caller))

    def callees(self, fqname: str) -> list[Edge]:
        return sorted(self._callees.get(fqname, ()), key=lambda e: (e.kind != "exact", e.callee))

    def chunks_in_file(self, file: str) -> list[CodeChunk]:
        return [c for c in self.chunks.values() if c.file == file]

    def enclosing_chunks(self, file: str, lines: list[int]) -> list[CodeChunk]:
        """Functions/classes whose line range overlaps any of the given lines (the 'changed functions').

        A changed line outside every definition is usually a deletion anchor: e.g. a PR that deletes
        the last methods of a class is anchored on the first line *after* the class. For such lines
        we take the definition ending just before it (within NEAR_LINES), plus its class, so the agents
        see the code the deletion was cut from instead of a bare diff.
        """
        near_lines = 2
        file_chunks = self.chunks_in_file(file)
        hits: dict[str, CodeChunk] = {}
        for ln in set(lines):
            inside = [c for c in file_chunks if c.start_line <= ln <= c.end_line]
            if not inside:
                before = [c for c in file_chunks if 0 < ln - c.end_line <= near_lines]
                inside = [max(before, key=lambda c: c.end_line)] if before else []
                for c in list(inside):
                    if c.kind == "method":
                        inside += [k for k in file_chunks if k.kind == "class" and k.qualname == c.class_name]
            for c in inside:
                hits[c.chunk_id] = c
        return sorted(hits.values(), key=lambda c: c.start_line)

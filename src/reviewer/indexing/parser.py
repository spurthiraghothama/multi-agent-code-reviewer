"""AST-based chunking.

Why chunk by function/class instead of fixed-size windows?
  * A function is the unit a reviewer reasons about, and the unit a call graph talks about.
  * Fixed windows cut functions in half, so a retrieved chunk often lacks its signature or return.
  * The metadata (file, qualified name, line range) lets us map a diff line -> enclosing function
    exactly, and lets agents cite precise line numbers.

Each chunk also records the raw call sites inside it; call_graph.py resolves them into edges.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".tox", ".eggs",
             "site-packages"}


@dataclass
class CallSite:
    """A call expression, kept in a resolvable form.

    kind:
      name      - foo()                -> name="foo"
      self      - self.foo() / cls.foo -> name="foo"
      attr      - x.foo()              -> base="x", name="foo"
    """
    kind: str
    name: str
    base: str = ""


@dataclass
class CodeChunk:
    file: str            # repo-relative posix path
    module: str          # dotted module name
    qualname: str        # e.g. "Client.send" or "parse_url"
    kind: str            # function | method | class
    start_line: int
    end_line: int
    source: str
    class_name: str = ""
    calls: list[CallSite] = field(default_factory=list)

    @property
    def fqname(self) -> str:
        return f"{self.module}.{self.qualname}" if self.module else self.qualname

    @property
    def chunk_id(self) -> str:
        # The start line makes ids unique even when a name repeats in one file
        # (property getter/setter pairs, platform-specific definitions).
        return f"{self.file}::{self.qualname}::{self.start_line}"

    def numbered_source(self) -> str:
        """Source with real file line numbers - LLMs cite lines far more accurately with these."""
        return "\n".join(f"{self.start_line + i:>5} | {line}" for i, line in enumerate(self.source.splitlines()))


@dataclass
class ModuleInfo:
    file: str
    module: str
    chunks: list[CodeChunk]
    # alias -> fully qualified target, e.g. {"np": "numpy", "get": "requests.api.get"}
    imports: dict[str, str]
    file_hash: str


def module_name_for(rel_path: str) -> str:
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[0] in {"src", "lib"}:
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    pkg_parts = module.split(".") if module else []
    if not is_package:
        pkg_parts = pkg_parts[:-1]
    if level > 1:
        pkg_parts = pkg_parts[: len(pkg_parts) - (level - 1)]
    return ".".join([*pkg_parts, *([target] if target else [])])


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[CallSite] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            self.calls.append(CallSite("name", func.id))
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in {"self", "cls"}:
                self.calls.append(CallSite("self", func.attr))
            elif isinstance(func.value, ast.Name):
                self.calls.append(CallSite("attr", func.attr, base=func.value.id))
            else:
                # e.g. self.session.get(...), foo().bar() - we only know the method name
                self.calls.append(CallSite("attr", func.attr, base=""))
        self.generic_visit(node)


def _calls_in(node: ast.AST) -> list[CallSite]:
    c = _CallCollector()
    c.visit(node)
    return c.calls


def _is_overload_stub(node: ast.AST) -> bool:
    """`@overload` declarations are type-hint stubs with no logic; only the final definition is real."""
    for d in getattr(node, "decorator_list", []):
        if (isinstance(d, ast.Name) and d.id == "overload") or (isinstance(d, ast.Attribute) and d.attr == "overload"):
            return True
    return False


def parse_source(source: str, rel_path: str) -> ModuleInfo:
    module = module_name_for(rel_path)
    is_package = rel_path.endswith("__init__.py")
    file_hash = hashlib.sha256(source.encode()).hexdigest()
    lines = source.splitlines()
    with warnings.catch_warnings():  # e.g. invalid escape sequences in old code: not our problem
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=rel_path)

    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = (_resolve_relative(module, is_package, node.level, node.module)
                    if node.level else (node.module or ""))
            for a in node.names:
                imports[a.asname or a.name] = f"{base}.{a.name}" if base else a.name

    def src(n: ast.AST, end: int | None = None) -> str:
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        return "\n".join(lines[start - 1:(end or n.end_lineno)])

    chunks: list[CodeChunk] = []
    for node in tree.body:
        if _is_overload_stub(node):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(CodeChunk(rel_path, module, node.name, "function", node.lineno, node.end_lineno,
                                    src(node), calls=_calls_in(node)))
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            # Class chunk = header, docstring and class attributes (up to the first method), so that
            # methods are not embedded twice.
            header_end = (min(m.decorator_list[0].lineno if m.decorator_list else m.lineno for m in methods) - 1
                          if methods else node.end_lineno)
            chunks.append(CodeChunk(rel_path, module, node.name, "class", node.lineno, max(header_end, node.lineno),
                                    src(node, max(header_end, node.lineno)), class_name=node.name))
            for m in methods:
                if _is_overload_stub(m):
                    continue
                chunks.append(CodeChunk(rel_path, module, f"{node.name}.{m.name}", "method", m.lineno,
                                        m.end_lineno, src(m), class_name=node.name, calls=_calls_in(m)))
    return ModuleInfo(rel_path, module, chunks, imports, file_hash)


def iter_python_files(repo_dir: Path) -> list[Path]:
    return sorted(p for p in repo_dir.rglob("*.py") if not (SKIP_DIRS & set(p.relative_to(repo_dir).parts)))


def parse_repo(repo_dir: Path) -> dict[str, ModuleInfo]:
    """Parse every Python file. Files that don't parse (e.g. Python 2, templates) are skipped."""
    modules: dict[str, ModuleInfo] = {}
    for path in iter_python_files(repo_dir):
        rel = path.relative_to(repo_dir).as_posix()
        try:
            modules[rel] = parse_source(path.read_text(encoding="utf-8", errors="replace"), rel)
        except SyntaxError as e:
            log.debug("Skipping %s: %s", rel, e)
    return modules

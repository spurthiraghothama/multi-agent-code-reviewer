from pathlib import Path

from reviewer.indexing.call_graph import CallGraph
from reviewer.indexing.context import build_context
from reviewer.indexing.parser import module_name_for, parse_repo, parse_source
from reviewer.models import ChangedFile, ReviewRequest


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "pricing.py").write_text(
        "def apply_discount(price, pct):\n"
        "    return price * (1 - pct / 100)\n"
        "\n"
        "\n"
        "class Cart:\n"
        "    TAX = 0.2\n"
        "\n"
        "    def total(self, items):\n"
        "        return sum(self.line_total(i) for i in items)\n"
        "\n"
        "    def line_total(self, item):\n"
        "        return apply_discount(item.price, item.discount)\n"
    )
    (tmp_path / "shop" / "checkout.py").write_text(
        "from shop.pricing import apply_discount, Cart\n"
        "from . import pricing as p\n"
        "\n"
        "def checkout(items):\n"
        "    cart = Cart()\n"
        "    return cart.total(items)\n"
        "\n"
        "def quote(price):\n"
        "    return p.apply_discount(price, 10)\n"
    )
    return tmp_path


def test_module_names():
    assert module_name_for("src/pkg/mod.py") == "pkg.mod"
    assert module_name_for("pkg/__init__.py") == "pkg"


def test_chunks_have_line_ranges():
    info = parse_source("import os\n\ndef a():\n    return 1\n\nclass B:\n    x = 1\n    def m(self):\n        a()\n",
                        "m.py")
    by_name = {c.qualname: c for c in info.chunks}
    assert (by_name["a"].start_line, by_name["a"].end_line) == (3, 4)
    assert by_name["B"].kind == "class" and by_name["B"].end_line == 7  # header only, methods excluded
    assert by_name["B.m"].kind == "method" and by_name["B.m"].start_line == 8


def test_call_graph_resolution(tmp_path):
    graph = CallGraph(parse_repo(make_repo(tmp_path)))
    callers = {e.caller: e.kind for e in graph.callers("shop.pricing.apply_discount")}
    assert callers["shop.pricing.Cart.line_total"] == "exact"   # same-module call
    assert callers["shop.checkout.quote"] == "exact"            # via relative `import ... as p`
    assert {e.caller for e in graph.callers("shop.pricing.Cart.line_total")} == {"shop.pricing.Cart.total"}
    # `Cart()` has no __init__, and `cart.total(...)` on an instance is resolved by name only
    assert graph.callers("shop.pricing.Cart.total")[0].kind == "fuzzy"


def test_enclosing_chunks_and_context(tmp_path):
    repo = make_repo(tmp_path)
    graph = CallGraph(parse_repo(repo))
    assert [c.qualname for c in graph.enclosing_chunks("shop/pricing.py", [2])] == ["apply_discount"]

    req = ReviewRequest(repo_full_name="o/r", base_sha="a", head_sha="b", repo_dir=repo,
                        changed_files=[ChangedFile(path="shop/pricing.py", status="modified",
                                                   patch="@@ -2 +2 @@\n-x\n+y", added_lines=[2])])
    ctx = build_context(req, graph, None, max_chars=50_000, max_callers=5, k_similar=0)
    assert ctx.changed_functions == ["shop.pricing.apply_discount"]
    assert ctx.caller_count == 2
    assert "Cart.line_total" in ctx.text and "quote" in ctx.text
    assert "    2 |     return price" in ctx.text  # real line numbers
    assert "import shop.pricing" in ctx.text
    assert ctx.files == {"shop/pricing.py", "shop/checkout.py"}


def test_context_budget_truncates(tmp_path):
    repo = make_repo(tmp_path)
    graph = CallGraph(parse_repo(repo))
    req = ReviewRequest(repo_full_name="o/r", base_sha="a", head_sha="b", repo_dir=repo,
                        changed_files=[ChangedFile(path="shop/pricing.py", status="modified",
                                                   patch="+y", added_lines=[2])])
    ctx = build_context(req, graph, None, max_chars=300, max_callers=5, k_similar=0)
    assert ctx.truncated and len(ctx.text) <= 300


def test_overload_stubs_skipped_and_ids_unique():
    src = (
        "import typing as t\n"
        "class C:\n"
        "    @t.overload\n"
        "    def f(self, x: int) -> int: ...\n"
        "    @t.overload\n"
        "    def f(self, x: str) -> str: ...\n"
        "    def f(self, x):\n"
        "        return x\n"
        "    @property\n"
        "    def v(self):\n"
        "        return 1\n"
        "    @v.setter\n"
        "    def v(self, value):\n"
        "        pass\n"
    )
    chunks = parse_source(src, "m.py").chunks
    f_chunks = [c for c in chunks if c.qualname == "C.f"]
    assert len(f_chunks) == 1 and f_chunks[0].start_line == 7  # only the real implementation
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # getter and setter both kept, without colliding


def test_deletion_after_last_method_maps_to_class():
    """Regression: deleting the last methods of a class anchors the change *after* the class."""
    src = ("class Sentinel:\n"          # 1
           "    A = 1\n"                 # 2
           "\n"                          # 3
           "    def __repr__(self):\n"   # 4
           "        return 'x'\n"        # 5
           "\n"                          # 6
           "\n"                          # 7  <- anchor of the deleted __copy__/__deepcopy__
           "UNSET = Sentinel.A\n")       # 8
    graph = CallGraph({"m.py": parse_source(src, "m.py")})
    assert [c.qualname for c in graph.enclosing_chunks("m.py", [7])] == ["Sentinel", "Sentinel.__repr__"]
    assert graph.enclosing_chunks("m.py", [8]) == []  # a genuinely module-level change stays unattributed

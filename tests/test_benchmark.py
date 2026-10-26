import subprocess

from reviewer.benchmark.metrics import markdown_table, matches, score_case, summarize
from reviewer.benchmark.mine import BugCase, is_test_path, mine_repo
from reviewer.benchmark.prepare import prepare_case
from reviewer.models import Finding

GT = {"pkg/a.py": [20, 21]}


def F(file="pkg/a.py", line=20):
    return Finding(agent="x", category="c", file=file, line=line, severity="high", confidence=1,
                   description="d", suggested_fix="s", evidence="e")


def test_matching_tolerance():
    assert matches(F(line=25), GT)
    assert not matches(F(line=27), GT)
    assert not matches(F(file="pkg/b.py"), GT)


def test_summary_metrics():
    results = [
        score_case("ma", "c1", [F(), F(line=90)], GT, 0.10, 30),
        score_case("ma", "c2", [], GT, 0.20, 50),
        score_case("bl", "c1", [F(line=200), F(line=300), F()], GT, 0.01, 5),
        score_case("bl", "c2", [F(line=300)], GT, 0.01, 5),
    ]
    s = {x.system: x for x in summarize(results)}
    assert s["ma"].catch_rate == 0.5 and s["ma"].false_positive_rate == 0.5
    assert s["bl"].catch_rate == 0.5 and s["bl"].false_positive_rate == 0.75
    assert abs(s["ma"].avg_cost_usd - 0.15) < 1e-9
    assert "| ma | 2 | 50% | 50% |" in markdown_table(list(s.values()))


def test_is_test_path():
    assert is_test_path("tests/test_core.py") and is_test_path("src/x/test_y.py")
    assert not is_test_path("src/click/core.py")


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   capture_output=True)


def test_mine_and_prepare_roundtrip(tmp_path):
    """Build a tiny repo with a bug-fix commit, mine it, and reintroduce the bug."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    src = tmp_path / "pkg" / "calc.py"
    src.write_text("def mean(xs):\n    return sum(xs) / len(xs)\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "initial")

    src.write_text("def mean(xs):\n    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)\n")
    (tmp_path / "tests" / "test_calc.py").write_text("from pkg.calc import mean\ndef test_empty():\n    assert mean([]) == 0.0\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "Fix crash on empty input")

    cases = mine_repo("o/proj", tmp_path, limit=5)
    assert len(cases) == 1
    case: BugCase = cases[0]
    assert case.src_files == ["pkg/calc.py"] and case.test_files == ["tests/test_calc.py"]

    prepared = prepare_case(case, tmp_path)
    # base keeps the fix but not its test; bug branch reverts the fix
    show = lambda sha, p: subprocess.run(["git", "show", f"{sha}:{p}"], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert "if not xs" in show(prepared["base_sha"], "pkg/calc.py").stdout
    assert show(prepared["base_sha"], "tests/test_calc.py").returncode != 0
    assert "if not xs" not in show(prepared["bug_sha"], "pkg/calc.py").stdout
    assert prepared["ground_truth"] == {"pkg/calc.py": [2]}  # deletion anchor where the guard was


def test_worktree_relative_dest_lands_in_cwd_not_in_clone(tmp_path, monkeypatch):
    """Regression: git runs with cwd=clone, so a relative dest used to be created inside the clone."""
    from reviewer.benchmark.run import worktree

    clone = tmp_path / "clone"
    clone.mkdir()
    _git(clone, "init", "-q")
    (clone / "a.py").write_text("x = 1\n")
    _git(clone, "add", ".")
    _git(clone, "commit", "-q", "-m", "init")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True,
                         check=True).stdout.strip()

    monkeypatch.chdir(tmp_path)
    wt = worktree(clone, sha, __import__("pathlib").Path("work/head"))
    assert wt == (tmp_path / "work" / "head").resolve()
    assert (wt / "a.py").exists()
    assert not (clone / "work").exists()


def test_bug_fix_subject_filter():
    from reviewer.benchmark.mine import is_bug_fix_subject

    assert is_bug_fix_subject("Fix crash when option has no default")
    assert is_bug_fix_subject("fix: wrong exit code")
    assert not is_bug_fix_subject("Add more tests to prevent regression")
    assert not is_bug_fix_subject("Fix pager test race by raising before yield")
    assert is_bug_fix_subject("fixed incorrect exit code")
    assert not is_bug_fix_subject("Deprecate `isolated_filesystem` and document its limits")
    assert not is_bug_fix_subject("Fix typo in error message")
    assert not is_bug_fix_subject("fix mypy errors")
    assert not is_bug_fix_subject("Fix types for class Meta options (#2806)")
    assert not is_bug_fix_subject("Fix type errors in Pyright (#999)")
    assert not is_bug_fix_subject("Add prefix option")  # 'fix' inside a word doesn't count

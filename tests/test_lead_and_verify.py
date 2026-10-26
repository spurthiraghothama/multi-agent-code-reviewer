from reviewer.agents.lead import apply_duplicate_groups, deterministic_dedupe, rank, should_post
from reviewer.models import Finding, Severity
from reviewer.models import VerificationStatus as VS
from reviewer.sandbox.runner import CaseOutcome, SandboxRun, parse_junit
from reviewer.sandbox.verify import apply_verdicts, classify, suite_findings


def F(**kw) -> Finding:
    base = dict(agent="bug", category="logic_error", file="a.py", line=10, severity="high", confidence=0.9,
                description="off by one", suggested_fix="fix", evidence="ev")
    return Finding(**{**base, **kw})


def test_dedupe_keeps_strongest_of_near_duplicates():
    weak = F(confidence=0.5)
    strong = F(agent="test", line=12, confidence=0.95)
    other_file = F(file="b.py")
    kept = deterministic_dedupe([weak, strong, other_file])
    assert strong in kept and other_file in kept and weak not in kept


def test_dedupe_keeps_distinct_issues_in_same_area():
    a = F(category="logic_error", description="wrong comparison operator in loop bound")
    b = F(category="n_plus_one", line=11, description="database query issued for every item")
    assert len(deterministic_dedupe([a, b])) == 2


def test_rank_prefers_confirmed_then_severity():
    unverified_crit = F(severity="critical")
    confirmed_med = F(severity="medium", status=VS.confirmed)
    assert rank([unverified_crit, confirmed_med])[0] is confirmed_med


def test_posting_policy():
    assert should_post(F(status=VS.confirmed, severity="low", confidence=0.1), 0.8)
    assert should_post(F(severity="high", confidence=0.85), 0.8)
    assert not should_post(F(severity="medium", confidence=0.99), 0.8)
    assert not should_post(F(severity="critical", confidence=0.99, status=VS.not_reproduced), 0.8)
    assert not should_post(F(category="missing_test", severity="high", status=VS.not_reproduced), 0.8)
    assert should_post(F(category="missing_test", status=VS.confirmed), 0.8)


def test_llm_duplicate_groups_keep_strongest_regardless_of_order():
    a, b = F(confidence=0.6), F(agent="performance", line=40, confidence=0.9, status=VS.confirmed)
    out = apply_duplicate_groups([a, b], [[a.id, b.id], ["nonexistent"]])
    assert out == [b]


def _run(outcome, message="", name="test_x", fid=""):
    return SandboxRun(generated=[CaseOutcome(f"_ai_review_tests.test_ai_{fid}", name, outcome, message)])


def test_classify_separates_broken_tests_from_real_failures():
    assert classify([CaseOutcome("c", "t", "failed", "AssertionError: 3 != 4")]) == "failed"
    assert classify([CaseOutcome("c", "t", "failed", "ModuleNotFoundError: No module named 'x'")]) == "broken"
    assert classify([CaseOutcome("c", "t", "error", "fixture 'db' not found")]) == "broken"
    assert classify([CaseOutcome("c", "t", "passed")]) == "passed"
    assert classify([]) == "missing"


def test_differential_verdicts():
    f = F(test_code="def test_x(): assert False")
    no_test = F(line=50)
    # fails on head, passes on base -> confirmed regression
    assert apply_verdicts([f], _run("failed", "AssertionError", fid=f.id), _run("passed", fid=f.id))[0].status == VS.confirmed
    # fails on both -> pre-existing, not blamed on the PR
    assert apply_verdicts([f], _run("failed", "AssertionError", fid=f.id),
                          _run("failed", "AssertionError", fid=f.id))[0].status == VS.unverified
    # new function: test can't import on base -> still confirmed
    assert apply_verdicts([f], _run("failed", "AssertionError", fid=f.id),
                          _run("failed", "ImportError: cannot import", fid=f.id))[0].status == VS.confirmed
    assert apply_verdicts([f], _run("passed", fid=f.id), None)[0].status == VS.not_reproduced
    assert apply_verdicts([no_test], _run("passed"), None)[0].status == VS.unverified


def test_suite_regressions_only():
    head = SandboxRun(suite=[CaseOutcome("tests.test_a", "test_new_fail", "failed", "assert 1 == 2", "tests/test_a.py", 5),
                             CaseOutcome("tests.test_a", "test_old_fail", "failed", "boom", "tests/test_a.py", 9)])
    base = SandboxRun(suite=[CaseOutcome("tests.test_a", "test_new_fail", "passed"),
                             CaseOutcome("tests.test_a", "test_old_fail", "failed", "boom")])
    found = suite_findings(head, base)
    assert len(found) == 1 and "test_new_fail" in found[0].description
    assert found[0].status == VS.confirmed and found[0].severity == Severity.high


def test_parse_junit():
    xml = """<testsuites><testsuite>
      <testcase classname="_ai_review_tests.test_ai_abc" name="test_one" file="_ai_review_tests/test_ai_abc.py" line="3"/>
      <testcase classname="_ai_review_tests.test_ai_abc" name="test_two"><failure message="AssertionError">trace</failure></testcase>
      <testcase classname="" name="_ai_review_tests.test_ai_def"><error message="collection failure">ImportError</error></testcase>
    </testsuite></testsuites>"""
    out = parse_junit(xml)
    assert [o.outcome for o in out] == ["passed", "failed", "error"]
    assert out[0].line == 3 and "trace" in out[1].message

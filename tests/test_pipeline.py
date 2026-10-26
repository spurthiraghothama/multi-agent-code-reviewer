"""End-to-end pipeline test with a scripted LLM and a fake sandbox (no network, no Docker)."""

import asyncio

from reviewer.agents.cost import cost_usd
from reviewer.agents.llm import LLMError
from reviewer.config import Settings
from reviewer.models import ChangedFile, Finding, ReviewRequest, TokenUsage
from reviewer.models import VerificationStatus as VS
from reviewer.pipeline import ReviewPipeline
from tests.test_indexing import make_repo


def finding(**kw):
    base = dict(category="logic_error", file="shop/pricing.py", line=2, severity="high", confidence=0.6,
                description="discount over 100% yields negative price", suggested_fix="clamp pct",
                evidence="apply_discount(10, 150) == -5", test_code="")
    return {**base, **kw}


class FakeLLM:
    def __init__(self, fail: set[str] = frozenset()):
        self.calls: list[str] = []
        self.fail = fail
        self.contexts: dict[str, str] = {}

    async def complete_json(self, *, agent, system, context, instructions, schema):
        self.calls.append(agent)
        self.contexts[agent] = context
        usage = TokenUsage(agent=agent, model="claude-opus-5-5", input_tokens=1000, output_tokens=200,
                           cost_usd=cost_usd("claude-opus-5-5", 1000, 200))
        if agent in self.fail:
            raise LLMError(f"{agent} exploded", usage)
        if agent == "bug":
            return {"findings": [
                finding(test_code="from shop.pricing import apply_discount\ndef test_x():\n    assert apply_discount(10, 150) >= 0"),
                finding(file="hallucinated/file.py"),
            ]}, usage
        if agent == "performance":
            return {"findings": [finding(category="unnecessary_work", line=9, severity="low", confidence=0.4,
                                         description="sum over generator recomputed")]}, usage
        if agent == "test":
            return {"findings": [finding(category="missing_test", severity="medium", line=1,
                                         description="no test for zero discount",
                                         test_code="def test_zero():\n    assert True")]}, usage
        if agent == "lead":
            return {"duplicate_groups": [], "summary_markdown": "One confirmed bug."}, usage
        raise AssertionError(agent)


class FakeVerifier:
    """Pretends the bug's test failed and the test agent's test passed."""

    def __init__(self):
        self.seen: list[Finding] = []

    async def verify(self, req, findings):
        self.seen = findings
        out = []
        for f in findings:
            if f.agent == "bug" and f.test_code:
                f = f.model_copy(update={"status": VS.confirmed})
            elif f.agent == "test":
                f = f.model_copy(update={"status": VS.not_reproduced})
            out.append(f)
        return out


def make_request(tmp_path):
    repo = make_repo(tmp_path)
    return ReviewRequest(repo_full_name="o/r", base_sha="a", head_sha="b", repo_dir=repo, title="Change discount",
                         changed_files=[ChangedFile(path="shop/pricing.py", status="modified",
                                                    patch="@@ -2 +2 @@\n-x\n+y", added_lines=[2],
                                                    commentable_lines=[1, 2, 3])])


def test_pipeline_end_to_end(tmp_path):
    llm, verifier = FakeLLM(), FakeVerifier()
    pipeline = ReviewPipeline(Settings(), llm, verifier, use_vector_index=False)
    result = asyncio.run(pipeline.review(make_request(tmp_path)))

    assert set(llm.calls[:3]) == {"bug", "performance", "test"} and llm.calls[3] == "lead"
    # specialists received callers from the call graph, not just the diff
    assert "Caller of `apply_discount`" in llm.contexts["bug"]
    # hallucinated path dropped before verification
    assert all(f.file != "hallucinated/file.py" for f in verifier.seen)
    # confirmed bug posted; low-severity perf note and passing missing-test are not
    assert [f.agent for f in result.posted_findings] == ["bug"]
    assert result.posted_findings[0].status == VS.confirmed
    # passing generated test is offered as a suggested test in the summary
    assert "One confirmed bug." in result.summary_markdown and "def test_zero" in result.summary_markdown
    assert len(result.usage) == 4 and result.total_cost_usd > 0


def test_pipeline_survives_agent_failure(tmp_path):
    llm = FakeLLM(fail={"performance", "lead"})
    result = asyncio.run(ReviewPipeline(Settings(), llm, FakeVerifier(), use_vector_index=False)
                         .review(make_request(tmp_path)))
    assert [f.agent for f in result.posted_findings] == ["bug"]
    assert "Automated review found 1 issue" in result.summary_markdown  # deterministic fallback summary
    assert len(result.usage) == 4  # failed calls still counted in cost


def test_cost_calculation():
    # 1M input + 1M output on Opus 5.5 = $4 + $20
    assert cost_usd("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    # cache reads are much cheaper than fresh input
    assert cost_usd("claude-opus-5-5", 0, 0, cache_read_tokens=1_000_000) == 0.2
    assert cost_usd("unknown-model", 10, 10) == 0.0

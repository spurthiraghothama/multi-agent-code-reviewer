"""benchmark.run: stops cleanly on a used-up daily quota and resumes where it left off."""

import asyncio

from reviewer.benchmark import run as run_mod
from reviewer.models import ReviewResult

CASES = [{"id": f"c{i}", "ground_truth": {"a.py": [1]}} for i in range(1, 4)]


def install_fakes(monkeypatch, quota_at: str | None):
    calls = []

    class FakePipeline:
        def __init__(self, *a, **k):
            pass

        async def review(self, req):
            calls.append(req)
            errors = ["bug: Gemini free-tier DAILY quota used up"] if req == quota_at else []
            return ReviewResult(summary_markdown="", posted_findings=[], all_findings=[], usage=[], latency_s=1,
                                errors=errors)

    async def fake_baseline(req, llm):
        return [], [], 1.0, ""

    monkeypatch.setattr(run_mod, "make_llm", lambda s: None)
    monkeypatch.setattr(run_mod, "ReviewPipeline", FakePipeline)
    monkeypatch.setattr(run_mod, "run_baseline", fake_baseline)
    monkeypatch.setattr(run_mod, "build_request", lambda case, clones, root: case["id"])
    return calls


def test_quota_stops_run_and_resume_continues(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, quota_at="c2")
    asyncio.run(run_mod.run(CASES, tmp_path, tmp_path, False, {"multi_agent", "baseline"}))
    first, _ = run_mod.load_previous(tmp_path)
    assert calls == ["c1", "c2"]
    assert {r.case_id for r in first} == {"c1"}  # c2 was NOT recorded as a fake "missed" case

    calls = install_fakes(monkeypatch, quota_at=None)
    asyncio.run(run_mod.run(CASES, tmp_path, tmp_path, False, {"multi_agent", "baseline"}, resume=True))
    final, _ = run_mod.load_previous(tmp_path)
    assert calls == ["c2", "c3"]  # c1 not re-run
    assert sorted((r.case_id, r.system) for r in final) == sorted(
        (c, s) for c in ["c1", "c2", "c3"] for s in ["multi_agent", "baseline"])
    assert (tmp_path / "summary.md").exists()

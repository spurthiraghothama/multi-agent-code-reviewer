"""FastAPI app: receives GitHub webhooks and runs reviews in the background.

GitHub expects a webhook response within 10 seconds, and a review takes minutes. So the endpoint
only validates and acknowledges (202), and the review runs as a background task.

Trade-off: FastAPI BackgroundTasks run in the same process. If the server restarts mid-review,
that review is lost. For production you'd push a job onto a durable queue (Redis + a worker, SQS,
...) and have workers consume it; the pipeline code wouldn't change. For a single-repo bot this
is simpler and good enough.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from reviewer.agents.llm import make_llm
from reviewer.config import Settings, get_settings
from reviewer.github.client import GitHubClient
from reviewer.github.webhook import should_review, verify_signature
from reviewer.pipeline import ReviewPipeline
from reviewer.sandbox.verify import NoopVerifier, SandboxVerifier, Verifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("reviewer")

app = FastAPI(title="Multi-agent code reviewer")

_seen_deliveries: OrderedDict[str, None] = OrderedDict()  # GitHub retries deliveries; dedupe them
_review_slots = asyncio.Semaphore(2)                       # bound concurrent reviews (LLM + Docker cost)


def make_verifier(settings: Settings) -> Verifier:
    try:
        return SandboxVerifier(settings)
    except Exception as e:  # Docker not reachable
        log.warning("Docker unavailable (%s); findings will not be sandbox-verified", e)
        return NoopVerifier()


async def run_review(repo_full_name: str, pr_number: int) -> None:
    settings = get_settings()
    if not settings.github_token:
        log.error("GITHUB_TOKEN is not set; cannot review %s#%d", repo_full_name, pr_number)
        return
    async with _review_slots:
        gh = GitHubClient(settings)
        try:
            req = await asyncio.to_thread(gh.build_review_request, repo_full_name, pr_number)
            pipeline = ReviewPipeline(settings, make_llm(settings), make_verifier(settings))
            result = await pipeline.review(req)
            log.info("Review %s#%d: %d findings posted (of %d), $%.4f, %.1fs", repo_full_name, pr_number,
                     len(result.posted_findings), len(result.all_findings), result.total_cost_usd,
                     result.latency_s)
            summary = result.summary_markdown + (
                f"\n\n<sub>Cost ${result.total_cost_usd:.3f} · {result.total_tokens:,} tokens · "
                f"{result.latency_s:.0f}s</sub>")
            if settings.dry_run:
                log.info("DRY RUN - would post:\n%s", summary)
            else:
                await asyncio.to_thread(gh.post_review, req, summary, result.posted_findings)
        except Exception:
            log.exception("Review of %s#%d failed", repo_full_name, pr_number)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.post("/webhook", status_code=202)
async def webhook(
    request: Request,
    background: BackgroundTasks,
    x_github_event: str = Header(default=""),
    x_hub_signature_256: str | None = Header(default=None),
    x_github_delivery: str = Header(default=""),
) -> dict:
    body = await request.body()  # raw bytes: the signature is over these exact bytes
    if not verify_signature(get_settings().github_webhook_secret, body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="invalid signature")

    if x_github_event == "ping":
        return {"status": "pong"}
    if x_github_delivery in _seen_deliveries:
        return {"status": "duplicate delivery"}
    _seen_deliveries[x_github_delivery] = None
    if len(_seen_deliveries) > 1000:
        _seen_deliveries.popitem(last=False)

    payload = json.loads(body)
    if not should_review(x_github_event, payload):
        return {"status": "ignored"}

    repo = payload["repository"]["full_name"]
    number = payload["pull_request"]["number"]
    background.add_task(run_review, repo, number)
    return {"status": "queued", "repo": repo, "pr": number}

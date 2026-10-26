"""Run the multi-agent system and the baseline on every prepared case; write a table + CSVs.

    python -m reviewer.benchmark.run --prepared benchmark/prepared.json --out results/

Reviews run locally from git worktrees (no GitHub API, no posting), through the same
ReviewPipeline the webhook uses. Cases run sequentially: it keeps latency numbers honest (no
contention between cases) and keeps API rate limits and Docker load predictable.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import subprocess
from pathlib import Path

import yaml

from reviewer.agents.llm import make_llm
from reviewer.benchmark.baseline import run_baseline
from reviewer.benchmark.metrics import CaseResult, markdown_table, score_case, summarize, write_csv
from reviewer.config import get_settings
from reviewer.github.diff import local_changed_files
from reviewer.models import ReviewRequest
from reviewer.pipeline import ReviewPipeline
from reviewer.sandbox.verify import NoopVerifier, SandboxVerifier

log = logging.getLogger(__name__)


def worktree(repo_dir: Path, sha: str, dest: Path) -> Path:
    # git runs with cwd=repo_dir, so a relative dest would be created *inside the clone*.
    dest = dest.resolve()
    if dest.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(dest)], cwd=repo_dir, check=False)
    subprocess.run(["git", "worktree", "add", "-f", "--detach", str(dest), sha], cwd=repo_dir, check=True,
                   capture_output=True)
    return dest


def build_request(case: dict, clones: Path, wt_root: Path) -> ReviewRequest:
    repo_dir = clones / case["repo"].replace("/", "__")
    head = worktree(repo_dir, case["bug_sha"], wt_root / f"{case['id']}-head")
    base = worktree(repo_dir, case["base_sha"], wt_root / f"{case['id']}-base")
    return ReviewRequest(
        repo_full_name=f"bench/{case['repo']}", base_sha=case["base_sha"], head_sha=case["bug_sha"],
        repo_dir=head, base_dir=base,
        title="Update " + ", ".join(Path(p).name for p in case["src_files"]),
        changed_files=local_changed_files(repo_dir, case["base_sha"], case["bug_sha"]),
    )


QUOTA_MARKER = "DAILY quota"


class QuotaExhausted(RuntimeError):
    pass


def _check_quota(errors: list[str]) -> None:
    # A review whose agents hit the daily quota would be scored as "missed everything" - that would
    # corrupt the results, so stop instead and let --resume continue tomorrow.
    if any(QUOTA_MARKER in e for e in errors):
        raise QuotaExhausted(next(e for e in errors if QUOTA_MARKER in e))


def load_previous(out: Path) -> tuple[list[CaseResult], list[dict]]:
    results: list[CaseResult] = []
    if (out / "per_case.csv").exists():
        with (out / "per_case.csv").open(newline="") as fh:
            for row in csv.DictReader(fh):
                results.append(CaseResult(
                    system=row["system"], case_id=row["case_id"], caught=row["caught"] == "True",
                    posted=int(row["posted"]), false_positives=int(row["false_positives"]),
                    cost_usd=float(row["cost_usd"]), latency_s=float(row["latency_s"]),
                    tokens=int(row.get("tokens") or 0), error=row.get("error", "")))
    details = json.loads((out / "details.json").read_text()) if (out / "details.json").exists() else []
    return results, details


def finish(results: list[CaseResult], details: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_case.csv", results)
    (out / "details.json").write_text(json.dumps(details, indent=2))
    summaries = summarize(results)
    write_csv(out / "summary.csv", summaries)
    table = markdown_table(summaries)
    (out / "summary.md").write_text(table + "\n")
    print(table)


async def run(prepared: list[dict], clones: Path, out: Path, use_sandbox: bool, systems: set[str],
              resume: bool = False) -> None:
    settings = get_settings()
    llm = make_llm(settings)
    verifier = SandboxVerifier(settings) if use_sandbox else NoopVerifier()
    pipeline = ReviewPipeline(settings, llm, verifier)
    results, details = load_previous(out) if resume else ([], [])

    # A case is done when every requested system has an error-free result for it; anything else reruns.
    done = {cid for cid in {r.case_id for r in results}
            if all(any(r.case_id == cid and r.system == s and not r.error for r in results) for s in systems)}
    results = [r for r in results if r.case_id in done]
    details = [d for d in details if d["case"] in done]
    todo = [c for c in prepared if c["id"] not in done]
    if resume:
        log.info("Resuming: %d cases already done, %d to go", len(done), len(todo))

    try:
        for i, case in enumerate(todo, 1):
            log.info("[%d/%d] %s", i, len(todo), case["id"])
            req = build_request(case, clones, settings.workspace_dir / "bench")
            gt = case["ground_truth"]
            case_results, case_details = [], []

            if "multi_agent" in systems:
                try:
                    res = await pipeline.review(req)
                    _check_quota(res.errors)
                    case_results.append(score_case("multi_agent", case["id"], res.posted_findings, gt,
                                                   res.total_cost_usd, res.latency_s, tokens=res.total_tokens))
                    case_details.append({"case": case["id"], "system": "multi_agent",
                                         "posted": [f.model_dump(mode="json") for f in res.posted_findings],
                                         "all": [f.model_dump(mode="json") for f in res.all_findings],
                                         "errors": res.errors})
                except QuotaExhausted:
                    raise
                except Exception as e:  # one bad case shouldn't kill the whole benchmark run
                    log.exception("multi_agent failed on %s", case["id"])
                    case_results.append(score_case("multi_agent", case["id"], [], gt, 0.0, 0.0, error=str(e)))

            if "baseline" in systems:
                try:
                    findings, usage, latency, err = await run_baseline(req, llm)
                except Exception as e:  # same isolation as above
                    log.exception("baseline failed on %s", case["id"])
                    findings, usage, latency, err = [], [], 0.0, str(e)
                _check_quota([err])
                case_results.append(score_case("baseline", case["id"], findings, gt, sum(u.cost_usd for u in usage),
                                               latency, error=err,
                                               tokens=sum(u.input_tokens + u.output_tokens + u.cache_read_input_tokens
                                                          + u.cache_creation_input_tokens for u in usage)))
                case_details.append({"case": case["id"], "system": "baseline",
                                     "posted": [f.model_dump(mode="json") for f in findings]})

            # Only whole cases are recorded, so both systems are always compared on the same cases.
            results += case_results
            details += case_details
            write_csv(out / "per_case.csv", results)  # incremental: a crash keeps finished cases
    except QuotaExhausted as e:
        log.warning("Stopping: %s", e)
        log.warning("Finished cases are saved. Run again later with --resume to continue.")
    finish(results, details, out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("benchmark/config.yaml"))
    ap.add_argument("--prepared", type=Path, default=Path("benchmark/prepared.json"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    ap.add_argument("--no-sandbox", action="store_true")
    ap.add_argument("--systems", default="multi_agent,baseline")
    ap.add_argument("--resume", action="store_true", help="skip cases already finished in --out")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prepared = json.loads(args.prepared.read_text())
    if args.limit:
        prepared = prepared[: args.limit]
    asyncio.run(run(prepared, Path(cfg.get("clone_dir", "data/bench_clones")), args.out,
                    not args.no_sandbox, set(args.systems.split(",")), resume=args.resume))


if __name__ == "__main__":
    main()

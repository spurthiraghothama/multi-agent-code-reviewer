"""Scoring a review against ground truth.

Definitions (state these explicitly in interviews - metric definitions are where benchmarks lie):

  caught            a case is caught if ANY posted finding is in a ground-truth file within
                    +/- LINE_TOLERANCE lines of a line the bug PR changed.
  catch rate        caught cases / all cases                          (recall at the case level)
  false positives   posted findings that don't match the ground truth
  FP rate           false positives / all posted findings             (1 - precision)
  cost, tokens,     averaged per review (on Gemini's free tier cost is $0, so tokens
  latency           are the meaningful efficiency number)

Caveat: an "FP" is only false relative to the one known bug. The reviewer may have found a real,
different issue. So the FP rate is an upper bound; spot-check a sample by hand when reporting.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from reviewer.models import Finding

LINE_TOLERANCE = 5


def matches(f: Finding, ground_truth: dict[str, list[int]], tolerance: int = LINE_TOLERANCE) -> bool:
    lines = ground_truth.get(f.file)
    return bool(lines) and any(abs(f.line - gl) <= tolerance for gl in lines)


@dataclass
class CaseResult:
    system: str
    case_id: str
    caught: bool
    posted: int
    false_positives: int
    cost_usd: float
    latency_s: float
    tokens: int = 0
    error: str = ""


def score_case(system: str, case_id: str, posted: list[Finding], ground_truth: dict[str, list[int]],
               cost: float, latency: float, error: str = "", tokens: int = 0) -> CaseResult:
    hits = [matches(f, ground_truth) for f in posted]
    return CaseResult(system, case_id, any(hits), len(posted), hits.count(False), cost, latency, tokens, error)


@dataclass
class Summary:
    system: str
    cases: int
    catch_rate: float
    false_positive_rate: float
    avg_findings_posted: float
    avg_cost_usd: float
    avg_tokens: float
    avg_latency_s: float
    p50_latency_s: float


def summarize(results: list[CaseResult]) -> list[Summary]:
    out = []
    for system in sorted({r.system for r in results}):
        rs = [r for r in results if r.system == system]
        posted = sum(r.posted for r in rs)
        out.append(Summary(
            system=system, cases=len(rs),
            catch_rate=sum(r.caught for r in rs) / len(rs),
            false_positive_rate=(sum(r.false_positives for r in rs) / posted) if posted else 0.0,
            avg_findings_posted=posted / len(rs),
            avg_cost_usd=statistics.mean(r.cost_usd for r in rs),
            avg_tokens=statistics.mean(r.tokens for r in rs),
            avg_latency_s=statistics.mean(r.latency_s for r in rs),
            p50_latency_s=statistics.median(r.latency_s for r in rs),
        ))
    return out


def markdown_table(summaries: list[Summary]) -> str:
    head = ("| System | Cases | Catch rate | FP rate | Findings/review | Avg cost | Avg tokens | Avg latency "
            "| p50 latency |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [f"| {s.system} | {s.cases} | {s.catch_rate:.0%} | {s.false_positive_rate:.0%} | "
            f"{s.avg_findings_posted:.1f} | ${s.avg_cost_usd:.3f} | {s.avg_tokens:,.0f} | {s.avg_latency_s:.1f}s | "
            f"{s.p50_latency_s:.1f}s |"
            for s in summaries]
    return "\n".join([head, sep, *rows])


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(r) for r in rows]
    if not dicts:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dicts[0]))
        w.writeheader()
        w.writerows(dicts)

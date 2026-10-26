"""Turns sandbox test outcomes into verdicts on findings.

The rule for "confirmed" is intentionally strict, because a confirmed finding is posted no matter
what its confidence was:

  1. the finding's generated test ran and FAILED on the PR head (an assertion or a runtime
     exception inside the test body), and
  2. the failure is not the test being broken: collection errors, ImportError/SyntaxError/NameError,
     missing fixtures etc. mean the LLM wrote a bad test, which says nothing about the code, and
  3. (differential mode) the same test does NOT fail the same way on the base commit. If it fails
     on base too, the problem predates the PR - not something this PR should be blamed for. If the
     test can't even import on base (e.g. the function is new), that's fine: it can't be pre-existing.

Repo-suite failures are also turned into findings, but only if they don't also fail on base.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Protocol

from reviewer.config import Settings
from reviewer.models import Finding, ReviewRequest, Severity, VerificationStatus
from reviewer.sandbox.runner import CaseOutcome, DockerSandbox, SandboxRun

log = logging.getLogger(__name__)

BROKEN_PATTERNS = re.compile(
    r"(ImportError|ModuleNotFoundError|SyntaxError|IndentationError|NameError|fixture '.*' not found|"
    r"AttributeError: module|collection failure|ERROR collecting|TypeError: .*missing \d+ required positional)"
)


class Verifier(Protocol):
    async def verify(self, req: ReviewRequest, findings: list[Finding]) -> list[Finding]: ...


def generated_test_filename(f: Finding) -> str:
    return f"test_ai_{f.id}.py"


def classify(outcomes: list[CaseOutcome]) -> str:
    """Collapse all test-cases of one generated file into: passed | failed | broken | missing."""
    if not outcomes:
        return "missing"
    if any(o.outcome == "error" or (o.outcome == "failed" and BROKEN_PATTERNS.search(o.message))
           for o in outcomes):
        return "broken"
    if any(o.outcome == "failed" for o in outcomes):
        return "failed"
    if all(o.outcome == "skipped" for o in outcomes):
        return "missing"
    return "passed"


def outcomes_for(f: Finding, run: SandboxRun) -> list[CaseOutcome]:
    key = f"test_ai_{f.id}"
    return [o for o in run.generated if key in o.classname or key in o.name or key in o.file]


def apply_verdicts(findings: list[Finding], head: SandboxRun, base: SandboxRun | None) -> list[Finding]:
    out = []
    for f in findings:
        if not f.test_code.strip():
            out.append(f)
            continue
        head_v = classify(outcomes_for(f, head))
        base_v = classify(outcomes_for(f, base)) if base is not None else None
        log_lines = [f"head: {head_v}"] + ([f"base: {base_v}"] if base_v else [])
        if head.timed_out:
            log_lines.append("sandbox timed out")

        if head_v == "failed" and base_v == "failed":
            status = VerificationStatus.unverified
            log_lines.append("test fails on base too: pre-existing, not introduced by this PR")
        elif head_v == "failed":
            status = VerificationStatus.confirmed
        elif head_v == "passed":
            status = VerificationStatus.not_reproduced
        elif head_v == "broken":
            status = VerificationStatus.broken_test
        else:
            status = VerificationStatus.unverified
        msg = next((o.message for o in outcomes_for(f, head) if o.outcome != "passed"), "")
        out.append(f.model_copy(update={"status": status,
                                        "verification_log": "; ".join(log_lines) + (f"\n{msg[:800]}" if msg else "")}))
    return out


def suite_findings(head: SandboxRun, base: SandboxRun | None) -> list[Finding]:
    base_failed = {(o.classname, o.name) for o in (base.suite if base else []) if o.outcome in {"failed", "error"}}
    found = []
    for o in head.suite:
        if o.outcome != "failed" or (o.classname, o.name) in base_failed:
            continue
        if base is not None and not base.suite:
            continue  # base suite didn't run - can't tell regressions from pre-existing failures
        found.append(Finding(
            agent="test_suite", category="failing_test", file=o.file or o.classname.replace(".", "/") + ".py",
            line=max(o.line, 1), severity=Severity.high, confidence=1.0,
            description=f"Existing test `{o.classname}::{o.name}` fails on this PR but passes on the base branch.",
            suggested_fix="Investigate the regression shown by the failing test.",
            evidence=o.message[:1500], status=VerificationStatus.confirmed,
        ))
    return found


class SandboxVerifier:
    def __init__(self, settings: Settings, sandbox: DockerSandbox | None = None):
        self.s = settings
        self.sandbox = sandbox or DockerSandbox(settings)

    def _verify_sync(self, req: ReviewRequest, findings: list[Finding]) -> list[Finding]:
        tests = {generated_test_filename(f): f.test_code for f in findings if f.test_code.strip()}
        image = self.sandbox.deps_image(req.repo_dir)
        head = self.sandbox.run(image, req.repo_dir, tests, run_suite=True)
        base = None
        if self.s.sandbox_differential and req.base_dir is not None:
            base = self.sandbox.run(image, req.base_dir, tests, run_suite=True)
        log.info("Sandbox: %d generated test cases on head, %d suite tests, timed_out=%s",
                 len(head.generated), len(head.suite), head.timed_out)
        return apply_verdicts(findings, head, base) + suite_findings(head, base)

    async def verify(self, req: ReviewRequest, findings: list[Finding]) -> list[Finding]:
        # The Docker SDK is blocking; run it in a worker thread so the event loop (and the webhook
        # server) stays responsive.
        try:
            return await asyncio.to_thread(self._verify_sync, req, findings)
        except Exception:  # sandbox failure must degrade to "unverified", not crash the review
            log.exception("Sandbox verification failed")
            return findings


class NoopVerifier:
    """Used when Docker isn't available (and by the baseline): every finding stays unverified."""

    async def verify(self, req: ReviewRequest, findings: list[Finding]) -> list[Finding]:
        return findings

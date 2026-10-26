"""Core data types shared by every module.

The Finding is the contract between agents, the sandbox, the lead agent, GitHub posting and the
benchmark. Keeping it in one place (and validating it with Pydantic) means a malformed LLM output
fails loudly at the boundary instead of crashing three modules later.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"

    @property
    def rank(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}[self.value]


class VerificationStatus(str, Enum):
    unverified = "unverified"      # no test was run for it
    confirmed = "confirmed"        # a generated test failed on the PR code (and passed on base, if checked)
    not_reproduced = "not_reproduced"  # the generated test passed on the PR code
    broken_test = "broken_test"    # the test itself was invalid (import/syntax/collection error)


class Finding(BaseModel):
    agent: str = Field(description="Which agent produced this finding")
    category: str = Field(description="e.g. logic_error, edge_case, broken_caller, n_plus_one, missing_test")
    file: str
    line: int
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    suggested_fix: str
    evidence: str = Field(description="Code-level reasoning: what input triggers it, which caller breaks, etc.")
    test_code: str = Field(default="", description="Optional pytest code that fails if the issue is real")
    status: VerificationStatus = VerificationStatus.unverified
    verification_log: str = ""

    @property
    def id(self) -> str:
        """Stable id so the same finding can be tracked through sandbox -> lead -> GitHub."""
        key = f"{self.agent}|{self.file}|{self.line}|{self.category}|{self.description[:80]}"
        return hashlib.sha1(key.encode()).hexdigest()[:10]


class ChangedFile(BaseModel):
    path: str
    status: str  # added | modified | removed | renamed
    patch: str = ""
    # Line numbers (in the new file) that were added or modified - these are where we may comment.
    added_lines: list[int] = Field(default_factory=list)
    # Lines that exist in the diff hunks on the RIGHT side (added + context). GitHub accepts
    # review comments only on lines that appear in the diff.
    commentable_lines: list[int] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    """Everything the pipeline needs about a PR, independent of where it came from.

    Both the GitHub webhook and the benchmark build one of these. That decoupling is what lets the
    benchmark run the exact same pipeline locally without opening real PRs.
    """

    repo_full_name: str
    pr_number: int | None = None
    base_sha: str
    head_sha: str
    repo_dir: Path  # local checkout at head_sha
    base_dir: Path | None = None  # local checkout at base_sha (for differential testing)
    title: str = ""
    body: str = ""
    changed_files: list[ChangedFile]


class TokenUsage(BaseModel):
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class ReviewResult(BaseModel):
    summary_markdown: str
    posted_findings: list[Finding]
    all_findings: list[Finding]
    usage: list[TokenUsage]
    latency_s: float
    errors: list[str] = Field(default_factory=list)  # agent failures the review survived

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage)

    @property
    def total_tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens
                   for u in self.usage)

"""Mine real bug-fix commits from open-source repos.

We want commits where (a) the fix is small and localised, so "did the reviewer point at the bug?"
has a clear answer, and (b) the fix came with a test, which is decent evidence that the commit
really fixed a bug (not a typo or refactor labelled "fix").

Heuristic filter, applied to `git log` of a local clone (no API rate limits):
  * message matches fix|bug|regression|crash|incorrect|wrong (case-insensitive), not a merge
  * touches 1-2 non-test .py files, with 1..MAX_SRC_LINES changed lines in total
  * touches at least one test file

    python -m reviewer.benchmark.mine --config benchmark/config.yaml --out benchmark/cases.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

FIX_RE = r"fix|bug|regression|crash|incorrect|wrong"
# git --grep searches the whole message body, which over-matches ("Deprecate X ... fixes #12").
# The subject line must *start* with a fix word (optionally after a "fix:" style prefix), and must not
# be about tests, flakiness, docs, typing or lint - those "fixes" aren't bugs in the library's behaviour.
SUBJECT_FIX_RE = re.compile(r"^\W*(bug\s*)?fix(es|ed)?\b", re.I)
SUBJECT_NOT_BUG_RE = re.compile(r"typo|\bdocs?\b|docstring|\blint|flake8|mypy|typing|type hint|\btypes?\b|pyright|deprecat|"
                                r"\btests?\b|flaky|changelog|admonition|\bci\b", re.I)
MAX_SRC_LINES = 30


def is_bug_fix_subject(subject: str) -> bool:
    return bool(SUBJECT_FIX_RE.search(subject)) and not SUBJECT_NOT_BUG_RE.search(subject)


def patch_id(repo_dir: Path, parent: str, sha: str, paths: list[str]) -> str:
    """Content hash of the source change. Identical patches (cherry-picks onto another branch,
    backports) share a patch-id, so the same bug isn't counted twice."""
    diff = git(repo_dir, "diff", parent, sha, "--", *paths)
    out = subprocess.run(["git", "patch-id", "--stable"], cwd=repo_dir, input=diff, capture_output=True,
                         text=True, check=True).stdout
    return out.split(" ", 1)[0]


@dataclass
class BugCase:
    id: str
    repo: str          # upstream, e.g. "pallets/click"
    fix_sha: str
    parent_sha: str
    message: str
    src_files: list[str]
    test_files: list[str]
    src_lines_changed: int


def git(repo_dir: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True).stdout


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    return any(p in {"tests", "test", "testing"} for p in parts) or Path(path).name.startswith("test_")


def ensure_clone(repo: str, root: Path) -> Path:
    dest = root / repo.replace("/", "__")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(dest)], check=True)
    return dest


def mine_repo(repo: str, repo_dir: Path, limit: int, max_commits: int = 4000) -> list[BugCase]:
    # Always walk the upstream default branch, not whatever is checked out: prepare.py leaves the
    # clone on an old benchmark branch, and mining from there would silently pick older commits.
    has_origin = subprocess.run(["git", "rev-parse", "--verify", "-q", "origin/HEAD"], cwd=repo_dir,
                                capture_output=True, check=False).returncode == 0
    rev = "origin/HEAD" if has_origin else "HEAD"
    log_out = git(repo_dir, "log", "--no-merges", "-i", "-E", f"--grep={FIX_RE}", f"-n{max_commits}",
                  "--format=%H%x09%P%x09%s", rev)
    cases: list[BugCase] = []
    seen_patches: set[str] = set()
    for line in log_out.splitlines():
        sha, parents, subject = line.split("\t", 2)
        if " " in parents or not parents or not is_bug_fix_subject(subject):
            continue
        numstat = git(repo_dir, "show", "--numstat", "--format=", sha)
        src, tests, src_lines = [], [], 0
        for row in numstat.splitlines():
            added, deleted, path = row.split("\t", 2)
            if not path.endswith(".py") or added == "-":
                continue
            if is_test_path(path):
                tests.append(path)
            else:
                src.append(path)
                src_lines += int(added) + int(deleted)
        if 1 <= len(src) <= 2 and tests and 1 <= src_lines <= MAX_SRC_LINES:
            pid = patch_id(repo_dir, parents, sha, src)
            if pid in seen_patches:
                continue
            seen_patches.add(pid)
            short = repo.split("/")[1]
            cases.append(BugCase(f"{short}-{sha[:8]}", repo, sha, parents, subject, src, tests, src_lines))
            if len(cases) >= limit:
                break
    return cases


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("benchmark/config.yaml"))
    ap.add_argument("--out", type=Path, default=Path("benchmark/cases.json"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    clones = Path(cfg.get("clone_dir", "data/bench_clones"))

    all_cases: list[BugCase] = []
    for repo in cfg["repos"]:
        repo_dir = ensure_clone(repo, clones)
        found = mine_repo(repo, repo_dir, cfg.get("cases_per_repo", 5))
        log.info("%s: %d cases", repo, len(found))
        all_cases += found
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(c) for c in all_cases], indent=2))
    print(f"Wrote {len(all_cases)} cases to {args.out}")


if __name__ == "__main__":
    main()

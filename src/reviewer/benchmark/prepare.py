"""Turn each mined bug-fix commit into a "PR that reintroduces the bug".

For each case we create two branches in the local clone:

  bench/<id>/base   = the fix commit, but with the fix's *test* changes reverted.
                      (Otherwise the existing suite would contain the regression test and the
                      sandbox would catch every bug trivially - that would measure nothing.)
  bench/<id>/bug    = base + the fix's *source* changes reverted, i.e. the bug is back.

The PR is bug -> base. Its diff is exactly the bug being (re)introduced, and the lines it adds are
the ground truth "bug location". Commit messages are neutral ("Update foo.py") so the reviewer
can't just read "reintroduce bug" from the title.

Known bias: a reverted fix can look like "deleting a guard clause", which is a recognisable
pattern. Real regressions are often subtler. That's why the report also includes the baseline -
both systems see the same diffs, so the comparison is fair even if absolute numbers are optimistic.

With --push-to OWNER/REPO the branches are pushed and real PRs are opened there (use a fork or an
empty repo you own), so you can watch the webhook flow end to end.

    python -m reviewer.benchmark.prepare --cases benchmark/cases.json [--push-to you/bench-repo]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import yaml

from reviewer.benchmark.mine import BugCase, git
from reviewer.github.diff import local_changed_files

log = logging.getLogger(__name__)


def _commit(repo_dir: Path, message: str) -> str:
    env_args = ["-c", "user.name=bench", "-c", "user.email=bench@example.com"]
    subprocess.run(["git", *env_args, "commit", "-q", "--allow-empty", "-m", message], cwd=repo_dir, check=True)
    return git(repo_dir, "rev-parse", "HEAD").strip()


def _restore_from(repo_dir: Path, sha: str, paths: list[str]) -> None:
    """Make each path look like it did at `sha` (deleting it if it didn't exist there)."""
    for p in paths:
        exists = subprocess.run(["git", "cat-file", "-e", f"{sha}:{p}"], cwd=repo_dir, capture_output=True, check=False).returncode == 0
        if exists:
            git(repo_dir, "checkout", sha, "--", p)
        else:
            git(repo_dir, "rm", "-q", "--ignore-unmatch", p)


def prepare_case(case: BugCase, repo_dir: Path) -> dict:
    base_branch, bug_branch = f"bench/{case.id}/base", f"bench/{case.id}/bug"
    git(repo_dir, "checkout", "-q", "-f", "-B", base_branch, case.fix_sha)
    _restore_from(repo_dir, case.parent_sha, case.test_files)
    base_sha = _commit(repo_dir, "Benchmark base")

    git(repo_dir, "checkout", "-q", "-B", bug_branch)
    _restore_from(repo_dir, case.parent_sha, case.src_files)
    bug_sha = _commit(repo_dir, "Update " + ", ".join(Path(p).name for p in case.src_files))

    changed = local_changed_files(repo_dir, base_sha, bug_sha)
    ground_truth = {f.path: f.added_lines for f in changed if f.path in case.src_files}
    return {**case.__dict__, "base_branch": base_branch, "bug_branch": bug_branch,
            "base_sha": base_sha, "bug_sha": bug_sha, "ground_truth": ground_truth}


def push_and_open_pr(prepared: dict, repo_dir: Path, target: str, token: str) -> int:
    from github import Auth, Github

    remote = f"https://x-access-token:{token}@github.com/{target}.git"
    git(repo_dir, "push", "-q", "-f", remote, f"{prepared['base_branch']}", f"{prepared['bug_branch']}")
    gh_repo = Github(auth=Auth.Token(token)).get_repo(target)
    pr = gh_repo.create_pull(title=f"Update {', '.join(Path(p).name for p in prepared['src_files'])}",
                             body="Small refactor.", base=prepared["base_branch"], head=prepared["bug_branch"])
    return pr.number


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("benchmark/config.yaml"))
    ap.add_argument("--cases", type=Path, default=Path("benchmark/cases.json"))
    ap.add_argument("--out", type=Path, default=Path("benchmark/prepared.json"))
    ap.add_argument("--push-to", default="", help="OWNER/REPO to push branches and open PRs in")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    clones = Path(cfg.get("clone_dir", "data/bench_clones"))
    prepared = []
    seen_truth: set[str] = set()
    for raw in json.loads(args.cases.read_text()):
        case = BugCase(**raw)
        repo_dir = clones / case.repo.replace("/", "__")
        try:
            p = prepare_case(case, repo_dir)
        except subprocess.CalledProcessError as e:
            log.warning("Skipping %s: %s", case.id, e.stderr or e)
            continue
        if not p["ground_truth"]:
            log.warning("Skipping %s: empty ground truth", case.id)
            continue
        truth_key = case.repo + json.dumps(p["ground_truth"], sort_keys=True)
        if truth_key in seen_truth:  # two commits, same bug location: count it once
            log.warning("Skipping %s: same bug location as an earlier case", case.id)
            continue
        seen_truth.add(truth_key)
        if args.push_to:
            from reviewer.config import get_settings
            p["pr_number"] = push_and_open_pr(p, repo_dir, args.push_to, get_settings().github_token)
            log.info("%s -> PR #%d", case.id, p["pr_number"])
        prepared.append(p)
    args.out.write_text(json.dumps(prepared, indent=2))
    print(f"Prepared {len(prepared)} cases -> {args.out}")


if __name__ == "__main__":
    main()

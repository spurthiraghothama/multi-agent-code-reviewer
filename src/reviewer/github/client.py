"""Thin wrapper over PyGithub + git.

Responsibilities:
  * fetch the PR's changed files and patches (GitHub API)
  * materialise local checkouts of the PR head and base (git) - needed for AST indexing and for
    running tests in the sandbox; the API alone can't give us a runnable repo
  * post one review containing inline comments + a summary body
"""

from __future__ import annotations

import base64
import logging
import subprocess
from pathlib import Path

from github import Auth, Github, GithubException

from reviewer.config import Settings
from reviewer.github.diff import changed_file_from_patch
from reviewer.models import ChangedFile, Finding, ReviewRequest

log = logging.getLogger(__name__)


class GitHubClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gh = Github(auth=Auth.Token(settings.github_token))

    # ---------------------------------------------------------------- fetching
    def build_review_request(self, repo_full_name: str, pr_number: int) -> ReviewRequest:
        repo = self.gh.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        changed: list[ChangedFile] = [
            changed_file_from_patch(f.filename, f.status, f.patch or "") for f in pr.get_files()
        ]
        head_dir, base_dir = self._checkout(repo_full_name, pr_number, pr.head.sha, pr.base.sha)
        return ReviewRequest(
            repo_full_name=repo_full_name, pr_number=pr_number,
            base_sha=pr.base.sha, head_sha=pr.head.sha,
            repo_dir=head_dir, base_dir=base_dir,
            title=pr.title or "", body=pr.body or "", changed_files=changed,
        )

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        # Pass the token as a one-off HTTP header instead of embedding it in the remote URL,
        # so it is never written to .git/config on disk.
        basic = base64.b64encode(f"x-access-token:{self.settings.github_token}".encode()).decode()
        cmd = ["git", "-c", f"http.extraHeader=Authorization: Basic {basic}", *args]
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout

    def _checkout(self, repo_full_name: str, pr_number: int, head_sha: str, base_sha: str) -> tuple[Path, Path]:
        # Absolute: git worktree commands run with cwd=mirror, so a relative path would land inside it.
        root = (self.settings.workspace_dir / repo_full_name.replace("/", "__")).resolve()
        mirror = root / "mirror"
        if not mirror.exists():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            # Partial clone: history without blobs; blobs are fetched lazily on checkout.
            self._git("clone", "--filter=blob:none", "--no-checkout",
                      f"https://github.com/{repo_full_name}.git", str(mirror))
        # pull/N/head works for PRs from forks too.
        self._git("fetch", "origin", base_sha, f"pull/{pr_number}/head", cwd=mirror)

        dirs = []
        for label, sha in (("head", head_sha), ("base", base_sha)):
            wt = root / f"pr{pr_number}-{label}"
            if wt.exists():
                self._git("worktree", "remove", "--force", str(wt), cwd=mirror)
            self._git("worktree", "add", "--detach", str(wt), sha, cwd=mirror)
            dirs.append(wt)
        return dirs[0], dirs[1]

    # ----------------------------------------------------------------- posting
    def post_review(self, req: ReviewRequest, summary: str, findings: list[Finding]) -> None:
        assert req.pr_number is not None
        repo = self.gh.get_repo(req.repo_full_name)
        pr = repo.get_pull(req.pr_number)
        commit = repo.get_commit(req.head_sha)

        commentable = {f.path: set(f.commentable_lines) for f in req.changed_files}
        inline, leftovers = [], []
        for f in findings:
            if f.line in commentable.get(f.file, set()):
                inline.append({"path": f.file, "line": f.line, "side": "RIGHT", "body": format_comment(f)})
            else:
                leftovers.append(f)

        body = summary
        if leftovers:
            body += "\n\n### Findings outside the diff\n" + "\n".join(
                f"- `{f.file}:{f.line}` **{f.severity.value}** - {f.description}" for f in leftovers
            )
        try:
            pr.create_review(commit=commit, body=body, event="COMMENT", comments=inline)
        except GithubException as e:
            # One bad line number makes GitHub reject the whole review (422). Degrade gracefully
            # to a summary-only review rather than posting nothing.
            log.warning("Inline review rejected (%s); posting summary only", e.status)
            all_lines = "\n".join(f"- `{f.file}:{f.line}` **{f.severity.value}** - {f.description}"
                                  for f in findings)
            pr.create_review(commit=commit, body=f"{summary}\n\n### Findings\n{all_lines}", event="COMMENT")


def format_comment(f: Finding) -> str:
    badge = {"confirmed": "Confirmed by failing test", "unverified": "Not verified by a test"}.get(
        f.status.value, f.status.value)
    parts = [
        f"**[{f.severity.value.upper()}] {f.category.replace('_', ' ')}** ({badge}, confidence {f.confidence:.0%})",
        "", f.description, "", f"**Evidence:** {f.evidence}", "", f"**Suggested fix:** {f.suggested_fix}",
    ]
    if f.test_code and f.status.value == "confirmed":
        parts += ["", "<details><summary>Reproducing test</summary>", "", "```python", f.test_code, "```",
                  "</details>"]
    return "\n".join(parts)

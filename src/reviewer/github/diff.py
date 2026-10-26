"""Unified-diff parsing.

We need two things from a patch:
  * added_lines       - lines the PR added/modified (used to find changed functions, and as
                        ground truth in the benchmark)
  * commentable_lines - every RIGHT-side line that appears in a hunk (added + context). GitHub
                        rejects an inline review comment on any line outside the diff, so we check
                        this before posting and fall back to the summary otherwise.

For pure deletions there is no added line to point at, so we record the new-file line right after
the deletion as an "anchor" in added_lines. Otherwise a PR that only deletes a guard clause would
look like it changed nothing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from reviewer.models import ChangedFile

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch: str) -> tuple[list[int], list[int]]:
    added: set[int] = set()
    commentable: set[int] = set()
    new_line = 0
    pending_deletion = False

    for raw in patch.splitlines():
        m = HUNK_RE.match(raw)
        if m:
            new_line = int(m.group(3))
            pending_deletion = False
            continue
        if raw.startswith("\\"):  # "\ No newline at end of file"
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            added.add(new_line)
            commentable.add(new_line)
            new_line += 1
            pending_deletion = False
        elif raw.startswith("-") and not raw.startswith("---"):
            pending_deletion = True
        else:  # context line
            if pending_deletion:
                added.add(new_line)  # anchor for a pure deletion
                pending_deletion = False
            commentable.add(new_line)
            new_line += 1

    if pending_deletion and new_line > 0:
        added.add(max(new_line - 1, 1))
    return sorted(added), sorted(commentable)


def changed_file_from_patch(path: str, status: str, patch: str) -> ChangedFile:
    added, commentable = parse_patch(patch or "")
    return ChangedFile(path=path, status=status, patch=patch or "", added_lines=added,
                       commentable_lines=commentable)


def split_git_diff(diff_text: str) -> dict[str, tuple[str, str]]:
    """Split `git diff` output into {path: (status, patch)}. Used by the benchmark (no GitHub API)."""
    files: dict[str, tuple[str, str]] = {}
    current: str | None = None
    status = "modified"
    buf: list[str] = []

    def flush() -> None:
        if current is not None:
            hunk_start = next((i for i, l in enumerate(buf) if l.startswith("@@")), len(buf))
            files[current] = (status, "\n".join(buf[hunk_start:]))

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            current = line.split(" b/", 1)[-1]
            status, buf = "modified", []
            continue
        if line.startswith("new file mode"):
            status = "added"
        elif line.startswith("deleted file mode"):
            status = "removed"
        elif line.startswith("rename to "):
            status = "renamed"
        buf.append(line)
    flush()
    return files


def local_changed_files(repo_dir: Path, base_sha: str, head_sha: str) -> list[ChangedFile]:
    out = subprocess.run(
        ["git", "diff", "--unified=3", f"{base_sha}...{head_sha}"],
        cwd=repo_dir, capture_output=True, text=True, check=True,
    ).stdout
    return [changed_file_from_patch(p, s, patch) for p, (s, patch) in split_git_diff(out).items()]

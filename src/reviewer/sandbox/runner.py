"""Docker sandbox for running untrusted code (the PR's code and LLM-written tests).

Threat model: both the PR and the generated tests are untrusted. A malicious PR could try to
exfiltrate secrets, mine crypto, fork-bomb, or fill the disk. Defences, layered:

  * network_disabled=True          - no exfiltration, no downloading payloads
  * mem_limit / nano_cpus          - bounded resource use
  * pids_limit                     - no fork bombs
  * cap_drop=ALL, no-new-privileges, non-root user
  * wall-clock timeout + kill      - no infinite loops
  * no host bind mounts            - files go in via `put_archive` (a tar stream) and results come
                                     out via `get_archive`. This also means the reviewer itself can
                                     run inside Docker (talking to the host daemon via the socket)
                                     without the classic "bind-mount path must be a host path" bug.
  * no secrets in the environment  - the container never sees GITHUB_TOKEN or ANTHROPIC_API_KEY.

Dependencies: tests need the repo's dependencies, but tests must run offline. So we split it into
two phases: (1) a *networked* install step that pip-installs the repo into a container and
snapshots it as an image, cached by a hash of the dependency files; (2) offline test runs from that
image. Trade-off: phase 1 executes the repo's setup.py with network access. It's a smaller window
(no generated code, no tests), but if you review PRs from untrusted forks you should disable it
(SANDBOX_INSTALL_DEPS=false) or pre-build dependency images yourself.
"""

from __future__ import annotations

import hashlib
import io
import logging
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout

from reviewer.config import Settings

log = logging.getLogger(__name__)

WORK = "/work"
REPO = f"{WORK}/repo"
GEN_DIR = "_ai_review_tests"
DEP_FILES = ("requirements.txt", "requirements-dev.txt", "requirements/test.txt", "pyproject.toml",
             "setup.py", "setup.cfg")
EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox", ".mypy_cache"}
SANDBOX_UID = 1000


@dataclass
class CaseOutcome:
    classname: str
    name: str
    outcome: str  # passed | failed | error | skipped
    message: str = ""
    file: str = ""
    line: int = 0


@dataclass
class SandboxRun:
    generated: list[CaseOutcome] = field(default_factory=list)
    suite: list[CaseOutcome] = field(default_factory=list)
    timed_out: bool = False
    logs: str = ""


def _tar_dir(repo_dir: Path, extra_files: dict[str, str]) -> bytes:
    """Tar the checkout (minus VCS/venv junk) plus generated test files, owned by the sandbox user."""
    buf = io.BytesIO()

    def owned(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = ti.gid = SANDBOX_UID
        ti.uname = ti.gname = "sandbox"
        return ti

    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in sorted(repo_dir.rglob("*")):
            rel = p.relative_to(repo_dir)
            if EXCLUDE_DIRS & set(rel.parts) or p.is_symlink():
                continue
            tar.add(p, arcname=f"repo/{rel.as_posix()}", recursive=False, filter=owned)
        for rel, content in extra_files.items():
            data = content.encode()
            ti = owned(tarfile.TarInfo(f"repo/{rel}"))
            ti.size, ti.mode = len(data), 0o644
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def parse_junit(xml_text: str) -> list[CaseOutcome]:
    out = []
    root = ET.fromstring(xml_text)
    for tc in root.iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = tc.find(tag)
            if el is not None:
                outcome = {"failure": "failed"}.get(tag, tag)
                message = (el.get("message") or "") + "\n" + (el.text or "")[:3000]
                break
        out.append(CaseOutcome(tc.get("classname", ""), tc.get("name", ""), outcome, message.strip(),
                               tc.get("file", ""), int(tc.get("line") or 0)))
    return out


class DockerSandbox:
    def __init__(self, settings: Settings, client: docker.DockerClient | None = None):
        self.s = settings
        self.docker = client or docker.from_env()

    # ------------------------------------------------------------ dependencies
    def deps_image(self, repo_dir: Path) -> str:
        if not self.s.sandbox_install_deps:
            return self.s.sandbox_image
        h = hashlib.sha256(self.s.sandbox_image.encode())
        for name in DEP_FILES:
            p = repo_dir / name
            if p.exists():
                h.update(name.encode() + p.read_bytes())
        tag = f"code-review-deps:{h.hexdigest()[:16]}"
        try:
            self.docker.images.get(tag)
            return tag  # cache hit: no install, no network
        except ImageNotFound:
            pass

        log.info("Building dependency image %s", tag)
        # Installs at the same path tests will run from, so editable installs resolve to the code
        # we copy in later. Afterwards the source is deleted; only site-packages stays in the image.
        script = (
            f"cd {REPO} && "
            "for f in requirements.txt requirements-dev.txt requirements/test.txt; do "
            "  [ -f $f ] && pip install --no-cache-dir -r $f; done; "
            "if [ -f pyproject.toml ] || [ -f setup.py ]; then pip install --no-cache-dir -e . ; fi; "
            f"find {REPO} -mindepth 1 -delete; exit 0"
        )
        container = self.docker.containers.create(
            self.s.sandbox_image, ["sh", "-c", script], user="root",
            mem_limit="2g", nano_cpus=int(2e9), pids_limit=512,
        )
        try:
            container.put_archive(WORK, _tar_dir(repo_dir, {}))
            container.start()
            result = container.wait(timeout=900)
            if result.get("StatusCode", 1) != 0:
                log.warning("Dependency install failed; falling back to base image")
                return self.s.sandbox_image
            container.commit(repository="code-review-deps", tag=tag.split(":")[1],
                             changes=[f"USER {SANDBOX_UID}", f"WORKDIR {REPO}"])
            return tag
        except (APIError, ReadTimeout, RequestsConnectionError) as e:
            log.warning("Dependency image build failed (%s); using base image", e)
            return self.s.sandbox_image
        finally:
            container.remove(force=True)

    # ------------------------------------------------------------------- tests
    def run(self, image: str, repo_dir: Path, generated_tests: dict[str, str], run_suite: bool) -> SandboxRun:
        """Run generated tests (and optionally the repo's own suite) offline in a fresh container."""
        files = {f"{GEN_DIR}/{name}": code for name, code in generated_tests.items()}
        pytest = ("python -m pytest -q -p no:cacheprovider -o junit_family=xunit1 "
                  "--timeout=30 --continue-on-collection-errors")
        cmds = []
        if generated_tests:
            cmds.append(f"{pytest} --junitxml={WORK}/out/generated.xml {GEN_DIR}")
        if run_suite:
            cmds.append(f"{pytest} --junitxml={WORK}/out/suite.xml --ignore={GEN_DIR}")
        if not cmds:
            return SandboxRun()
        script = f"mkdir -p {WORK}/out && cd {REPO} && " + " ; ".join(cmds) + " ; exit 0"

        container = self.docker.containers.create(
            image, ["sh", "-c", script],
            user=str(SANDBOX_UID),
            network_disabled=True,
            mem_limit=self.s.sandbox_mem_limit,
            memswap_limit=self.s.sandbox_mem_limit,  # no swap: memory limit is real
            nano_cpus=int(self.s.sandbox_cpus * 1e9),
            pids_limit=self.s.sandbox_pids_limit,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            environment={"PYTHONPATH": f"{REPO}:{REPO}/src", "PYTHONDONTWRITEBYTECODE": "1", "HOME": "/tmp"},
            working_dir=REPO,
        )
        run = SandboxRun()
        try:
            container.put_archive(WORK, _tar_dir(repo_dir, files))
            container.start()
            try:
                container.wait(timeout=self.s.sandbox_timeout_s)
            except (ReadTimeout, RequestsConnectionError):
                run.timed_out = True
                container.kill()
            run.logs = container.logs(tail=200).decode(errors="replace")
            run.generated = self._read_junit(container, "generated.xml")
            run.suite = self._read_junit(container, "suite.xml")
        finally:
            container.remove(force=True)
        return run

    @staticmethod
    def _read_junit(container, name: str) -> list[CaseOutcome]:
        try:
            stream, _ = container.get_archive(f"{WORK}/out/{name}")
        except NotFound:
            return []
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as tar:
            member = tar.getmembers()[0]
            data = tar.extractfile(member).read().decode(errors="replace")
        try:
            return parse_junit(data)
        except ET.ParseError:
            return []

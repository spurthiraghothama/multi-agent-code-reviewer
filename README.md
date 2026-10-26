# Multi-Agent Code Reviewer

A GitHub bot that reviews pull requests with a team of specialised LLM agents. Each agent sees the
whole repository, not just the diff. Before anything is posted, the system tries to **prove**
findings by running agent-written tests in an isolated Docker sandbox.

When a PR is opened, the system:

1. receives the webhook and verifies its signature,
2. checks out the PR and indexes the repo: an AST call graph plus ChromaDB embeddings, re-embedding only files that changed,
3. builds context: the changed functions, **who calls them**, and semantically related code,
4. runs **bug**, **performance** and **test** agents in parallel (LangGraph),
5. runs the generated tests in a locked-down container against the PR head **and** the base commit,
6. has a **lead agent** deduplicate, rank and summarise the findings, then posts one review with inline comments.

Only findings confirmed by a failing test, or high-severity and high-confidence ones, get posted.

```
GitHub ──webhook──► FastAPI ──► checkout (head + base worktrees)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            AST parse + call graph      ChromaDB (incremental embed)
                    └──────────────┬──────────────┘
                                   ▼
                     context: diff + changed fns + callers + related code
                                   │
                ┌──────────────────┼──────────────────┐      (LangGraph, parallel)
                ▼                  ▼                  ▼
            bug agent        performance agent    test agent
                └──────────────────┼──────────────────┘
                                   ▼
             Docker sandbox: generated tests + repo suite, on head AND base
             (no network, CPU/mem/pids limits, timeout, non-root, no caps)
                                   ▼
             lead agent: dedupe → rank → filter → summary
                                   ▼
             one GitHub review: inline comments + summary (+ cost/latency footer)
```

**Stack:** Python 3.11, FastAPI, LangGraph, ChromaDB, PyGithub and the Docker SDK. For the LLM you
can choose **Google Gemini**, which has a free tier, or **Claude** (Opus 5.5 by default), set with
`LLM_PROVIDER`.

---

## Contents

- [Quick start](#quick-start)
- [Testing locally with ngrok and a test repo](#testing-locally-with-ngrok-and-a-test-repo)
- [How each part works (and why)](#how-each-part-works-and-why)
- [Benchmark](#benchmark)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Limitations and next steps](#limitations-and-next-steps)

---

## Quick start

```bash
cp .env.example .env            # fill in GITHUB_TOKEN, GITHUB_WEBHOOK_SECRET and an LLM key (see below)
docker compose build            # builds the reviewer and the sandbox base image
docker compose up reviewer      # serves the webhook on http://localhost:8000/webhook
```

Without Docker Compose (for development):

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
docker build -t code-review-sandbox:latest sandbox    # the sandbox still needs Docker
uvicorn reviewer.main:app --reload --port 8000
```

**Getting an LLM key for free:** create a Gemini key at https://aistudio.google.com/apikey (no card
needed), then set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` in `.env`. Free-tier rate limits are
tight, so the client spaces out its calls and retries when it is rate limited, which makes reviews
slower. Run `python -m reviewer.cli models` to see which models your key can use. To use Claude
instead, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.

You can also review a PR directly, without a webhook:

```bash
python -m reviewer.cli review your-user/your-repo 1 --dry-run   # prints the review instead of posting
```

## Testing locally with ngrok and a test repo

GitHub has to reach your machine, and ngrok gives your local port a public URL.

1. **Create a test repo** on GitHub with some Python code and a `tests/` folder. Any small project works.
2. **Create a token:** GitHub → Settings → Developer settings → Fine-grained tokens, scoped to that
   repo, with *Contents: Read* and *Pull requests: Read and write*. Put it in `.env` as `GITHUB_TOKEN`.
3. **Create a webhook secret:** `python -c "import secrets; print(secrets.token_hex(32))"`. Put it
   in `.env` as `GITHUB_WEBHOOK_SECRET`.
4. **Start the server:** `docker compose up reviewer` (or `uvicorn ...` as above).
5. **Start ngrok:** `ngrok http 8000`. Copy the `https://….ngrok-free.app` URL.
6. **Add the webhook:** in the test repo, go to Settings → Webhooks → Add webhook:
   - Payload URL: `https://….ngrok-free.app/webhook`
   - Content type: `application/json`
   - Secret: the same value as `GITHUB_WEBHOOK_SECRET`
   - Events: *Let me select individual events* → **Pull requests** only.

   GitHub sends a `ping` straight away, and the server answers `pong`. Check the webhook's
   *Recent Deliveries* tab for a 202 response.
7. **Open a PR that introduces a bug.** For example, remove an `if not items: return []` guard, or
   change `range(len(x))` to `range(len(x) - 1)`. Within a few minutes a review appears on the PR.
   Server logs show each stage, including the cost.

Troubleshooting:

- **401 in Recent Deliveries:** the secret in `.env` and on GitHub don't match.
- **Review posted without inline comments:** GitHub rejected a comment line. The bot falls back to a summary-only review and logs the reason.
- **"Docker unavailable" in the logs:** the server can't reach `/var/run/docker.sock`. Findings are still reviewed, but nothing gets verified. On Linux, set `DOCKER_GID` in `.env` to the group id that owns the socket.
- **API 400 mentioning `fallbacks`:** set `LLM_REFUSAL_FALLBACK=false`.

---

## How each part works (and why)

### 1. GitHub integration ([src/reviewer/github/](src/reviewer/github/))

- **Signature verification** ([webhook.py](src/reviewer/github/webhook.py)). GitHub signs the raw
  body with HMAC-SHA256. We compute the HMAC over the exact bytes received, not over re-serialised
  JSON, and compare with `hmac.compare_digest`, which takes constant time and so doesn't leak the
  signature through timing. If no secret is configured we reject everything rather than run unauthenticated.
- **Respond fast, work later** ([main.py](src/reviewer/main.py)). GitHub times out after 10 seconds,
  but a review takes minutes. The endpoint validates, answers `202`, and runs the review as a
  background task. It ignores redelivered events (deduplicated by `X-GitHub-Delivery`) and allows
  at most 2 reviews at a time.
  *Trade-off:* in-process background tasks are lost if the server restarts mid-review. In production
  you'd put jobs on a durable queue (Redis/SQS plus workers). The pipeline code wouldn't change,
  because it takes a plain `ReviewRequest`.
- **Checkouts** ([client.py](src/reviewer/github/client.py)). The API gives you diffs but not a
  runnable repo, and we need one for AST parsing and running tests. We keep one partial clone per
  repo (`--filter=blob:none`) and create git worktrees for the head and base commits. The token is
  passed as a one-off HTTP header, so it is never written into `.git/config`.
- **Posting.** GitHub only accepts inline comments on lines that appear in the diff. The diff parser
  ([diff.py](src/reviewer/github/diff.py)) records which lines are commentable. Findings on other
  lines (for example a broken caller in another file) go into the summary. If GitHub still rejects
  the review, we retry with a summary-only review rather than post nothing.

### 2. Codebase context / RAG ([src/reviewer/indexing/](src/reviewer/indexing/))

- **Chunking by function and class** ([parser.py](src/reviewer/indexing/parser.py)). A function is
  the unit both a reviewer and a call graph reason about. Fixed-size windows cut functions in half.
  Each chunk keeps its file, qualified name and line range. Class chunks hold only the header and
  attributes, so methods aren't embedded twice.
- **Call graph** ([call_graph.py](src/reviewer/indexing/call_graph.py)). For each call site we try
  to resolve the target. *Exact* edges come from rules we can trust: a same-module function,
  `self.method()`, an import we can follow (including relative imports), or a class constructor.
  *Fuzzy* edges match only the bare name, and we keep them only when the name is rare (3 or fewer
  definitions). Python is dynamic, so a static graph can't be complete. We lean towards recall,
  because a missed caller can hide a bug, while a spurious caller just costs a few tokens.
- **Vector store** ([store.py](src/reviewer/indexing/store.py)). ChromaDB with its default local
  embedding model (all-MiniLM-L6-v2). It's free, needs no API key, and is good enough for "find
  related code". The call graph does the precise structural work. Metadata stored: file, qualified
  name, kind, start line, end line.
- **Incremental re-indexing.** A manifest records `{file: sha256}`. On each PR we hash every file,
  which is cheap, and re-embed only the files whose hash changed (the expensive part). A per-repo
  lock stops two PRs on the same repo from interleaving writes. Measured on this project:
  the first sync embedded 142 chunks and the second embedded 0.
- **Context assembly** ([context.py](src/reviewer/indexing/context.py)). In priority order: the
  diff, the full source of each changed function, its callers, then related code from semantic
  search. Everything carries real line numbers, so agents can cite `file:line`. A character budget
  is filled in that order, so on a huge PR related code is dropped first and the diff last.
  The section also lists import paths, so generated tests import the right modules.

### 3. Agents ([src/reviewer/agents/](src/reviewer/agents/))

- **LangGraph** ([graph.py](src/reviewer/agents/graph.py)). The three specialists fan out from `START`
  and join at `verify`. The `findings` and `usage` lists use an `operator.add` reducer, so the
  parallel branches append to them instead of overwriting each other. `asyncio.gather` would also
  work at this size. The graph pays off as the system grows, for example adding a security agent
  or a conditional "skip the sandbox if there are no tests" edge.
- **Specialists** ([specialists.py](src/reviewer/agents/specialists.py),
  [prompts.py](src/reviewer/agents/prompts.py)). One "find everything" prompt tends to fixate on
  one kind of issue, so each agent gets one lens. All three share the system prompt, context and
  output schema and differ only in their instructions. That keeps one output format and a shared,
  cacheable prompt prefix.
- **Recall upstream, precision downstream.** Specialists are told to report every plausible issue
  with an honest confidence score. Filtering happens later, with better evidence: the sandbox and
  the lead agent. A bug a self-censoring agent drops is lost for good. A noisy finding can still be
  filtered out.
- **Structured output.** Every finding is JSON with `file, line, severity, confidence, description,
  suggested_fix, evidence, test_code`. We use Claude's structured outputs (`output_config.format`
  with a JSON schema), so responses are constrained to the schema at decoding time. Pydantic then
  validates each finding on its own, so one bad item doesn't discard the whole batch. Findings that
  name files the agents never saw (hallucinated paths) are dropped.
- **Two LLM backends behind one interface.** Agents depend on a small `LLM` protocol with Claude and
  Gemini implementations, chosen by `LLM_PROVIDER`; agent code never knows which one it's using.
  The Gemini backend handles free-tier limits: it spaces calls out, waits and retries on HTTP 429,
  and stops immediately with a clear message when the daily quota is used up.
- **Why vendor SDKs and not a LangChain chat model?** LangGraph orchestrates the agents, but
  each one calls the model through the vendor's own SDK
  ([llm.py](src/reviewer/agents/llm.py)), for two reasons. First, LangChain's structured-output
  helper relies on *forced* tool calls, which current Claude models reject; native structured
  outputs avoid that. Second, we need the exact `usage` block, including cache read/write tokens,
  for cost tracking. Agents depend on a small `LLM` protocol, so tests use a scripted fake.
- **Lead agent** ([lead.py](src/reviewer/agents/lead.py)). The work is split on purpose:
  - **Deterministic code** handles near-duplicates (same file, within 3 lines, same category or
    similar text), ranking (confirmed, then severity, then confidence) and the posting policy.
    Whether a confirmed critical bug gets posted shouldn't depend on an LLM's mood.
  - **The LLM** finds *semantic* duplicates ("off-by-one in pagination" vs "last page skipped") and
    writes the summary. It only sees findings that will actually be posted, so the summary can't
    mention issues we dropped. Even when it groups duplicates, the code decides which one survives.
    If the lead call fails we still post, with a templated summary.
- **Cost tracking.** Every call records input, output, cache-write and cache-read tokens, priced per
  model ([cost.py](src/reviewer/agents/cost.py)). Failed calls are counted too. Each review's footer
  shows its total cost, tokens and latency.
- **Resilience.** If one agent fails, its error is recorded and the review continues without it.
  Refusals and `max_tokens` truncation are handled explicitly. Server-side refusal fallback is on by
  default: reviewing security-related code can occasionally trip a safety classifier.

### 4. Sandbox verification ([src/reviewer/sandbox/](src/reviewer/sandbox/))

- **Isolation** ([runner.py](src/reviewer/sandbox/runner.py)). The PR code and the LLM-written tests
  are both untrusted. Each run gets a fresh container with:
  - `network_disabled`, memory and CPU limits, swap disabled, and a pids limit (no fork bombs),
  - `cap_drop=ALL`, `no-new-privileges`, and a non-root user,
  - a wall-clock timeout plus a per-test timeout (`pytest-timeout`),
  - no secrets in the environment.

  Files go in through `put_archive` (a tar stream) and results come out through `get_archive`.
  There are **no bind mounts**, which also avoids the usual Docker-in-Docker "that path doesn't
  exist on the host" bug when the reviewer itself runs in a container.
- **Dependencies without network.** Tests need the repo's dependencies, but tests run offline. So a
  separate *networked* step pip-installs the repo once and snapshots the result as an image, cached
  by a hash of the dependency files. Later runs reuse it offline.
  *Trade-off:* that install step runs the repo's `setup.py` with network access. For untrusted forks,
  set `SANDBOX_INSTALL_DEPS=false` or pre-build dependency images.
- **What "confirmed" means** ([verify.py](src/reviewer/sandbox/verify.py)). A finding is confirmed
  only if all three hold:
  1. its generated test **fails on the PR head**,
  2. the failure isn't the test being broken. Import, syntax, name and collection errors and
     missing fixtures mean the LLM wrote a bad test, which says nothing about the code,
  3. in differential mode, the same test **does not also fail on the base commit**. If it fails on
     base too, the problem predates the PR. If it can't even import on base because the function
     is new, that's fine.

  A finding whose test *passes* is marked `not_reproduced` and is not posted.
- **Repo test suite.** The repo's own tests run on head and base. A test that fails only on head is
  a regression and becomes a confirmed finding.
- **The test agent's tests:** if one fails, it has found a bug and is posted. If it passes, it goes
  into the summary as a *suggested test, verified passing*, ready to copy into the suite.
- **Posting policy:** a finding is posted if it is confirmed, or if it is unverified with severity
  `high` or above and confidence at least `MIN_CONFIDENCE_UNVERIFIED` (0.8). Performance issues
  are rarely testable, so this second rule lets a strong one through.

### 5. Benchmark ([src/reviewer/benchmark/](src/reviewer/benchmark/))

See [Benchmark](#benchmark) below.

---

## Benchmark

The benchmark asks whether this architecture beats "one LLM call with the diff". It compares the
two on real bugs.

**1. Mine real bug-fix commits** ([mine.py](src/reviewer/benchmark/mine.py)). This runs on local
clones, so there are no API rate limits. A commit qualifies if:
- its message matches `fix|bug|regression|crash|incorrect|wrong`,
- it touches 1–2 source files with at most 30 changed lines, so the bug location is unambiguous,
- it **also changes a test file**, which is decent evidence that it really fixed a bug.

**2. Recreate the bug as a PR** ([prepare.py](src/reviewer/benchmark/prepare.py)):
- `base` = the fix commit, with the fix's **test** changes reverted. Otherwise the existing suite
  would catch every bug trivially.
- `bug` = `base` with the fix's **source** changes reverted, so the bug is back.
- The PR is `bug → base`, with a neutral commit message. The lines it changes are the ground truth.
- `--push-to you/bench-repo` pushes the branches and opens real PRs, so you can watch the webhook
  flow end to end.

**3. Run both systems** ([run.py](src/reviewer/benchmark/run.py)):
- The baseline ([baseline.py](src/reviewer/benchmark/baseline.py)) uses the same model, effort and
  schema, but makes one call with only the diff. This isolates the architecture from the model.
- The multi-agent system runs through the same `ReviewPipeline` the webhook uses, with posting
  turned off.

```bash
python -m reviewer.benchmark.mine     --config benchmark/config.yaml   # -> benchmark/cases.json
python -m reviewer.benchmark.prepare  --cases benchmark/cases.json     # -> benchmark/prepared.json
python -m reviewer.benchmark.run      --prepared benchmark/prepared.json --out results/
#   --limit 3          try a few cases first (each costs real API money)
#   --no-sandbox       skip Docker (multi-agent then relies on confidence only)
#   --systems baseline run only one system
#   --resume           continue a run that stopped (e.g. free-tier daily quota used up)
```

With Compose: `docker compose run --rm reviewer python -m reviewer.benchmark.run --out /data/results`.

**Outputs:** `results/summary.md` (table), `results/summary.csv`, `results/per_case.csv`, and
`results/details.json`, which holds every finding for spot checks.

**Metric definitions** ([metrics.py](src/reviewer/benchmark/metrics.py)):

| Metric | Definition |
|---|---|
| Catch rate | Cases where at least one posted finding lies in the ground-truth file within ±5 lines of a changed line, divided by all cases |
| False-positive rate | Posted findings that don't match the ground truth, divided by all posted findings (1 − precision) |
| Avg cost | Mean USD per review, from the API's token usage and [cost.py](src/reviewer/agents/cost.py) ($0 on Gemini's free tier) |
| Avg tokens | Mean tokens per review (input + output + cache); the efficiency number to compare when cost is $0 |
| Latency | Mean and median wall-clock seconds per review |

**Honest caveats:**
- The FP rate is an *upper bound*. A "false positive" may be a real, different issue, so spot-check
  `details.json`.
- A reverted fix often looks like "deleting a guard clause", which is easier to spot than a natural
  regression. Absolute numbers are optimistic, but both systems see the same diffs, so the comparison holds.
- 20 cases (the default: 4 repos × 5) gives wide error bars. Report counts as well as percentages.

Results table: run the benchmark and paste `results/summary.md` here.

| System | Cases | Catch rate | FP rate | Findings/review | Avg cost | Avg tokens | Avg latency | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_agent | – | – | – | – | – | – | – | – |
| baseline | – | – | – | – | – | – | – | – |

---

## Configuration

All settings are environment variables (see [.env.example](.env.example) and
[config.py](src/reviewer/config.py)).

| Variable | Default | Meaning |
|---|---|---|
| `GITHUB_TOKEN` | – | Token with contents:read and pull-requests:write |
| `GITHUB_WEBHOOK_SECRET` | – | Webhook HMAC secret (required; requests are rejected without it) |
| `LLM_PROVIDER` | `anthropic` | `gemini` (free tier) or `anthropic` (Claude) |
| `GEMINI_API_KEY` | – | Free key from aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model; use `gemini-flash-lite-latest` if you hit the daily quota |
| `GEMINI_MAX_TOKENS` | `32768` | Gemini output limit (its thinking tokens count toward it) |
| `GEMINI_MIN_INTERVAL_S` | `6` | Minimum seconds between Gemini calls (free-tier rate limit) |
| `ANTHROPIC_API_KEY` | – | Claude API key |
| `LLM_MODEL` | `claude-opus-5-5` | Claude model for all agents |
| `LLM_EFFORT` | `high` | `low`/`medium`/`high`/`xhigh`/`max`: reasoning depth vs cost |
| `LLM_REFUSAL_FALLBACK` | `true` | Server-side fallback if a request is declined |
| `RETRIEVAL_K` | `5` | Related chunks per changed function |
| `MAX_CALLERS_PER_FUNCTION` | `5` | Callers included per changed function |
| `MAX_CONTEXT_CHARS` | `60000` | Context budget per agent |
| `SANDBOX_TIMEOUT_S` | `180` | Wall-clock limit per sandbox run |
| `SANDBOX_MEM_LIMIT` / `SANDBOX_CPUS` | `1g` / `1.0` | Container limits |
| `SANDBOX_INSTALL_DEPS` | `true` | Networked dependency install step (cached image) |
| `SANDBOX_DIFFERENTIAL` | `true` | Also run tests on the base commit |
| `MIN_CONFIDENCE_UNVERIFIED` | `0.8` | Confidence needed to post an unverified high/critical finding |
| `DRY_RUN` | `false` | Run everything but log the review instead of posting |

## Project layout

```
src/reviewer/
  config.py            env-var settings (pydantic-settings)
  models.py            Finding, ReviewRequest, TokenUsage, ...: the shared contracts
  main.py              FastAPI webhook app
  pipeline.py          ReviewRequest -> ReviewResult (shared by webhook and benchmark)
  cli.py               review / index / graph commands
  github/              webhook.py (HMAC), diff.py (patch parsing), client.py (PyGithub + git)
  indexing/            parser.py (AST chunks), call_graph.py, store.py (ChromaDB), context.py
  agents/              llm.py, cost.py, prompts.py, specialists.py, lead.py, graph.py (LangGraph)
  sandbox/             runner.py (Docker), verify.py (verdicts)
  benchmark/           mine.py, prepare.py, baseline.py, run.py, metrics.py
sandbox/Dockerfile     sandbox base image (python + pytest + pytest-timeout, non-root user)
Dockerfile             reviewer service image
docker-compose.yml     reviewer + sandbox image build
benchmark/config.yaml  repos and cases per repo
tests/                 unit and end-to-end tests (no network, no Docker, no API key needed)
docs/INTERVIEW.md      interview questions and answers about this project
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite covers:
- signature verification and the webhook endpoint (including redelivery dedupe),
- diff parsing, including pure-deletion anchors,
- AST chunking, call-graph resolution (exact and fuzzy), and the context budget,
- deduplication, ranking and the posting policy,
- sandbox verdict logic (differential, broken-test detection, JUnit parsing),
- benchmark metrics, plus a mine→prepare round trip on a throwaway git repo,
- **an end-to-end pipeline run** through the real LangGraph graph, with a scripted LLM and a fake
  sandbox, including partial agent failure.

## Limitations and next steps

- **Python only.** The call graph is static: no type inference, so `obj.method()` is resolved by
  name. Possible upgrades: Jedi/Pyright for type-aware resolution, tree-sitter for other languages.
- **Webhook jobs live in memory.** Move them to a durable queue. Also cancel an in-flight review
  when a new push arrives on the same PR.
- **Sandbox hardening.** Docker socket access is root-equivalent on the host. In production, run
  sandboxes on a separate host or under gVisor/Firecracker.
- **Prompt caching under parallelism.** The three specialists share a prefix, but on a cold cache
  they all miss at once. Starting one agent slightly earlier would let the other two hit the cache.
- **Benchmark ground truth** is line proximity. An LLM judge ("does this finding describe the
  known bug?") would score more precisely, at its own cost.
- **Embeddings:** a code-specific embedding model would likely retrieve better than MiniLM.

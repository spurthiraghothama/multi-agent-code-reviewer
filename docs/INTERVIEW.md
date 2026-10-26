# Interview questions about this project

Each answer below is a version you could say out loud in about a minute, followed by where to
point in the code if the interviewer wants to dig in.

---

### 1. Walk me through what happens when a PR is opened.

GitHub sends a `pull_request` webhook. FastAPI checks the HMAC signature against the raw body,
returns 202 right away, and queues a background review, because GitHub times out after 10 seconds
and a review takes minutes. The worker checks out the PR head and base as git worktrees. It parses
every Python file with `ast` to build a call graph, and re-embeds only the files whose hash changed
into ChromaDB. For the changed functions it collects their full source, their callers and
semantically similar code. That context goes to three agents (bug, performance, test) running in
parallel in LangGraph. Their tests run in a locked-down Docker container against both head and
base. A lead agent then deduplicates, ranks and filters the findings, and we post one review with
inline comments and a summary that includes the cost.

*Code:* `main.py` → `pipeline.py` → `agents/graph.py`.

### 2. Why multiple agents instead of one big prompt?

Two reasons: attention and verification. A single "find all issues" prompt tends to fixate on
whatever stands out first. Separate agents each spend their whole reasoning budget on one lens.
The structure also lets me put a verification step between "generate" and "post", and that is
where the precision comes from. I don't just assert multi-agent is better. The benchmark compares
it with a one-call baseline that uses the same model, effort and output schema, so the difference
measures the architecture, not the model. The cost is real: roughly 4 or more LLM calls instead of
1, plus sandbox time. The report shows that cost next to the catch rate.

### 3. How do you decide a finding is "confirmed"? How do you avoid confirming garbage?

A finding is confirmed only if its generated test fails on the PR head, the failure isn't the test
itself being broken, and the same test doesn't also fail on the base commit. LLM-written tests
often fail for boring reasons: wrong import path, missing fixture, syntax error. Those prove
nothing about the code, so I classify them as `broken_test`. The base-commit check is a
differential test. If the test fails on base too, the bug predates the PR, so I don't blame this
PR. If the test can't even import on base because the function is new, that still counts. And a
test that *passes* is evidence against the finding, so it isn't posted however confident the model was.

*Code:* `sandbox/verify.py`. Tested in `tests/test_lead_and_verify.py` and with a real container run.

### 4. How did you secure the sandbox? Anything you're still worried about?

Both the PR and the generated tests are untrusted code. Each run gets a fresh container with:
- no network, so nothing can be exfiltrated,
- memory and CPU limits with swap disabled, and a pids limit against fork bombs,
- all Linux capabilities dropped, `no-new-privileges`, and a non-root user,
- a wall-clock timeout plus per-test timeouts,
- no secrets in the environment.

Files go in and out as tar streams instead of bind mounts, so the container never sees the host
filesystem. I'm still worried about two things:
- The reviewer reaches Docker through the socket, which is root-equivalent on the host. In
  production I'd run sandboxes on a separate host or under gVisor or Firecracker.
- The dependency-install step needs network and runs the repo's `setup.py`. It's cached and
  separate from the tests, but for untrusted forks I'd disable it or pre-build the images.

### 5. Why build a call graph if you already have embeddings?

They answer different questions. Embeddings find code that *looks similar*. The call graph finds
code that *depends on* the change. The bugs that diff-only reviewers miss are usually
interactions: you change a return type and a caller three files away still expects the old one.
That caller may look nothing like the changed function, so semantic search won't surface it, but
the call graph will. Embeddings supply conventions, sibling implementations and existing tests.

### 6. Python is dynamic. How accurate is your call graph?

It's a best-effort static approximation, and every edge is labelled with how it was resolved.
*Exact* edges follow rules I trust: same-module functions, `self.method()`, imports I can follow
(including relative imports), and constructors. *Fuzzy* edges match only a bare method name, and I
keep them only when the name is rare (3 or fewer definitions), because `get` or `run` would connect
everything to everything. I accept the asymmetry on purpose. A missed caller can hide a bug, while
a spurious caller only costs some tokens, and the prompt marks it as name-based. The next step
would be type-aware resolution with Jedi or Pyright.

### 7. How do you handle context size on a big PR?

I fill a character budget in priority order: the diff, then the full changed functions, then
callers, then semantically related code. On a huge PR, related code gets dropped first and the diff
last, and the context records that it was truncated. Chunks carry real line numbers so agents cite
exact lines. I also cap callers per function and results per query. For cost, all three
specialists share an identical prompt prefix and the large context carries a cache breakpoint.
One caveat I'd mention: because they run in parallel, a cold cache means all three miss at once,
so caching mainly pays off on re-reviews and the lead call.

### 8. How do you get reliable structured output from the LLM, and what happens when it fails?

I use Claude's native structured outputs, a JSON schema passed in `output_config.format`, so
decoding is constrained to valid JSON in the right shape. Pydantic then validates each finding on
its own, so one bad item doesn't discard the batch, and findings naming files the agent was never
shown get dropped as hallucinations. I call the Anthropic SDK directly rather than through a
LangChain chat model, for two reasons: LangChain's structured-output helper depends on forced tool
calls, which current Claude models reject, and I need exact token usage, including cache tokens,
for cost tracking. On failure, refusals and truncation raise a typed error that still carries the
token usage, so the cost is counted. If one specialist fails, the review continues without it. If
the lead fails, I post with a templated summary.

### 9. How did you evaluate it, and what are the weaknesses of your benchmark?

I mine real bug-fix commits from open-source repos. A commit qualifies if it is a small fix that
also changes a test file, which is good evidence it's a real bug. I rebuild each one as a PR that
reintroduces the bug, with the fix's test removed from the base so the suite can't catch it
trivially. The lines that PR changes are the ground truth. I report:
- catch rate: a posted finding within ±5 lines of the bug,
- false-positive rate: unmatched findings divided by all posted findings,
- average cost and latency,

each for both my system and the single-call baseline. The weaknesses I'd name:
- the FP rate is an upper bound, because an "FP" might be a different real bug,
- a reverted fix often looks like a deleted guard clause, which is easier to spot than a natural regression,
- line proximity is a crude matcher,
- about 20 cases means wide error bars.

The comparison is still fair, because both systems see identical diffs.

### 10. What would you change to run this in production for a large org?

- **Durability:** move from in-process background tasks to a queue with workers, and cancel stale
  reviews when a new push arrives.
- **Isolation:** run sandboxes on dedicated hosts or microVMs, not through a shared Docker socket.
- **Scale:** a GitHub App instead of a personal token, per-installation rate limiting, and a shared
  vector store with per-repo collections.
- **Quality loop:** log which comments developers resolve or dismiss, and use that as a live
  precision signal to tune the posting threshold per repo.
- **Cost:** route the cheap, high-volume steps (such as a first triage pass) to a smaller model,
  and stagger the specialists so they share the prompt cache.
- **Coverage:** type-aware call graphs, more languages via tree-sitter, and an LLM-judge matcher in
  the benchmark.

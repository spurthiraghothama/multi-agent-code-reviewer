"""Central configuration, loaded from environment variables (and an optional .env file).

Every tunable lives here so that nothing else in the codebase reads os.environ directly.
That makes the system easy to reason about ("what can I change?") and easy to test
(construct a Settings object with overrides instead of monkeypatching the environment).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- GitHub -------------------------------------------------------------
    github_token: str = Field(default="", description="PAT or installation token with repo + pull_request scope")
    github_webhook_secret: str = Field(default="", description="Shared secret configured on the GitHub webhook")

    # --- LLM ----------------------------------------------------------------
    # "anthropic" (Claude, paid) or "gemini" (Google, has a free tier).
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5-5"
    # Effort controls how much the model thinks. Code review is reasoning-heavy, so "high"
    # is a sensible default; drop to "medium" or "low" to trade quality for cost.
    llm_effort: str = "high"
    llm_max_tokens: int = 16000
    # Server-side refusal fallback: code review of security-sensitive code can occasionally trip
    # a safety classifier; with this on, the API transparently re-runs on a fallback model.
    llm_refusal_fallback: bool = True
    # Gemini: the "-latest" alias always points at Google's current Flash model.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    # On Gemini, thinking tokens count toward the output limit, so it needs more room than Claude's
    # max_tokens; Flash models allow up to 65,536.
    gemini_max_tokens: int = 32768
    # Minimum seconds between Gemini calls; 6s keeps us under ~10 requests/minute on the free tier.
    gemini_min_interval_s: float = 6.0

    # --- Indexing / RAG ------------------------------------------------------
    chroma_path: Path = Path("./data/chroma")
    workspace_dir: Path = Path("./data/workspaces")
    retrieval_k: int = 5  # semantically similar chunks per changed function
    max_callers_per_function: int = 5
    max_context_chars: int = 60_000  # hard cap on the context we send to each agent

    # --- Sandbox -------------------------------------------------------------
    sandbox_image: str = "code-review-sandbox:latest"
    sandbox_timeout_s: int = 180
    sandbox_mem_limit: str = "1g"
    sandbox_cpus: float = 1.0
    sandbox_pids_limit: int = 256
    # Install the repo's dependencies in a *separate* networked step, then snapshot the image.
    # Tests themselves always run with networking disabled.
    sandbox_install_deps: bool = True
    # Also run generated tests against the base commit: "fails on PR, passes on base" is much
    # stronger evidence than "fails on PR" alone. Doubles sandbox time.
    sandbox_differential: bool = True

    # --- Posting policy ------------------------------------------------------
    # A finding is posted if it was confirmed by a failing test, OR if it is high/critical severity
    # with confidence >= this threshold.
    min_confidence_unverified: float = 0.8
    dry_run: bool = False  # run the full pipeline but don't post to GitHub


@lru_cache
def get_settings() -> Settings:
    return Settings()

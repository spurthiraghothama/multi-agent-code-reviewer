from __future__ import annotations

import os

# Tests must never depend on a developer's real .env or credentials.
os.environ.setdefault("GITHUB_WEBHOOK_SECRET", "test-secret")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

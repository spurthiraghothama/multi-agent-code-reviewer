"""GitHub webhook signature verification.

GitHub signs each delivery with HMAC-SHA256 over the *raw request body* using the webhook secret,
and sends it as `X-Hub-Signature-256: sha256=<hex>`. We must:
  1. compute the HMAC over the exact bytes received (not re-serialised JSON - key order and
     whitespace would change the digest), and
  2. compare with hmac.compare_digest, which runs in constant time so an attacker can't
     recover the signature byte-by-byte from response timing.
"""

from __future__ import annotations

import hashlib
import hmac

# Only these actions change the code under review.
REVIEWABLE_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}


def compute_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    if not secret:
        # Refuse to run unauthenticated rather than silently accepting everything.
        return False
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    return hmac.compare_digest(compute_signature(secret, body), signature_header)


def should_review(event: str, payload: dict) -> bool:
    if event != "pull_request":
        return False
    if payload.get("action") not in REVIEWABLE_ACTIONS:
        return False
    pr = payload.get("pull_request") or {}
    return not pr.get("draft", False)

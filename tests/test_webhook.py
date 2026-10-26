import json

import pytest
from fastapi.testclient import TestClient

from reviewer import main
from reviewer.github.webhook import compute_signature, should_review, verify_signature

SECRET = "test-secret"


def test_valid_signature_accepted():
    body = b'{"a": 1}'
    assert verify_signature(SECRET, body, compute_signature(SECRET, body))


@pytest.mark.parametrize("header", [None, "", "sha1=abc", "sha256=" + "0" * 64])
def test_bad_signatures_rejected(header):
    assert not verify_signature(SECRET, b"{}", header)


def test_tampered_body_rejected():
    sig = compute_signature(SECRET, b'{"action": "opened"}')
    assert not verify_signature(SECRET, b'{"action": "closed"}', sig)


def test_empty_secret_rejects_everything():
    assert not verify_signature("", b"{}", compute_signature("", b"{}"))


def test_should_review_filters_actions_and_drafts():
    assert should_review("pull_request", {"action": "opened", "pull_request": {}})
    assert should_review("pull_request", {"action": "synchronize", "pull_request": {"draft": False}})
    assert not should_review("pull_request", {"action": "closed", "pull_request": {}})
    assert not should_review("pull_request", {"action": "opened", "pull_request": {"draft": True}})
    assert not should_review("push", {"action": "opened"})


def _post(client, payload, delivery="d1", secret=SECRET, event="pull_request"):
    body = json.dumps(payload).encode()
    return client.post("/webhook", content=body, headers={
        "X-GitHub-Event": event, "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": compute_signature(secret, body), "Content-Type": "application/json"})


def test_webhook_endpoint(monkeypatch):
    queued = []

    async def fake_review(repo, number):
        queued.append((repo, number))

    monkeypatch.setattr(main, "run_review", fake_review)
    main._seen_deliveries.clear()
    client = TestClient(main.app)
    payload = {"action": "opened", "repository": {"full_name": "o/r"}, "pull_request": {"number": 7}}

    assert _post(client, payload, secret="wrong").status_code == 401
    r = _post(client, payload, delivery="abc")
    assert r.status_code == 202 and r.json()["status"] == "queued"
    assert queued == [("o/r", 7)]
    # GitHub redelivery of the same delivery id must not trigger a second review
    assert _post(client, payload, delivery="abc").json()["status"] == "duplicate delivery"
    assert _post(client, {**payload, "action": "closed"}, delivery="xyz").json()["status"] == "ignored"
    assert len(queued) == 1

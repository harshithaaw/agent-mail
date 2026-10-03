import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from email import message_from_bytes
from email.policy import default

import app
import gmail.draft
import agents.security.agent as security_agent


ROOT = Path(__file__).resolve().parents[1]


def test_real_security_fixture_reaches_gmail_draft_without_sender_name_error(monkeypatch, tmp_path, capsys):
    email = json.loads((ROOT / "data/fixtures/inbox_10.json").read_text())[0]
    email = {
        **email,
        "message_id": "security-draft-regression",
        "thread_id": "fixture-thread",
        "received_at": "2026-09-26T15:13:40+00:00",
        "message_id_rfc": "<fixture@example.org>",
    }
    created = []

    # Keep the real security graph, phishing check, and injection heuristics;
    # only replace the spam model boundary so this test works without pickles.
    monkeypatch.setattr(security_agent, "run_spam", lambda text: {"is_spam": False, "label": "ham"})

    class Request:
        def execute(self):
            return {"id": "fake-draft-id", "message": {"threadId": "fixture-thread"}}

    class Drafts:
        def create(self, **kwargs):
            created.append(kwargs)
            return Request()

    class Users:
        def drafts(self):
            return Drafts()

    class Service:
        def users(self):
            return Users()

    monkeypatch.setattr(gmail.draft, "get_gmail_service", lambda: Service())
    monkeypatch.setattr(app, "reply_to_email", lambda *args, **kwargs: {
        "draft": "Fixture generated response", "retrieved": [],
    })

    result = app.process_inbox(
        fetcher=lambda max_emails: [email],
        security=security_agent.run_security_agent,
        understanding=lambda text: {"category": "Personal"},
        gate=lambda route, sender, *headers: True,
        collection=object(),
        embed=lambda text: [0.0],
        processed_db=tmp_path / "processed.db",
        now=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )[0]

    assert result["route"] == "clean_full_pipeline"
    assert result["status"] == "draft_created"
    assert result["draft"] == "fake-draft-id"
    assert created
    raw = created[0]["body"]["message"]["raw"]
    draft_message = message_from_bytes(base64.urlsafe_b64decode(raw), policy=default)
    assert draft_message["To"] == email["sender_email"]
    assert "[security-draft-regression] route=clean_full_pipeline status=draft_created" in capsys.readouterr().out

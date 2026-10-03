import sqlite3
from datetime import datetime, timezone

import app


def _email(message_id, **overrides):
    return {
        "message_id": message_id,
        "thread_id": f"thread-{message_id}",
        "sender_email": "person@example.org",
        "subject": "Fixture",
        "body": "Fixture body",
        "timestamp": "2026-01-01T00:00:00+00:00",
        **overrides,
    }


def _base(tmp_path, email, **kwargs):
    values = dict(
        fetcher=lambda max_emails: [email],
        security=lambda *args: {"route": "clean_full_pipeline"},
        understanding=lambda text: {"category": "Personal"},
        drafter=lambda current, understood: "draft-id",
        gate=lambda route, sender, unsubscribe="", precedence="": not (
            "no-reply" in sender or unsubscribe or precedence in {"bulk", "list", "junk"}
        ),
        now=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    values.update(kwargs)
    values.setdefault("processed_db", tmp_path / "processed.db")
    return values


def test_automated_gate_skip_has_own_status(tmp_path):
    email = _email("automated", sender_email="no-reply@example.org")
    result = app.process_inbox(**_base(tmp_path, email))[0]
    assert result["status"] == "skipped_automated_sender"
    with sqlite3.connect(tmp_path / "processed.db") as db:
        assert db.execute("SELECT status FROM processed WHERE message_id='automated'").fetchone()[0] == "skipped_automated_sender"


def test_generation_failure_retries_three_times_then_stops(tmp_path):
    email = _email("retry")
    dependencies = _base(tmp_path, email, drafter=lambda *_: (_ for _ in ()).throw(RuntimeError("local model down")))
    for expected_attempts in (1, 2, 3):
        result = app.process_inbox(**dependencies)
        assert result[0]["status"] == "generation_failed"
        with sqlite3.connect(tmp_path / "processed.db") as db:
            assert db.execute("SELECT attempts FROM processed WHERE message_id='retry'").fetchone()[0] == expected_attempts
    assert app.process_inbox(**dependencies) == []


def test_go_live_cutoff_settings_and_environment_override(tmp_path, monkeypatch):
    db_path = tmp_path / "processed.db"
    old_email = _email("old", timestamp="2025-12-31T23:59:59+00:00")
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert app.process_inbox(**_base(tmp_path, old_email, processed_db=db_path, now=cutoff)) == []
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT value FROM settings WHERE key='go_live_at'").fetchone()[0] == cutoff.isoformat()
        assert db.execute("SELECT 1 FROM processed WHERE message_id='old'").fetchone() is None

    monkeypatch.setenv("AGENTMAIL_GO_LIVE_AT", "2025-12-31T00:00:00Z")
    result = app.process_inbox(**_base(tmp_path, old_email, processed_db=db_path))
    assert result[0]["status"] == "draft_created"


def test_go_live_cutoff_compares_offset_timestamps_as_aware_datetimes(tmp_path):
    cutoff = datetime.fromisoformat("2026-09-26T15:13:38+00:00")
    before = _email("offset-before", timestamp="2026-09-26T20:43:37+05:30")
    after = _email("offset-after", timestamp="2026-09-26T20:43:39+05:30")
    requested = []
    dependencies = _base(tmp_path, before, now=cutoff)
    dependencies["fetcher"] = lambda max_emails: [before, after]
    dependencies["drafter"] = lambda email, understood: requested.append(email["message_id"]) or "draft-id"

    results = app.process_inbox(**dependencies)

    assert [item["message_id"] for item in results] == ["offset-after"]
    assert requested == ["offset-after"]


def test_missing_and_unparseable_timestamps_are_skipped_and_logged(tmp_path, capsys):
    missing = _email("missing-time")
    missing.pop("timestamp")
    invalid = _email("invalid-time", timestamp="not-a-timestamp")
    drafted = []
    dependencies = _base(tmp_path, missing)
    dependencies["fetcher"] = lambda max_emails: [missing, invalid]
    dependencies["drafter"] = lambda email, understood: drafted.append(email["message_id"]) or "draft-id"

    assert app.process_inbox(**dependencies) == []
    assert drafted == []
    output = capsys.readouterr().out
    assert "[missing-time] skipped_missing_or_invalid_timestamp: timestamp is missing" in output
    assert "[invalid-time] skipped_missing_or_invalid_timestamp" in output


def test_received_at_takes_precedence_over_naive_timestamp(tmp_path):
    email = _email(
        "received-at-preferred",
        timestamp="2026-09-26T20:43:38",
        received_at="2026-09-26T15:13:39+00:00",
    )
    result = app.process_inbox(**_base(tmp_path, email))[0]
    assert result["status"] == "draft_created"


def test_naive_timestamp_without_received_at_is_skipped_and_logged(tmp_path, capsys):
    email = _email("naive-without-received-at", timestamp="2026-09-26T20:43:38")
    email.pop("received_at", None)
    drafted = []
    dependencies = _base(tmp_path, email, drafter=lambda current, _: drafted.append(current) or "draft-id")
    assert app.process_inbox(**dependencies) == []
    assert drafted == []
    assert "[naive-without-received-at] skipped_missing_or_invalid_timestamp" in capsys.readouterr().out


def test_received_at_before_cutoff_is_skipped(tmp_path):
    cutoff = datetime.fromisoformat("2026-09-26T15:13:38+00:00")
    email = _email(
        "received-at-before-cutoff",
        timestamp="2026-09-26T20:43:38",
        received_at="2026-09-26T15:13:37+00:00",
    )
    drafted = []
    dependencies = _base(tmp_path, email, now=cutoff, drafter=lambda current, _: drafted.append(current) or "draft-id")
    assert app.process_inbox(**dependencies) == []
    assert drafted == []


def test_fetch_full_email_adds_utc_received_at_without_changing_timestamp():
    from gmail.fetch import fetch_full_email

    class Request:
        def execute(self):
            return {
                "id": "gmail-id",
                "threadId": "thread-id",
                "internalDate": "1790435620000",
                "payload": {"headers": [
                    {"name": "Date", "value": "Sat, 26 Sep 2026 20:43:38"},
                    {"name": "From", "value": "Person <person@example.org>"},
                ]},
            }

    class Messages:
        def get(self, **kwargs):
            return Request()

    class Users:
        def messages(self):
            return Messages()

    class Service:
        def users(self):
            return Users()

    result = fetch_full_email(Service(), "gmail-id")
    assert result["received_at"] == "2026-09-26T15:13:40+00:00"
    assert result["timestamp"] == "2026-09-26T20:43:38"


def test_fetch_full_email_uses_empty_received_at_when_internal_date_is_missing():
    from unittest.mock import Mock
    from gmail.fetch import fetch_full_email

    service = Mock()
    service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "gmail-id",
        "payload": {"headers": [{"name": "Date", "value": "Sat, 26 Sep 2026 20:43:38"}]},
    }
    result = fetch_full_email(service, "gmail-id")
    assert result["received_at"] == ""


def test_inbox_fetch_limit_defaults_to_25(tmp_path):
    requested = []
    email = _email("limit")
    deps = _base(tmp_path, email)
    deps["fetcher"] = lambda max_emails: requested.append(max_emails) or []
    app.process_inbox(**deps)
    assert requested == [25]

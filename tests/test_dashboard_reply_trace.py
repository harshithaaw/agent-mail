import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import (
    _processed_display_rows,
    _reply_example_display_rows,
    load_reply_trace,
)


def _record(message_id, time, switch, examples):
    return {
        "time": time,
        "message_id": message_id,
        "switch": switch,
        "examples": examples,
    }


def _example(rank, *, above=False, sent=False):
    return {
        "rank": rank,
        "subject": f"Past reply {rank}",
        "distance": 0.59 if above else 0.61,
        "sent_to_sender": sent,
        "above_threshold": above,
    }


def test_reply_trace_groups_by_email_and_labels_threshold_by_switch(tmp_path):
    path = tmp_path / "reply_trace.jsonl"
    path.write_text(
        "{bad json}\n"
        + "\n".join(json.dumps(row) for row in [
            _record("email-old", "2026-10-07T10:00:00+00:00", "on", [
                _example(1, above=True, sent=True), _example(2),
            ]),
            _record("email-new", "2026-10-07T11:00:00+00:00", "off", [
                _example(1, above=True),
            ]),
        ])
        + "\n",
        encoding="utf-8",
    )

    emails = load_reply_trace(path)

    assert [email["email id"] for email in emails] == ["email-new", "email-old"]
    assert emails[0]["examples used"] == 1
    assert emails[0]["threshold label"] == "weak, still used"
    assert emails[0]["threshold count"] == 1
    assert emails[0]["examples"][0]["rank"] == 1
    assert emails[1]["examples used"] == 2
    assert emails[1]["threshold label"] == "dropped from the prompt"
    assert emails[1]["sent count"] == 1


def test_reply_trace_limits_number_of_emails(tmp_path):
    path = tmp_path / "reply_trace.jsonl"
    records = [
        _record(f"email-{index}", f"2026-10-07T{index:02}:00:00+00:00", "off", [])
        for index in range(12)
    ]
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    emails = load_reply_trace(path, limit=10)

    assert len(emails) == 10
    assert emails[0]["email id"] == "email-11"
    assert emails[-1]["email id"] == "email-2"


def test_reply_trace_handles_missing_file(tmp_path):
    assert load_reply_trace(tmp_path / "missing.jsonl") == []


def test_reply_trace_display_rows_use_short_labels_without_changing_trace_data():
    example = {
        "rank": 1,
        "past-reply subject": "Past reply 1",
        "distance": 0.59123,
        "sent to this sender": "yes",
    }
    displayed = _reply_example_display_rows([example])

    assert displayed == [{
        "Rank": 1,
        "Past reply": "Past reply 1",
        "Distance": 0.59123,
        "Sent to this sender": "✓",
    }]
    assert example["distance"] == 0.59123
    assert _reply_example_display_rows([{**example, "sent to this sender": "no"}])[0][
        "Sent to this sender"
    ] == "—"


def test_processed_display_rows_shorten_headers_and_status():
    row = {
        "time": "10:00 AM",
        "sender": "sender@example.com",
        "subject": "Hello",
        "route": "draft",
        "status": "draft_created",
    }

    assert _processed_display_rows([row]) == [{
        "Time": "10:00 AM",
        "From": "sender@example.com",
        "Subject": "Hello",
        "Route": "draft",
        "Status": "✅ Draft",
    }]
    assert row["status"] == "draft_created"

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import load_reply_trace


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

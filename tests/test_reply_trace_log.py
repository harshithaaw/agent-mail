import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.reply.trace_log import write_reply_trace


def test_trace_omits_incoming_text_and_addresses(tmp_path):
    path = tmp_path / "logs" / "reply_trace.jsonl"
    sender = "private.sender@example.test"
    incoming_subject = "incoming subject must not be recorded"
    incoming_body = "incoming body must not be recorded"
    write_reply_trace(
        {
            "message_id": "message-123",
            "sender_email": sender,
            "subject": incoming_subject,
            "body": incoming_body,
        },
        {
            "stopped_reason": "grounded_with_context",
            "validation": {
                "sufficient": True,
                "total_retrieved": 1,
                "usable_examples": [{"metadata": {"recipient_norm": sender}}],
            },
            "retrieved_examples": [{
                "id": "example-1",
                "document": "Past subject\nPast body",
                "distance": 0.123456,
                "metadata": {"recipient_norm": sender},
            }],
        },
        path=path,
    )

    record_text = path.read_text(encoding="utf-8")
    record = json.loads(record_text)
    assert incoming_subject not in record_text
    assert incoming_body not in record_text
    assert sender not in record_text
    assert record["validation"]["sufficient"] is True
    assert record["examples"][0]["distance"] == 0.1235
    assert record["examples"][0]["sent_to_sender"] is True


def test_trace_write_failure_does_not_raise(tmp_path):
    destination_is_directory = tmp_path / "trace.jsonl"
    destination_is_directory.mkdir()
    write_reply_trace({"message_id": "message-123"}, {}, path=destination_is_directory)


def test_above_threshold_flags_match_validate_context_drop_condition(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_reply_trace(
        {"message_id": "message-456"},
        {"retrieved_examples": [
            {"id": "below", "document": "below", "distance": 0.4237, "metadata": {}},
            {"id": "equal", "document": "equal", "distance": 0.60, "metadata": {}},
            {"id": "above", "document": "above", "distance": 0.631, "metadata": {}},
        ]},
        path=path,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert [example["above_threshold"] for example in record["examples"]] == [False, True, True]

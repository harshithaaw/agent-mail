import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import load_reply_trace


def test_reply_trace_reader_handles_missing_and_bad_lines(tmp_path):
    path = tmp_path / "reply_trace.jsonl"
    assert load_reply_trace(path) == []

    path.write_text(
        "{bad json}\n"
        + json.dumps({
            "time": "2026-10-07T10:00:00+00:00",
            "message_id": "message-1",
            "switch": "on",
            "examples": [{
                "rank": 1,
                "subject": "Past reply",
                "distance": 0.42,
                "sent_to_sender": True,
                "above_threshold": True,
            }],
        })
        + "\n",
        encoding="utf-8",
    )
    assert load_reply_trace(path) == [{
        "time": "2026-10-07T10:00:00+00:00",
        "email id": "message-1",
        "switch": "on",
        "rank": 1,
        "past-reply subject": "Past reply",
        "distance": 0.42,
        "sent to this sender": "yes",
        "above 0.60": "yes",
    }]

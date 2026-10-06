import json
import logging
from types import SimpleNamespace

from agents.understanding import agent, runner
from agents.understanding.prioritizer import detect_deadline, detect_deadline_legacy


def test_runner_received_at_reaches_real_agent_tools_without_tool_arguments(monkeypatch, tmp_path):
    received_at = "2026-10-03T10:00:00+05:30"
    text = "Reply with the venue choice by tomorrow, or we will keep the current booking."
    monkeypatch.setattr(agent, "DEADLINE_SHADOW_LOG_PATH", tmp_path / "deadline_shadow.jsonl")

    def invoke(state):
        assert set(state) == {"messages"}
        deadline = json.loads(agent.detect_deadline.invoke({"email_text": text}))
        priority = json.loads(agent.assess_priority.invoke({"email_text": text}))
        return {"result": {"deadline": deadline, "priority": priority}}

    monkeypatch.setattr(runner, "understanding_agent", SimpleNamespace(invoke=invoke))
    result = runner.understand_email(text, received_at=received_at, message_id="message-17")

    assert result["deadline"]["has_deadline"] is True
    assert result["deadline"]["deadline_date"] == "2026-10-04"
    assert result["priority"]["deadline"]["deadline_date"] == "2026-10-04"


def test_missing_received_at_returns_old_shape_false_and_logs_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(agent, "DEADLINE_SHADOW_LOG_PATH", tmp_path / "deadline_shadow.jsonl")
    with agent.deadline_context(None):
        with caplog.at_level(logging.WARNING):
            result = json.loads(agent.detect_deadline.invoke({"email_text": "Reply by tomorrow."}))

    assert result == {"has_deadline": False, "deadline_date": None, "raw_phrase": None}
    assert "received_at" in caplog.text


def test_v2_result_has_exact_legacy_key_set():
    result = detect_deadline("Reply by tomorrow.", "2026-10-03T10:00:00+05:30")
    assert set(result) == set(detect_deadline_legacy("No deadline here."))
    assert set(result) == {"has_deadline", "deadline_date", "raw_phrase"}


def test_legacy_environment_switch_returns_legacy_result(monkeypatch, tmp_path):
    text = "Please reply by October 9, 2032."
    monkeypatch.setenv("AGENTMAIL_DEADLINE_DETECTOR", "legacy")
    monkeypatch.setattr(agent, "DEADLINE_SHADOW_LOG_PATH", tmp_path / "deadline_shadow.jsonl")
    with agent.deadline_context("2026-10-03T10:00:00+05:30"):
        actual = json.loads(agent.detect_deadline.invoke({"email_text": text}))
    assert actual == detect_deadline_legacy(text)


def test_shadow_log_is_minimal_and_unwritable_path_does_not_break_detection(monkeypatch, tmp_path):
    received_at = "2026-10-03T10:00:00+05:30"
    text = "Please reply by October 9. PRIVATE_BODY_MARKER is in another sentence."
    log_path = tmp_path / "deadline_shadow.jsonl"
    monkeypatch.setattr(agent, "DEADLINE_SHADOW_LOG_PATH", log_path)
    with agent.deadline_context(received_at, "message-29"):
        result = json.loads(agent.detect_deadline.invoke({"email_text": text}))

    line = log_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert result["has_deadline"] is True
    assert record["message_id"] == "message-29"
    assert record["received_at"] == received_at
    assert set(record["old_result"]) == {"deadline_date", "raw_phrase"}
    assert set(record["new_result"]) == {"deadline_date", "raw_phrase"}
    assert "PRIVATE_BODY_MARKER" not in line
    assert text not in line

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    monkeypatch.setattr(agent, "DEADLINE_SHADOW_LOG_PATH", blocker / "deadline_shadow.jsonl")
    with agent.deadline_context(received_at):
        result = json.loads(agent.detect_deadline.invoke({"email_text": "Reply by October 11."}))
    assert result["has_deadline"] is True

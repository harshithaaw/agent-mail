import builtins
import json
import sys
import types
from pathlib import Path

import pytest

from pipelines import reply as reply_pipeline


EMAIL = {
    "subject": "Question",
    "body": "Could we meet?",
    "sender_name": "A Sender",
    "sender_email": "sender@example.test",
    "thread_id": "thread-1",
}
UNDERSTANDING = {"category": "Career", "summary": "A question"}


def _thin(monkeypatch):
    rows = [{"id": "thin-1", "document": "Past", "metadata": {}, "distance": 0.2}]
    monkeypatch.setattr(reply_pipeline, "retrieve_examples", lambda *args: rows)
    monkeypatch.setattr(reply_pipeline, "generate_grounded_draft", lambda *args: "Thin draft")
    return rows


def _fake_agent_module(fn):
    return types.SimpleNamespace(run_reply_agent=fn)


def test_switch_off_uses_thin_path_without_importing_agent(monkeypatch):
    monkeypatch.delenv("AGENTMAIL_REPLY_AGENT", raising=False)
    rows = _thin(monkeypatch)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "agents.reply.agent":
            raise AssertionError("agent imported while switch is off")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert reply_pipeline.reply_to_email(EMAIL, UNDERSTANDING, object(), lambda _: [], generate=lambda *_: "Thin draft") == {
        "draft": "Thin draft", "retrieved": rows, "category": "Career"
    }


def test_switch_on_passes_injected_store_and_returns_live_shape(monkeypatch):
    monkeypatch.setenv("AGENTMAIL_REPLY_AGENT", "1")
    collection, embed = object(), lambda text: [1.0]
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"draft": "Agent draft", "retrieved_examples": [{"id": "agent-1"}], "validation": {"sufficient": True}}

    monkeypatch.setitem(sys.modules, "agents.reply.agent", _fake_agent_module(run))
    result = reply_pipeline.reply_to_email(EMAIL, UNDERSTANDING, collection, embed)
    args, kwargs = calls[0]
    assert "Sender: A Sender <sender@example.test>" in args[0]
    assert args[1]["sender_name"] == "A Sender"
    assert kwargs == {"sender_email": "sender@example.test", "thread_id": "thread-1", "collection": collection, "embed": embed}
    assert result == {"draft": "Agent draft", "retrieved": [{"id": "agent-1"}], "category": "Career", "validation": {"sufficient": True}}


@pytest.mark.parametrize("agent_result", [RuntimeError("broken"), {"draft": "", "retrieved_examples": []}])
def test_agent_failure_or_empty_draft_falls_back(monkeypatch, agent_result):
    monkeypatch.setenv("AGENTMAIL_REPLY_AGENT", "1")
    rows = _thin(monkeypatch)

    def run(*args, **kwargs):
        if isinstance(agent_result, Exception):
            raise agent_result
        return agent_result

    monkeypatch.setitem(sys.modules, "agents.reply.agent", _fake_agent_module(run))
    result = reply_pipeline.reply_to_email(EMAIL, UNDERSTANDING, object(), lambda _: [], generate=lambda *_: "Thin draft")
    assert result == {"draft": "Thin draft", "retrieved": rows, "category": "Career"}


def test_run_reply_agent_old_call_without_injected_store_still_works(monkeypatch):
    from agents.reply import agent

    seen = {}

    def invoke(state):
        seen.update(state)
        return {**state, "draft": "Draft", "stopped_reason": "no_retrieval_needed"}

    monkeypatch.setattr(agent.reply_graph, "invoke", invoke)
    result = agent.run_reply_agent("Email", UNDERSTANDING)
    assert seen["collection"] is None and seen["embed"] is None
    assert result["draft"] == "Draft"
    assert set(result) == {"draft", "stopped_reason", "trace", "examples_used", "retrieved_examples", "validation"}


def test_agent_result_can_be_written_to_trace(tmp_path):
    from agents.reply.trace_log import write_reply_trace

    path = tmp_path / "trace.jsonl"
    write_reply_trace(EMAIL, {
        "draft": "Agent draft",
        "retrieved": [{"id": "sent-1", "document": "Subject\nBody", "metadata": {}, "distance": 0.2}],
        "validation": {"sufficient": True, "usable_examples": []},
    }, path=path)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["examples"][0]["id"] == "sent-1"
    assert row["validation"]["sufficient"] is True

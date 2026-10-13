import pytest


@pytest.fixture(autouse=True)
def reply_agent_off_by_default(monkeypatch):
    """Keep tests isolated from the local Reply Agent/Ollama runtime."""
    monkeypatch.setenv("AGENTMAIL_REPLY_AGENT", "0")

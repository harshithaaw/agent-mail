# Reply Agent live path

The Reply Agent is enabled by default and routes reply generation through `agents/reply/agent.py`. Set `AGENTMAIL_REPLY_AGENT=0`, `false`, or `off` (case-insensitive) in the worker environment to use the thin path.

With the agent enabled, the graph receives the thin path's subject/body query plus sender, thread, collection, and embedding information. It returns the same `draft`, `retrieved`, and `category` fields as the thin path, with validation details when available.

If the agent raises an error or returns an empty draft, the worker logs a warning and uses the existing thin path. The returned `reply_path` is `agent`, `thin_fallback`, or `thin`, identifying which path produced the draft.

The reply gate, Gmail draft creation, trace row fields, thin retrieval behavior, and person-aware retrieval switch are unchanged. The Reply Agent's retrieval source remains `gmail_sent` and its existing context validation remains in effect.

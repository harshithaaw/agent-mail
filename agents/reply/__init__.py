"""
Reply Agent Module

SUMMARY:
The Reply Agent is responsible for generating intelligent, context-aware draft
replies to emails using a LangGraph orchestrator with local qwen3:8b via Ollama.
The graph makes explicit, inspectable decisions about retrieval, context sufficiency,
and generation through a sequence of deterministic nodes.

OVERALL FLOW:
1. Gate Check (gate.py): Pipeline calls should_generate_reply() to determine
   if email is eligible for reply generation based on route and sender type
2. Agent Orchestration (agent.py): If eligible, pipeline calls run_reply_agent()
   which runs a LangGraph with explicit decision nodes
3. Decide Retrieval (agent.py): LLM decides whether retrieval is needed
4. Retrieve (agent.py): Deterministic retrieval call with category filtering
5. Validate (agent.py): Distance-threshold check on retrieved examples
6. Retry Decision (agent.py): Graph decides whether to retry with larger k
7. Generate (agent.py): LLM generates draft using validated context
8. Assemble (agent.py): Final reason determination for trace
9. Gmail Draft Creation (gmail/draft.py): Pipeline creates Gmail draft from result

WHY THIS MODULE EXISTS:
To provide a centralized, testable, and maintainable reply generation system that:
|- Uses local LLM (qwen3:8b) instead of external APIs for cost and privacy
|- Makes each decision explicit and inspectable via trace entries
|- Leverages existing RAG infrastructure with category-aware filtering
|- Prevents unsafe or wasteful reply generation through explicit gating
|- Provides fallback paths when retrieval is insufficient

WHO CALLS THIS MODULE:
|- gmail/draft.py: create_gmail_draft() function calls run_reply_agent()
|- app.py: Main pipeline orchestrator calls run_reply_agent()
|- Test scripts: agents/reply/agent.py has built-in smoke test

MAIN COMPONENTS:
|- gate.py: should_generate_reply() - Route and sender eligibility check
|- retriever.py: retrieve_context_raw() - RAG retrieval wrapper (plain function)
|- agent.py: run_reply_agent() - LangGraph orchestrator with qwen3:8b
|- context.py: validate_context() - Distance-threshold context validation
|- generate.py: generate_reply() - Draft generation using retrieved context

DESIGN PHILOSOPHY:
Each module has a single, well-defined responsibility:
|- Gate: Security and eligibility (does this email deserve a reply?)
|- Retriever: Data access (get similar examples with category filtering)
|- Agent: Explicit decision orchestration (inspectable trace for each step)
|- Context: Quality control (distance-threshold based validation)
|- Generate: Draft production (LLM with context)
"""

__all__ = [
    'should_generate_reply',
    'is_automated_sender',
    'retrieve_context',
    'run_reply_agent',
]


def __getattr__(name):
    """Load public components on demand, avoiding model setup for utilities."""
    if name in {'should_generate_reply', 'is_automated_sender'}:
        from . import gate
        return getattr(gate, name)
    if name == 'retrieve_context':
        from .retriever import retrieve_context
        return retrieve_context
    if name == 'run_reply_agent':
        from .agent import run_reply_agent
        return run_reply_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

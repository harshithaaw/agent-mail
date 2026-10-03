"""
Reply Agent -- LangGraph orchestrator using local qwen3:8b via Ollama.

No Gemini, no API key. This graph makes the plan's original flowchart
literal instead of implicit:

    START
      |
    decide_retrieval  (LLM: does this email need retrieval?)
      |
    need_retrieval? --NO--> generate --> assemble --> END
      |YES
    retrieve            (deterministic call, no LLM)
      |
    validate            (deterministic distance-threshold check)
      |
    sufficient? --YES--> generate --> assemble --> END
      |NO
    should_retry & attempts left? --YES--> prepare_retry --> retrieve (loop)
      |NO
    generate (fallback, without context) --> assemble --> END

Each node is independently inspectable via the returned "trace" list --
that's the point. This replaces the earlier bind_tools([retrieve_context])
loop, where the model implicitly decided retrieval, sufficiency, AND
generation in one pass with no checkpoint in between. With a small local
model, that gave zero visibility into *why* it skipped retrieval on a
given email versus retrieving. Now each decision is its own node and its
own trace entry.

Eligibility (should this email get a reply at all) is NOT this module's
job -- that's gate.py, called once by the caller before run_reply_agent().
"""
import os
import re
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from agents.reply.retriever import retrieve_context_raw
from agents.reply.context import validate_context
from agents.reply.generate import generate_reply

MODEL_NAME = os.getenv("REPLY_AGENT_MODEL", "qwen3:8b")

INITIAL_K = 3
MAX_K = 6
MAX_RETRIEVAL_ATTEMPTS = 2  # initial attempt + one retry, matching k=3 then k=6

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


# Separate, deterministic (temperature=0) LLM instance used only for the
# retrieve-or-not decision. Kept separate from generate.py's instance so
# tuning one prompt can't silently affect the other.
_decision_llm = ChatOllama(model=MODEL_NAME, temperature=0.0)


class ReplyState(TypedDict):
    email_text: str
    agent_output: Dict[str, Any]
    need_retrieval: Optional[bool]
    retrieval_query: Optional[str]
    retrieval_k: int
    retrieval_attempts: int
    retrieved_examples: List[Dict[str, Any]]
    validation: Optional[Dict[str, Any]]
    draft: Optional[str]
    stopped_reason: Optional[str]
    trace: List[Dict[str, Any]]


def _decide_retrieval_node(state: ReplyState) -> dict:
    prompt = f"""Decide whether drafting a reply to this email would benefit
from looking at similar past replies for tone and structure.

Answer with exactly one word, nothing else: RETRIEVE or SKIP.

Email:
{state['email_text']}"""
    response = _decision_llm.invoke([HumanMessage(content=prompt)])
    raw = _strip_think(response.content if isinstance(response.content, str) else "")
    decision_word = raw.strip().upper()
    need = decision_word.startswith("RETRIEVE")

    return {
        "need_retrieval": need,
        "retrieval_query": state["email_text"],
        "trace": state["trace"] + [{
            "node": "decide_retrieval",
            "raw_response": raw,
            "decision": "RETRIEVE" if need else "SKIP",
        }],
    }


def _retrieve_node(state: ReplyState) -> dict:
    attempt = state["retrieval_attempts"] + 1
    k = state["retrieval_k"]
    query = state["retrieval_query"]
    category = state["agent_output"].get("category")
    # Only the user's own sent mail (gmail_sent). Synthetic fixture data
    # and Enron are excluded.
    rag_sources = ["gmail_sent"]
    results = retrieve_context_raw(query, k=k, source_filter=rag_sources, category=category)

    return {
        "retrieved_examples": results,
        "retrieval_attempts": attempt,
        "trace": state["trace"] + [{
            "node": "retrieve",
            "attempt": attempt,
            "query": query,
            "k": k,
            "source_filter": rag_sources,
            "category": category,
            "result_count": len(results),
            "distances": [r.get("distance") for r in results if isinstance(r, dict)],
        }],
    }


def _validate_node(state: ReplyState) -> dict:
    result = validate_context(state["retrieved_examples"])
    return {
        "validation": result,
        "trace": state["trace"] + [{
            "node": "validate",
            "sufficient": result["sufficient"],
            "total_retrieved": result["total_retrieved"],
            "usable_count": len(result["usable_examples"]),
            "best_distance": result["best_distance"],
            "should_retry": result["should_retry"],
        }],
    }


def _prepare_retry_node(state: ReplyState) -> dict:
    new_k = min(state["retrieval_k"] * 2, MAX_K)
    return {
        "retrieval_k": new_k,
        "trace": state["trace"] + [{"node": "prepare_retry", "new_k": new_k}],
    }


def _generate_node(state: ReplyState) -> dict:
    validation = state.get("validation")
    usable = validation["usable_examples"] if validation else []
    draft = generate_reply(state["email_text"], state["agent_output"], usable)
    return {
        "draft": draft,
        "trace": state["trace"] + [{
            "node": "generate",
            "usable_examples_count": len(usable),
        }],
    }


def _assemble_node(state: ReplyState) -> dict:
    v = state.get("validation")
    if not state["need_retrieval"]:
        reason = "no_retrieval_needed"
    elif v is None:
        reason = "unexpected_no_validation"
    elif v["sufficient"]:
        reason = "grounded_with_context"
    elif state["retrieval_attempts"] >= MAX_RETRIEVAL_ATTEMPTS:
        reason = "context_insufficient_after_retry_generated_without_context"
    else:
        reason = "context_insufficient_generated_without_context"

    if not state.get("draft"):
        reason = "empty_draft"

    return {"stopped_reason": reason}


def _route_after_decision(state: ReplyState) -> str:
    return "retrieve" if state["need_retrieval"] else "generate"


def _route_after_validate(state: ReplyState) -> str:
    v = state["validation"]
    if v["sufficient"]:
        return "generate"
    if v["should_retry"] and state["retrieval_attempts"] < MAX_RETRIEVAL_ATTEMPTS:
        return "retry"
    return "generate"  # fallback: proceed without context rather than loop forever


_graph = StateGraph(ReplyState)
_graph.add_node("decide_retrieval", _decide_retrieval_node)
_graph.add_node("retrieve", _retrieve_node)
_graph.add_node("validate", _validate_node)
_graph.add_node("prepare_retry", _prepare_retry_node)
_graph.add_node("generate", _generate_node)
_graph.add_node("assemble", _assemble_node)

_graph.set_entry_point("decide_retrieval")
_graph.add_conditional_edges("decide_retrieval", _route_after_decision, {
    "retrieve": "retrieve",
    "generate": "generate",
})
_graph.add_edge("retrieve", "validate")
_graph.add_conditional_edges("validate", _route_after_validate, {
    "generate": "generate",
    "retry": "prepare_retry",
})
_graph.add_edge("prepare_retry", "retrieve")
_graph.add_edge("generate", "assemble")
_graph.add_edge("assemble", END)

reply_graph = _graph.compile()


def run_reply_agent(email_text: str, agent_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Reply Agent for one already-gate-approved email.

    Returns:
        draft: str or None
        stopped_reason: "no_retrieval_needed" | "grounded_with_context" |
            "context_insufficient_generated_without_context" |
            "context_insufficient_after_retry_generated_without_context" |
            "empty_draft"
        trace: full ordered list of per-node decisions -- inspect this,
            not just the draft, when judging agent behavior.
        examples_used: how many examples were retrieved on the final attempt
        retrieved_examples: the raw example dicts from the final attempt
        validation: the final validate_context() result, or None if
            retrieval was never attempted
    """
    init_state: ReplyState = {
        "email_text": email_text,
        "agent_output": agent_output,
        "need_retrieval": None,
        "retrieval_query": None,
        "retrieval_k": INITIAL_K,
        "retrieval_attempts": 0,
        "retrieved_examples": [],
        "validation": None,
        "draft": None,
        "stopped_reason": None,
        "trace": [],
    }
    final_state = reply_graph.invoke(init_state)
    return {
        "draft": final_state["draft"],
        "stopped_reason": final_state["stopped_reason"],
        "trace": final_state["trace"],
        "examples_used": len(final_state["retrieved_examples"]),
        "retrieved_examples": final_state["retrieved_examples"],
        "validation": final_state["validation"],
    }


if __name__ == "__main__":
    """
    Smoke test against real fixtures. Inspect the full trace, not just
    the draft text, when judging whether this worked correctly.
    """
    import json
    from screening import screen_email
    from agents.understanding.agent import understanding_agent
    from agents.reply.gate import should_generate_reply

    TEST_FILE = "data/fixtures/test_all_routes.json"

    with open(TEST_FILE, "r") as f:
        emails = json.load(f)

    print(f"Loaded {len(emails)} emails from {TEST_FILE}.\n")

    for i, email in enumerate(emails, 1):
        message_id = email.get("message_id", f"email_{i}")
        sender_email = email["sender_email"]
        subject = email.get("subject", "")
        body = email["body"]
        email_text = f"{subject}\n\n{body}"

        print(f"[{i}/{len(emails)}] {message_id} -- subject: {subject!r}")

        screening_result = screen_email(body, sender_email)
        route = screening_result["route"]
        print(f"    route: {route}")

        if not should_generate_reply(route, sender_email):
            print("    reply generation: SKIPPED (gate decision)\n")
            continue

        understanding_result = understanding_agent.invoke({
            "messages": [HumanMessage(content=email_text)]
        })
        agent_output = understanding_result["result"]

        result = run_reply_agent(email_text, agent_output)

        print(f"    stopped_reason: {result['stopped_reason']}")
        print(f"    trace: {result['trace']}")
        print(f"    examples_used: {result['examples_used']}")
        print(f"    draft: {result['draft']!r}")
        print()

from langgraph.graph import END, StateGraph

from agents.security.injection import heuristic_injection_check, run_injection
from agents.security.phishing import phishing_score, run_phishing
from agents.security.spam import run_spam
from agents.security.state import SecurityState
from utils.preprocessing import clean_text


def preprocess(state: SecurityState) -> dict:
    cleaned_text, extracted_urls = clean_text(state["email_text"])
    urls = state["urls"] or extracted_urls
    return {"cleaned_text": cleaned_text, "urls": urls}


def injection_node(state: SecurityState) -> dict:
    return {"injection_result": run_injection(state["cleaned_text"])}


def spam_node(state: SecurityState) -> dict:
    return {"spam_result": run_spam(state["cleaned_text"])}


def phishing_node(state: SecurityState) -> dict:
    return {
        "phishing_result": run_phishing(
            state["cleaned_text"], state["urls"], state["sender_email"]
        )
    }


def make_security_decision(state: SecurityState) -> dict:
    spam = state.get("spam_result")
    phishing = state.get("phishing_result")
    injection = state.get("injection_result")

    if any(not result or result.get("error") for result in (spam, phishing, injection)):
        decision, route = "REVIEW", "flagged_classify_summarize_only"
    elif injection["flagged"]:
        decision, route = "BLOCK", "flagged_skip_reply_generation"
    elif phishing["risk_level"] == "High":
        decision, route = "BLOCK", "flagged_skip_reply_generation"
    elif phishing["risk_level"] == "Medium":
        decision, route = "REVIEW", "flagged_classify_summarize_only"
    elif spam["is_spam"]:
        decision, route = "REVIEW", "low_priority_skip_downstream"
    else:
        decision, route = "SAFE", "clean_full_pipeline"
    return {"decision": decision, "route": route}


def _report_runtime_imports() -> None:
    """Print the concrete helper implementations used by each graph node."""
    for node_name, function in (
        ("preprocess", clean_text),
        ("run_spam", run_spam),
        ("run_phishing", phishing_score),
        ("run_injection", heuristic_injection_check),
    ):
        print(f"runtime import {node_name}: {function.__module__}.{function.__qualname__}")


_graph = StateGraph(SecurityState)
_graph.add_node("preprocess", preprocess)
_graph.add_node("run_injection", injection_node)
_graph.add_node("run_spam", spam_node)
_graph.add_node("run_phishing", phishing_node)
_graph.add_node("make_security_decision", make_security_decision)
_graph.set_entry_point("preprocess")
_graph.add_edge("preprocess", "run_injection")
_graph.add_edge("run_injection", "run_spam")
_graph.add_edge("run_spam", "run_phishing")
_graph.add_edge("run_phishing", "make_security_decision")
_graph.add_edge("make_security_decision", END)
security_graph = _graph.compile()


def run_security_agent(email_text: str, sender_email: str, urls: list[str]) -> dict:
    """Run deterministic security checks and return normalized results."""
    _report_runtime_imports()
    result = security_graph.invoke(
        {
            "email_text": email_text,
            "sender_email": sender_email,
            "urls": urls,
            "cleaned_text": "",
            "spam_result": None,
            "phishing_result": None,
            "injection_result": None,
            "decision": None,
            "route": None,
        }
    )
    return {
        "decision": result["decision"],
        "route": result["route"],
        "spam": result["spam_result"],
        "phishing": result["phishing_result"],
        "injection": result["injection_result"],
    }

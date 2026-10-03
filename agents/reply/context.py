"""
Context sufficiency check for the Reply Agent.

ACTIVE MODULE. Called directly by agents/reply/agent.py's "validate" node
on every retrieval attempt. Answers: given what retrieval returned, is it
good enough to draft from, should the graph retry with a different
query/larger k, or should it give up on retrieval and generate without
examples.

DESIGN DECISION (still an assumption -- verify before trusting):
"Usable" = distance below DISTANCE_THRESHOLD. "Sufficient" = at least
MIN_USABLE_EXAMPLES usable examples. These numbers are unvalidated guesses.
Run tests/test_retrieval_quality.py against your real ChromaDB collection
(known-good vs known-bad matches for your actual emails) before trusting
them -- don't tune blind.
"""
from typing import List, Dict, Any
from langchain_core.tools import tool

# Provisional value set from real sent-mail evidence (20 emails, 8 test queries,
# see agentmail-master-context.md) — revisit once real corpus exceeds ~100 emails.
DISTANCE_THRESHOLD = 0.60
MIN_USABLE_EXAMPLES = 1

# If the *best* distance found is within this margin above the threshold,
# a retry with a different query/larger k is judged worth trying once.
# Beyond this margin, results are far enough off that retrying is unlikely
# to help -- generate without context instead of burning another retrieval
# call and another few seconds of local inference.
RETRY_WORTH_IT_MARGIN = 0.15


def validate_context(retrieved_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Decide whether retrieved examples are sufficient to generate from.

    Args:
        retrieved_examples: output of retrieve_context_raw() -- list of
            dicts with at least 'distance' (lower = more similar).

    Returns:
        {
          "sufficient": bool,
          "usable_examples": list of examples that passed the threshold,
          "total_retrieved": int,
          "best_distance": float or None,
          "should_retry": bool,   # only meaningful when sufficient is False
        }
    """
    if not retrieved_examples:
        return {
            "sufficient": False,
            "usable_examples": [],
            "total_retrieved": 0,
            "best_distance": None,
            "should_retry": True,  # nothing came back; a different query might help
        }

    usable = [
        e for e in retrieved_examples
        if isinstance(e, dict) and "distance" in e and e["distance"] < DISTANCE_THRESHOLD
    ]
    distances = [
        e["distance"] for e in retrieved_examples
        if isinstance(e, dict) and "distance" in e
    ]
    best_distance = min(distances) if distances else None
    sufficient = len(usable) >= MIN_USABLE_EXAMPLES

    should_retry = (
        not sufficient
        and best_distance is not None
        and best_distance < (DISTANCE_THRESHOLD + RETRY_WORTH_IT_MARGIN)
    )

    return {
        "sufficient": sufficient,
        "usable_examples": usable,
        "total_retrieved": len(retrieved_examples),
        "best_distance": best_distance,
        "should_retry": should_retry,
    }


@tool
def validate_context_tool(retrieved_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check whether retrieved past-email examples are relevant enough to
    ground a reply in. Pass the raw list returned by retrieve_context."""
    return validate_context(retrieved_examples)

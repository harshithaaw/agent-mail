"""
Retrieval tool for the Reply Agent.

Thin wrapper around the existing RAG pipeline (rag/retrieve.py, ChromaDB,
all-MiniLM-L6-v2). No new retrieval logic here on purpose -- this module's
only job is fetching, not deciding.

Exposed twice, deliberately:

- retrieve_context_raw(query, k): plain function. This is what
  agents/reply/agent.py's LangGraph actually calls, deterministically,
  from the "retrieve" node. No LLM is involved in this call.
- retrieve_context: the same logic wrapped as a LangChain @tool. Not
  currently used in the current LangGraph architecture, but kept
  in case a future design wants the model to call retrieval directly again.
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from rag.retrieve import retrieve_similar


def retrieve_context_raw(query: str, k: int = 3, source_filter: Optional[str | List[str]] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve up to k similar past emails from the RAG store.

    Args:
        query: The query text to find similar examples for
        k: Number of results to retrieve
        source_filter: Optional metadata filter on 'source' field (a string or
            list of accepted source values, e.g. ["gmail_sent", "real_user"])
        category: Optional metadata filter on 'category' field (e.g. "Career", "Personal")

    Returns a list of dicts with 'id', 'document', 'metadata', 'distance'
    (lower distance = more similar). Returns [] on any failure rather than
    raising -- a retrieval error should never crash the graph. Downstream,
    validate_context() will just see zero usable examples and the graph's
    fallback path (generate without context) takes over.
    """
    try:
        return retrieve_similar(query, k=k, source_filter=source_filter, category=category)
    except Exception as e:
        print(f"[retriever] retrieval failed for query={query!r}: {e}")
        return []


@tool
def retrieve_context(query: str, k: int = 3, source_filter: Optional[str | List[str]] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve up to k similar past emails from the RAG store, for tone
    and structure reference only. Each result has 'id', 'document', and
    'distance' (lower = more similar)."""
    return retrieve_context_raw(query, k=k, source_filter=source_filter, category=category)

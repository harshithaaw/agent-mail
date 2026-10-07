"""
Minimal flat RAG layer: ChromaDB persistence + embedding + retrieval.
No persona filtering yet . No reply generation (waits on routing fix).
"""
import os
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

# Persist alongside the project root
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
COLLECTION_NAME = "email_memory"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model once at module level - same pattern as summarization.py / prioritization.py
try:
    _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    EMBEDDER_AVAILABLE = True
except Exception as e:
    print(f"Error loading embedding model: {e}")
    _embedder = None
    EMBEDDER_AVAILABLE = False

_client = None
_collection = None


def get_chroma_client() -> "chromadb.PersistentClient":
    """Persistent ChromaDB client. Data survives restarts across review sessions."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_or_create_collection(name: str = COLLECTION_NAME):
    """
    Single flat collection. Future persona-aware filtering would use a metadata
    field on each vector, not a separate collection - avoids a schema migration later.

    Uses cosine space for distance calculations.
    """
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def embed_text(text: str) -> List[float]:
    """
    Embed a single string with all-MiniLM-L6-v2.
    """
    if not EMBEDDER_AVAILABLE:
        raise RuntimeError("Embedding model failed to load - check sentence-transformers install")
    vec = _embedder.encode(text, convert_to_numpy=True)
    return vec.tolist()


def add_examples(
    ids: List[str],
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    upsert: bool = False,
) -> None:
    """
    Embed and store a batch of example texts (e.g. sent emails) into the collection.

    Args:
        ids: unique string id per example
        texts: raw text per example (subject+body concatenated by caller)
        metadatas: optional per-example metadata dicts, e.g. {"source": "enron"}
    """
    if len(ids) != len(texts):
        raise ValueError("ids and texts must be the same length")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must match texts length if provided")

    collection = get_or_create_collection()
    embeddings = [embed_text(t) for t in texts]

    write = collection.upsert if upsert else collection.add
    write(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas or [{} for _ in texts],
    )


def retrieve_similar(
    query_text: str,
    k: int = 3,
    source_filter: Optional[str | List[str]] = None,
    category: Optional[str] = None,
    person_addr: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most similar stored examples for a query email.
    Flat retrieval, optionally reranked for a matching correspondent/thread.

    Args:
        query_text: The query text to find similar examples for
        k: Number of results to retrieve
        source_filter: Optional metadata filter on 'source' field (e.g. "real_user", "gmail_sent", "enron"). A list matches any listed source.
        category: Optional metadata filter on 'category' field (e.g. "Career", "Personal")
                 If provided and combined with source_filter, uses AND logic.
                 If category filter returns zero results, falls back to source_filter-only.
        person_addr: Normalized correspondent address for optional person-aware reranking.
        thread_id: Current Gmail thread ID for an optional thread-match bonus.

    Returns:
        List of dicts: {"id": str, "document": str, "metadata": dict, "distance": float}
    """
    collection = get_or_create_collection()
    query_embedding = embed_text(query_text)

    # Build query parameters
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": k,
    }

    # Build where clause based on filters
    where_clause = {}
    if source_filter:
        where_clause["source"] = (
            {"$in": source_filter}
            if isinstance(source_filter, list)
            else source_filter
        )

    # Try with category filter first if provided
    if category:
        # Use AND operator for multiple conditions
        where_clause_with_category = {
            "$and": [
                ({"source": {"$in": source_filter}}
                 if isinstance(source_filter, list)
                 else {"source": source_filter}) if source_filter else {},
                {"category": category}
            ]
        }
        # Remove empty conditions
        where_clause_with_category["$and"] = [
            cond for cond in where_clause_with_category["$and"] if cond
        ]
        if len(where_clause_with_category["$and"]) == 1:
            where_clause_with_category = where_clause_with_category["$and"][0]

        query_params["where"] = where_clause_with_category

        results = collection.query(**query_params)

        # Check if we got results with category filter
        if results.get("ids", [[]])[0]:
            # Category filter worked, use these results
            pass
        else:
            # Category filter returned zero results, fall back to source_filter-only
            print(f"[retrieve_similar] Category filter '{category}' returned zero results, falling back to source_filter-only")
            query_params["where"] = where_clause if where_clause else None
            results = collection.query(**query_params)
    else:
        # No category filter, use where clause as-is
        if where_clause:
            query_params["where"] = where_clause
        results = collection.query(**query_params)

    out = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        out.append({
            "id": ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],
        })

    if os.getenv("AGENTMAIL_PERSON_AWARE") != "1" or not person_addr:
        return out

    person_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"$and": [
            {"source": "gmail_sent"},
            {"recipient_norm": person_addr},
        ]},
    )
    person_ids = person_results.get("ids", [[]])[0]
    person_documents = person_results.get("documents", [[]])[0]
    person_metadatas = person_results.get("metadatas", [[]])[0]
    person_distances = person_results.get("distances", [[]])[0]
    by_id = {result["id"]: result for result in out}
    for i, result_id in enumerate(person_ids):
        by_id.setdefault(result_id, {
            "id": result_id,
            "document": person_documents[i],
            "metadata": person_metadatas[i],
            "distance": person_distances[i],
        })

    def adjusted_distance(result):
        metadata = result.get("metadata") or {}
        bonus = 0.05 if metadata.get("recipient_norm") == person_addr else 0.0
        if thread_id and metadata.get("thread") == thread_id:
            bonus += 0.05
        return result["distance"] - bonus

    return sorted(by_id.values(), key=adjusted_distance)[:k]


def collection_count() -> int:
    """Sanity-check helper: how many examples are currently stored."""
    return get_or_create_collection().count()


if __name__ == "__main__":
    # Smoke test
    print("Running RAG layer smoke test...")
    print(f"Chroma persistence path: {CHROMA_PATH}")
    print(f"Embedder available: {EMBEDDER_AVAILABLE}")

    add_examples(
        ids=["test_1", "test_2"],
        texts=[
            "Hi team, quick update - the report is attached, let me know your thoughts.",
            "Hey! Are we still on for coffee Friday? Let me know what time works.",
        ],
        metadatas=[{"source": "smoke_test"}, {"source": "smoke_test"}],
    )
    print(f"Collection count after add: {collection_count()}")

    results = retrieve_similar("Can you send me the quarterly report?", k=2)
    print("\nRetrieval results:")
    for r in results:
        print(f"  id={r['id']} distance={r['distance']:.4f} doc={r['document'][:60]}...")

    print("\nSmoke test complete.")

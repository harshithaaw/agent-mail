"""
Test retrieval quality independently of the Reply Agent.

Bypasses agent.py and every LLM entirely. Calls the RAG layer directly so
you can see actual distances for known-similar and known-unrelated emails
BEFORE trusting context.py's DISTANCE_THRESHOLD (currently 0.35).

Run this against your REAL, populated ChromaDB collection -- not the
throwaway smoke-test collection created by `python rag/retrieve.py`
directly, or every query below will just match "test_1"/"test_2".
"""
from rag.retrieve import retrieve_similar, collection_count

# Replace these with real content from your own ingested examples. The
# point of A vs B is to know, in advance, that they SHOULD pull back
# different stored examples -- so you can judge whether distance actually
# tracks relevance for your data, not just "it returned something."
TEST_QUERIES = [
    {
        "label": "Email A (expect: similar work-status-update examples)",
        "text": "Hi team, just a quick update on the Q3 report status, "
                "should have it ready by Friday.",
    },
    {
        "label": "Email B (expect: DIFFERENT examples than A -- casual/personal)",
        "text": "Hey! Are we still on for coffee Friday? Let me know what "
                "time works for you.",
    },
    {
        "label": "Unrelated email (expect: no confident match)",
        "text": "The quarterly board meeting has been rescheduled to next "
                "Tuesday at 10am in conference room B.",
    },
]

K = 5


def run():
    count = collection_count()
    print(f"Chroma collection currently holds {count} examples.")
    if count == 0:
        print("Collection is empty -- populate it via your ingestion script "
              "before running this test.")
        return

    for q in TEST_QUERIES:
        print("\n" + "=" * 80)
        print(q["label"])
        print("-" * 80)
        results = retrieve_similar(q["text"], k=K)
        if not results:
            print("  (no results)")
            continue
        for r in results:
            doc = r.get("document", "")
            preview = (doc[:90] + "...") if len(doc) > 90 else doc
            dist = r.get("distance")
            dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "?"
            print(f"  distance={dist_str}  id={r.get('id')}")
            print(f"    {preview}")

    print("\n" + "=" * 80)
    print("Inspect manually:")
    print("- Do Email A's and Email B's top matches actually look relevant to them?")
    print("- Do A and B pull back noticeably DIFFERENT examples from each other?")
    print("- Is the unrelated email's best distance clearly WORSE (higher) than")
    print("  A/B's best matches? If not, DISTANCE_THRESHOLD (0.35 in context.py)")
    print("  isn't doing useful work and needs to be recalibrated against these")
    print("  real numbers -- don't leave it as a guess.")


if __name__ == "__main__":
    run()
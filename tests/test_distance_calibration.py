"""
Distance calibration -- run this BEFORE changing DISTANCE_THRESHOLD.

Answers two separate questions that test_retrieval_quality.py's output
conflates:

1. SCALE: what does "close" even look like in this collection's current
   distance space? (percentiles of nearest-neighbor distances WITHIN the
   corpus itself -- doc vs. its own nearest neighbors, excluding itself)

2. FIT: are the out-of-domain test queries (personal/casual/business
   emails you'd actually send) meaningfully closer to SOME corpus
   documents than a typical corpus document is to another random corpus
   document? If out-of-domain best-distances land INSIDE the normal
   within-corpus range, the corpus has usable matches and this is a
   threshold problem. If they land clearly WORSE (higher) than the
   within-corpus range, the corpus itself doesn't contain relevant
   material for these emails, and no threshold fixes that -- you need
   different source data.

This does not change anything -- it only measures. Decide the new
threshold and any corpus changes from what this prints, not before.
"""
import random
import statistics
from rag.retrieve import get_or_create_collection, retrieve_similar, collection_count

# Reuse the same out-of-domain queries as test_retrieval_quality.py so the
# two reports are directly comparable.
OUT_OF_DOMAIN_QUERIES = [
    "Hi team, just a quick update on the Q3 report status, should have it ready by Friday.",
    "Hey! Are we still on for coffee Friday? Let me know what time works for you.",
    "The quarterly board meeting has been rescheduled to next Tuesday at 10am in conference room B.",
    "Want to grab lunch at noon?",
    "Can we push our 1:1 tomorrow from 10am to 2pm? Something came up in the morning.",
]

SAMPLE_SIZE = 30  # how many corpus docs to use as within-corpus anchors
K = 5


def _percentiles(values, ps=(10, 25, 50, 75, 90)):
    if not values:
        return {p: None for p in ps}
    sorted_vals = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(sorted_vals) - 1, int(round((p / 100) * (len(sorted_vals) - 1))))
        out[p] = sorted_vals[idx]
    return out


def run():
    count = collection_count()
    print(f"Collection holds {count} examples.\n")
    if count < SAMPLE_SIZE:
        print(f"WARNING: fewer than {SAMPLE_SIZE} examples -- sampling all of them instead.")

    collection = get_or_create_collection()
    all_data = collection.get(include=["documents"])
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]

    sample_size = min(SAMPLE_SIZE, len(all_ids))
    sample_indices = random.sample(range(len(all_ids)), sample_size)

    print("=" * 80)
    print(f"STEP 1: Within-corpus nearest-neighbor distances (n={sample_size} anchors, k={K})")
    print("-" * 80)
    within_corpus_best = []
    within_corpus_all = []
    for idx in sample_indices:
        doc_text = all_docs[idx]
        doc_id = all_ids[idx]
        results = retrieve_similar(doc_text, k=K + 1)  # +1 because the doc will match itself at distance ~0
        # drop self-match
        others = [r for r in results if r["id"] != doc_id]
        if not others:
            continue
        within_corpus_best.append(others[0]["distance"])
        within_corpus_all.extend(r["distance"] for r in others)

    wc_pct = _percentiles(within_corpus_best)
    print(f"Best-match distance percentiles (each anchor's closest OTHER doc):")
    for p, v in wc_pct.items():
        print(f"  p{p}: {v:.4f}" if v is not None else f"  p{p}: n/a")
    print(f"Overall within-corpus distance range (all neighbors, not just best): "
          f"{min(within_corpus_all):.4f} to {max(within_corpus_all):.4f}"
          if within_corpus_all else "  (no data)")

    print("\n" + "=" * 80)
    print("STEP 2: Out-of-domain query best-match distances")
    print("-" * 80)
    ood_best = []
    for q in OUT_OF_DOMAIN_QUERIES:
        results = retrieve_similar(q, k=1)
        if results:
            d = results[0]["distance"]
            ood_best.append(d)
            print(f"  {d:.4f}  <- {q[:70]}")

    print("\n" + "=" * 80)
    print("STEP 3: Verdict")
    print("-" * 80)
    if not within_corpus_best or not ood_best:
        print("Not enough data to compare -- check collection_count() and retrieval above.")
        return

    wc_median = statistics.median(within_corpus_best)
    ood_median = statistics.median(ood_best)
    print(f"Within-corpus median best-distance : {wc_median:.4f}")
    print(f"Out-of-domain median best-distance : {ood_median:.4f}")

    if ood_median <= wc_pct[75]:
        print(
            "\n-> Out-of-domain queries land WITHIN the normal within-corpus range.\n"
            "   This looks like a THRESHOLD problem, not a corpus problem.\n"
            f"   Candidate DISTANCE_THRESHOLD: something around p50-p75 of the\n"
            f"   within-corpus best-distances above ({wc_pct[50]:.4f} - {wc_pct[75]:.4f}),\n"
            "   not the current 0.35."
        )
    else:
        print(
            "\n-> Out-of-domain queries are consistently WORSE than typical within-corpus\n"
            "   matches. This looks like a CORPUS FIT problem: the stored examples\n"
            "   (Enron corpus) may not contain material relevant to the kind of email\n"
            "   you're actually testing against. Raising or lowering the threshold\n"
            "   will not fix this -- consider ingesting real sent-mail examples before\n"
            "   trusting retrieval for these categories."
        )

    print(
        "\nNOTE: this run used whatever distance space the collection was created\n"
        "with (check if metadata={'hnsw:space': ...} was ever set -- if not, it's\n"
        "Chroma's default, squared L2, on non-unit-normalized MiniLM embeddings).\n"
        "If you migrate the collection to cosine space, re-run this script --\n"
        "the raw numbers above will change and any threshold decided from this\n"
        "run becomes stale."
    )


if __name__ == "__main__":
    run()
"""
Recalibration, broken out per category, for the new email_memory_v2 collection.

Reproduces the same idea as your original calibration script (within-corpus
nearest-neighbor distances vs. out-of-domain distances), but reports it per
category so you can see exactly which categories are/aren't well covered
yet, instead of one aggregate number.

Usage:
    python recalibrate_by_category.py --collection email_memory_v2
"""

import argparse
import statistics
from collections import defaultdict

import chromadb

# A handful of generic out-of-domain probes, same spirit as your original
# calibration run. Swap these for real held-out examples from each category
# once you have more than the minimum 15/category.
DEFAULT_OOD_QUERIES = [
    "Hi team, just a quick update on the Q3 report status, should have it ready soon.",
    "Hey! Are we still on for coffee Friday? Let me know what time works for you.",
    "The quarterly board meeting has been rescheduled to next Tuesday at 10am.",
    "Want to grab lunch at noon?",
    "Can we push our 1:1 tomorrow from 10am to 2pm? Something came up.",
]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="email_memory_v2")
    parser.add_argument("--db-path", default="./chroma_db")
    parser.add_argument("--k", type=int, default=5, help="Neighbors to pull per query")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(args.collection)

    all_docs = collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    docs = all_docs["documents"]
    metas = all_docs["metadatas"]

    by_category = defaultdict(list)
    for doc_id, doc, meta in zip(ids, docs, metas):
        by_category[meta.get("category", "uncategorized")].append((doc_id, doc))

    print("=" * 80)
    print("WITHIN-CATEGORY NEAREST-NEIGHBOR DISTANCES")
    print("=" * 80)

    within_all = []
    for category, items in sorted(by_category.items()):
        if len(items) < 2:
            print(f"\n{category}: only {len(items)} example(s), skipping (need >=2).")
            continue
        best_dists = []
        for anchor_id, anchor_doc in items:
            result = collection.query(
                query_texts=[anchor_doc],
                n_results=min(args.k + 1, len(items)),
                where={"category": category},
            )
            # drop the anchor itself (distance ~0 to itself), take best of the rest
            dists = [d for i, d in zip(result["ids"][0], result["distances"][0]) if i != anchor_id]
            if dists:
                best_dists.append(min(dists))
        within_all.extend(best_dists)
        print(f"\n{category}  (n={len(items)}):")
        print(f"  median best-match distance: {statistics.median(best_dists):.4f}")
        print(f"  p75: {percentile(best_dists, 75):.4f}")

    print("\n" + "=" * 80)
    print("OUT-OF-DOMAIN PROBE DISTANCES (best match, any category)")
    print("=" * 80)
    ood_dists = []
    for q in DEFAULT_OOD_QUERIES:
        result = collection.query(query_texts=[q], n_results=1)
        dist = result["distances"][0][0] if result["distances"][0] else float("nan")
        ood_dists.append(dist)
        print(f"  {dist:.4f}  <- {q[:70]}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if within_all and ood_dists:
        w_median = statistics.median(within_all)
        w_p75 = percentile(within_all, 75)
        o_median = statistics.median(ood_dists)
        print(f"Overall within-corpus median: {w_median:.4f}  (p75: {w_p75:.4f})")
        print(f"Out-of-domain probe median  : {o_median:.4f}")
        if o_median > w_p75:
            print("\n-> Still a gap between real matches and the OOD probes above p75.")
            print("   That's expected -- the probes are meant to be dissimilar.")
            print("   What matters now: are per-category medians tight and low")
            print("   compared to your OLD Enron numbers (0.16 within / 0.71 OOD)?")
        else:
            print("\n-> OOD probes are landing inside your within-corpus range.")
            print("   Either the probes aren't actually OOD for your categories,")
            print("   or your categories are broad enough to catch them -- check by hand.")
    print("\nNext: set DISTANCE_THRESHOLD from the per-category p75/p90 values above.")


if __name__ == "__main__":
    main()
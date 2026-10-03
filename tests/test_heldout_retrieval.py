"""
Evaluate RAG retrieval quality on held-out examples.

This script tests whether our categorized RAG corpus retrieves useful examples
for unseen emails before we choose a distance threshold.

Usage:
    python test_heldout_retrieval.py --input examples_template.jsonl
"""

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer


def load_examples(path: str) -> List[Dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {line_num}: {e}", file=__import__("sys").stderr)
                continue
            missing = [k for k in ("category", "incoming", "reply") if k not in obj]
            if missing:
                print(f"Skipping line {line_num}, missing keys: {missing}", file=__import__("sys").stderr)
                continue
            examples.append(obj)
    return examples


def split_examples_by_category(examples: List[Dict[str, Any]], random_state: int = 42) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Deterministically split examples by category into train and held-out sets.
    
    Splits:
    - Academic: 15 train, 4 held-out
    - Career: 15 train, 5 held-out
    - Personal: 15 train, 5 held-out
    - Promotional: 15 train, 5 held-out
    """
    import random
    random.seed(random_state)
    
    # Define split targets per category
    split_targets = {
        "Academic": {"train": 15, "heldout": 4},
        "Career": {"train": 15, "heldout": 5},
        "Personal": {"train": 15, "heldout": 5},
        "Promotional": {"train": 15, "heldout": 5},
    }
    
    # Group examples by category
    by_category = defaultdict(list)
    for ex in examples:
        by_category[ex["category"]].append(ex)
    
    # Split each category
    splits = {}
    for category, items in by_category.items():
        random.shuffle(items)
        target = split_targets.get(category, {"train": len(items), "heldout": 0})
        
        train_size = min(target["train"], len(items))
        heldout_size = min(target["heldout"], len(items) - train_size)
        
        splits[category] = {
            "train": items[:train_size],
            "heldout": items[train_size:train_size + heldout_size],
        }
        
        print(f"{category}: {len(splits[category]['train'])} train, {len(splits[category]['heldout'])} held-out")
    
    return splits


def create_temp_collection(train_examples: List[Dict[str, Any]], model: SentenceTransformer) -> chromadb.Collection:
    """Create a temporary Chroma collection with training examples only."""
    # Create temporary directory for Chroma
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    
    client = chromadb.PersistentClient(path=temp_dir)
    collection = client.get_or_create_collection(
        name="test_heldout_collection",
        metadata={"hnsw:space": "cosine"},
    )
    
    # Prepare data for insertion
    ids = [f"{ex['category']}_train_{i}" for i, ex in enumerate(train_examples)]
    documents = [ex["incoming"] for ex in train_examples]
    embeddings = [model.encode(doc, convert_to_numpy=True).tolist() for doc in documents]
    metadatas = [
        {
            "category": ex["category"],
            "reply": ex["reply"],
            "source": ex.get("source", "unknown"),
        }
        for ex in train_examples
    ]
    
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    
    print(f"Created temporary collection with {len(train_examples)} examples in {temp_dir}")
    return collection


def percentile(values: List[float], p: float) -> float:
    """Calculate percentile."""
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
    parser.add_argument("--input", required=True, help="Path to examples .jsonl file")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve per query")
    parser.add_argument("--output", default="heldout_retrieval_results.json", help="Output JSON file for results")
    args = parser.parse_args()
    
    # Load examples
    print("Loading examples...")
    examples = load_examples(args.input)
    print(f"Loaded {len(examples)} total examples\n")
    
    # Split by category
    print("Splitting examples by category (random_state=42)...")
    splits = split_examples_by_category(examples, random_state=42)
    
    # Flatten training examples
    train_examples = []
    for category in splits:
        train_examples.extend(splits[category]["train"])
    print(f"Total training examples: {len(train_examples)}\n")
    
    # Load embedding model (same as production)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create temporary collection
    collection = create_temp_collection(train_examples, model)
    
    # Evaluate held-out examples
    print("\n" + "=" * 80)
    print("EVALUATING HELD-OUT RETRIEVAL")
    print("=" * 80 + "\n")
    
    all_results = []
    category_stats = defaultdict(lambda: {"distances": [], "category_matches": 0, "total": 0})
    
    for category in sorted(splits.keys()):
        heldout = splits[category]["heldout"]
        if not heldout:
            continue
        
        print(f"\n--- {category} ({len(heldout)} held-out examples) ---\n")
        
        for i, ex in enumerate(heldout, 1):
            # Query with category filtering
            query_embedding = model.encode(ex["incoming"], convert_to_numpy=True).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=args.k,
                where={"category": category},
            )
            
            # Process results
            retrieved = []
            for rank in range(len(results["ids"][0])):
                retrieved.append({
                    "rank": rank + 1,
                    "distance": results["distances"][0][rank],
                    "category": results["metadatas"][0][rank]["category"],
                    "incoming": results["documents"][0][rank],
                    "reply": results["metadatas"][0][rank]["reply"],
                })
            
            # Record top-1 stats
            if retrieved:
                top1 = retrieved[0]
                category_stats[category]["distances"].append(top1["distance"])
                category_stats[category]["total"] += 1
                if top1["category"] == category:
                    category_stats[category]["category_matches"] += 1
            
            # Print detailed results
            print(f"Query {i}/{len(heldout)}:")
            print(f"  Category: {category}")
            print(f"  Incoming: {ex['incoming'][:100]}...")
            
            for r in retrieved:
                print(f"  Rank {r['rank']}: distance={r['distance']:.4f}, category={r['category']}")
                print(f"    Retrieved incoming: {r['incoming'][:80]}...")
                print(f"    Retrieved reply: {r['reply'][:80]}...")
            
            # Store result
            all_results.append({
                "heldout_category": category,
                "heldout_incoming": ex["incoming"],
                "heldout_reply": ex["reply"],
                "retrieved": retrieved,
            })
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80 + "\n")
    
    # Overall stats
    all_distances = []
    total_queries = 0
    total_category_matches = 0
    
    for category, stats in category_stats.items():
        all_distances.extend(stats["distances"])
        total_queries += stats["total"]
        total_category_matches += stats["category_matches"]
    
    print("OVERALL:")
    print(f"  Total held-out queries: {total_queries}")
    if all_distances:
        print(f"  Top-1 distance median: {statistics.median(all_distances):.4f}")
        print(f"  Top-1 distance p75: {percentile(all_distances, 75):.4f}")
        print(f"  Top-1 distance p90: {percentile(all_distances, 90):.4f}")
    print(f"  Top-1 category match rate: {total_category_matches}/{total_queries} ({100*total_category_matches/total_queries if total_queries > 0 else 0:.1f}%)")
    
    # Per-category stats
    print("\nPER CATEGORY:")
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        distances = stats["distances"]
        
        print(f"\n  {category}:")
        print(f"    Queries: {stats['total']}")
        if distances:
            print(f"    Median top-1 distance: {statistics.median(distances):.4f}")
            print(f"    p75 top-1 distance: {percentile(distances, 75):.4f}")
            print(f"    p90 top-1 distance: {percentile(distances, 90):.4f}")
        print(f"    Category match rate: {stats['category_matches']}/{stats['total']} ({100*stats['category_matches']/stats['total'] if stats['total'] > 0 else 0:.1f}%)")
    
    # Save results to JSON
    output_data = {
        "summary": {
            "total_queries": total_queries,
            "overall_median_distance": statistics.median(all_distances) if all_distances else None,
            "overall_p75_distance": percentile(all_distances, 75) if all_distances else None,
            "overall_p90_distance": percentile(all_distances, 90) if all_distances else None,
            "overall_category_match_rate": total_category_matches / total_queries if total_queries > 0 else None,
            "per_category": {
                category: {
                    "queries": stats["total"],
                    "median_distance": statistics.median(stats["distances"]) if stats["distances"] else None,
                    "p75_distance": percentile(stats["distances"], 75) if stats["distances"] else None,
                    "p90_distance": percentile(stats["distances"], 90) if stats["distances"] else None,
                    "category_match_rate": stats["category_matches"] / stats["total"] if stats["total"] > 0 else None,
                }
                for category, stats in category_stats.items()
            },
        },
        "detailed_results": all_results,
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Manually inspect the retrieved examples above for semantic usefulness")
    print("2. Review the distance statistics to inform DISTANCE_THRESHOLD selection")
    print("3. Consider adding more examples if category match rates are low")
    print("=" * 80)


if __name__ == "__main__":
    main()
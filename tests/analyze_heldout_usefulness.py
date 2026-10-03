"""
Analyze held-out retrieval results for manual usefulness labeling.

This script loads the heldout_retrieval_results.json and creates a compact table
for manual inspection and labeling of retrieval usefulness.

Purpose: Determine whether distance correlates with actual semantic usefulness
before selecting DISTANCE_THRESHOLD.

Usage:
    python tests/analyze_heldout_usefulness.py
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_results(results_path: str) -> Dict[str, Any]:
    """Load held-out retrieval results from JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_usefulness_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create a compact table of held-out queries for manual usefulness labeling.
    
    Returns a list of dicts with fields:
    - category
    - heldout_incoming
    - top1_distance
    - retrieved_incoming
    - retrieved_reply
    - usefulness (blank for manual filling)
    """
    table = []
    
    for result in results["detailed_results"]:
        if result["retrieved"]:
            top1 = result["retrieved"][0]
            table.append({
                "category": result["heldout_category"],
                "heldout_incoming": result["heldout_incoming"],
                "top1_distance": top1["distance"],
                "retrieved_incoming": top1["incoming"],
                "retrieved_reply": top1["reply"],
                "usefulness": "",  # To be filled manually
            })
    
    return table


def calculate_summary(table: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics once usefulness labels are filled.
    
    Returns:
    - useful_count
    - partially_useful_count
    - not_useful_count
    - usefulness_rate
    - median_distance_useful
    - median_distance_partially_useful
    - median_distance_not_useful
    - per_category breakdown
    """
    import statistics
    
    # Group by usefulness
    by_usefulness = {
        "useful": [],
        "partially_useful": [],
        "not_useful": [],
    }
    
    # Group by category and usefulness
    by_category = {}
    
    for row in table:
        usefulness = row.get("usefulness", "").strip().lower()
        if usefulness in by_usefulness:
            by_usefulness[usefulness].append(row["top1_distance"])
        
        category = row["category"]
        if category not in by_category:
            by_category[category] = {
                "useful": [],
                "partially_useful": [],
                "not_useful": [],
                "total": 0,
            }
        by_category[category]["total"] += 1
        if usefulness in by_category[category]:
            by_category[category][usefulness].append(row["top1_distance"])
    
    # Calculate overall stats
    labeled_count = sum(len(v) for v in by_usefulness.values())
    useful_count = len(by_usefulness["useful"])
    partially_useful_count = len(by_usefulness["partially_useful"])
    not_useful_count = len(by_usefulness["not_useful"])
    
    summary = {
        "overall": {
            "total_queries": len(table),
            "labeled_count": labeled_count,
            "useful_count": useful_count,
            "partially_useful_count": partially_useful_count,
            "not_useful_count": not_useful_count,
            "usefulness_rate": useful_count / labeled_count if labeled_count > 0 else None,
            "median_distance_useful": statistics.median(by_usefulness["useful"]) if by_usefulness["useful"] else None,
            "median_distance_partially_useful": statistics.median(by_usefulness["partially_useful"]) if by_usefulness["partially_useful"] else None,
            "median_distance_not_useful": statistics.median(by_usefulness["not_useful"]) if by_usefulness["not_useful"] else None,
        },
        "per_category": {},
    }
    
    # Calculate per-category stats
    for category, stats in by_category.items():
        summary["per_category"][category] = {
            "total_queries": stats["total"],
            "useful_count": len(stats["useful"]),
            "partially_useful_count": len(stats["partially_useful"]),
            "not_useful_count": len(stats["not_useful"]),
            "usefulness_rate": len(stats["useful"]) / stats["total"] if stats["total"] > 0 else None,
            "median_distance_useful": statistics.median(stats["useful"]) if stats["useful"] else None,
            "median_distance_partially_useful": statistics.median(stats["partially_useful"]) if stats["partially_useful"] else None,
            "median_distance_not_useful": statistics.median(stats["not_useful"]) if stats["not_useful"] else None,
        }
    
    return summary


def main():
    # Check if labeled data already exists
    usefulness_path = "heldout_usefulness.json"
    if Path(usefulness_path).exists():
        print(f"Loading existing labeled data from {usefulness_path}...")
        with open(usefulness_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        table = existing_data.get("table", [])
        print(f"Loaded {len(table)} labeled queries\n")
    else:
        # Load results and create new table
        results_path = "heldout_retrieval_results.json"
        print(f"Loading results from {results_path}...")
        results = load_results(results_path)
        
        # Create usefulness table
        print("Creating usefulness table...")
        table = create_usefulness_table(results)
        print(f"Created table with {len(table)} queries\n")
    
    # Calculate summary (will be None until labels are filled)
    summary = calculate_summary(table)
    
    # Prepare output
    output = {
        "table": table,
        "summary": summary,
        "instructions": {
            "usefulness_values": ["useful", "partially_useful", "not_useful"],
            "fill_instructions": "Manually fill the 'usefulness' field for each row based on semantic relevance",
            "recalculate_after_labeling": "Run this script again after filling labels to get updated statistics",
        },
    }
    
    # Save to JSON
    output_path = "heldout_usefulness.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to {output_path}")
    
    # Print summary statistics if labels exist
    labeled_count = sum(1 for row in table if row.get("usefulness", "").strip() in ["useful", "partially_useful", "not_useful"])
    
    if labeled_count > 0:
        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)
        
        overall = summary["overall"]
        print(f"\nOVERALL:")
        print(f"  Total queries: {overall['total_queries']}")
        print(f"  Labeled: {overall['labeled_count']}")
        print(f"  Useful: {overall['useful_count']}")
        print(f"  Partially useful: {overall['partially_useful_count']}")
        print(f"  Not useful: {overall['not_useful_count']}")
        if overall['usefulness_rate'] is not None:
            print(f"  Usefulness rate: {overall['usefulness_rate']:.1%}")
        if overall['median_distance_useful'] is not None:
            print(f"  Median distance (useful): {overall['median_distance_useful']:.4f}")
        if overall['median_distance_partially_useful'] is not None:
            print(f"  Median distance (partially useful): {overall['median_distance_partially_useful']:.4f}")
        if overall['median_distance_not_useful'] is not None:
            print(f"  Median distance (not useful): {overall['median_distance_not_useful']:.4f}")
        
        print(f"\nPER CATEGORY:")
        for category, stats in summary["per_category"].items():
            print(f"\n  {category}:")
            print(f"    Total: {stats['total_queries']}")
            print(f"    Useful: {stats['useful_count']}")
            print(f"    Partially useful: {stats['partially_useful_count']}")
            print(f"    Not useful: {stats['not_useful_count']}")
            if stats['usefulness_rate'] is not None:
                print(f"    Usefulness rate: {stats['usefulness_rate']:.1%}")
            if stats['median_distance_useful'] is not None:
                print(f"    Median distance (useful): {stats['median_distance_useful']:.4f}")
            if stats['median_distance_partially_useful'] is not None:
                print(f"    Median distance (partially useful): {stats['median_distance_partially_useful']:.4f}")
            if stats['median_distance_not_useful'] is not None:
                print(f"    Median distance (not useful): {stats['median_distance_not_useful']:.4f}")
    else:
        # Print table for manual inspection
        print("\n" + "=" * 100)
        print("HELD-OUT USEFULNESS TABLE (for manual labeling)")
        print("=" * 100)
        print(f"\n{'Category':<12} {'Distance':<10} {'Usefulness':<18} {'Held-out Incoming':<40} {'Retrieved Incoming':<40}")
        print("-" * 100)
        
        for i, row in enumerate(table, 1):
            print(f"{row['category']:<12} {row['top1_distance']:<10.4f} {row['usefulness']:<18} {row['heldout_incoming'][:38]:<40} {row['retrieved_incoming'][:38]:<40}")
            
            # Show retrieved reply on next line for context
            print(f"{'':12} {'':10} {'':18} {'':40} → Reply: {row['retrieved_reply'][:60]}...")
            
            if i < len(table):
                print()
        
        print("\n" + "=" * 100)
        print("INSTRUCTIONS")
        print("=" * 100)
        print("\n1. Open heldout_usefulness.json")
        print("2. For each row in the 'table' array, set 'usefulness' to one of:")
        print("   - useful: The retrieved example is semantically relevant and helpful")
        print("   - partially_useful: Somewhat relevant but limited utility")
        print("   - not_useful: Not semantically relevant or helpful")
        print("3. Save the file")
        print("4. Run this script again to calculate summary statistics")
        print("\nPurpose: Determine if distance correlates with actual semantic usefulness")
        print("         before selecting DISTANCE_THRESHOLD for production.")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
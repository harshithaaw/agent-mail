"""
Build a fresh, categorized, cosine-space Chroma collection from your own
email->reply examples (replacing the Enron seed corpus).

Usage:
    python build_corpus.py --input examples_template.jsonl --collection email_memory_v2

Input file: one JSON object per line, with keys:
    category  (str, required)
    incoming  (str, required)  -- the received email text; this is what gets embedded
    reply     (str, required)  -- your actual reply; stored as metadata, not embedded
    source    (str, optional)  -- e.g. "gmail", "synthetic"
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb


def load_examples(path: str):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {line_num}: {e}", file=sys.stderr)
                continue
            missing = [k for k in ("category", "incoming", "reply") if k not in obj]
            if missing:
                print(f"Skipping line {line_num}, missing keys: {missing}", file=sys.stderr)
                continue
            examples.append(obj)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to examples .jsonl file")
    parser.add_argument("--collection", default="email_memory_v2", help="Target Chroma collection name")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to Chroma persistent store")
    parser.add_argument("--reset", action="store_true", help="Delete the collection first if it already exists")
    args = parser.parse_args()

    examples = load_examples(args.input)
    if not examples:
        print("No valid examples loaded, aborting.", file=sys.stderr)
        sys.exit(1)

    by_cat = {}
    for ex in examples:
        by_cat.setdefault(ex["category"], 0)
        by_cat[ex["category"]] += 1
    print("Loaded examples per category:")
    for cat, n in sorted(by_cat.items()):
        flag = "  <-- consider adding more (aim for 15+)" if n < 15 else ""
        print(f"  {cat}: {n}{flag}")

    client = chromadb.PersistentClient(path=args.db_path)

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted existing collection '{args.collection}'.")
        except Exception:
            pass

    # Explicit cosine space -- no more silent default-L2 surprises.
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{ex['category']}_{i}" for i, ex in enumerate(examples)]
    documents = [ex["incoming"] for ex in examples]  # embed the incoming email only
    metadatas = [
        {
            "category": ex["category"],
            "reply": ex["reply"],
            "source": ex.get("source", "unknown"),
        }
        for ex in examples
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"\nIngested {len(examples)} examples into collection '{args.collection}' "
          f"at {Path(args.db_path).resolve()} (cosine space).")
    print("Next: run recalibrate_by_category.py against this collection.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Evaluate current top-3 retrieval on one synthetic query split."""
import argparse
import json
from pathlib import Path

from agents.reply.retriever import retrieve_context_raw
from scripts.load_synthetic import STORE
from scripts.synthetic_chroma import chroma_at

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data/synthetic/queries.jsonl"


def evaluate(queries_path=QUERIES, chroma_path=STORE, split="dev", *, retrieve=None, classify=None):
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    retrieve = retrieve or retrieve_context_raw
    if classify is None:
        from agents.understanding.classifier import classify_email
        classify = classify_email
    with Path(queries_path).open(encoding="utf-8") as source:
        queries = [json.loads(line) for line in source if line.strip()]
    selected = [q for q in queries if int(q["id"]) % 2 == (1 if split == "dev" else 0)]
    rows = []
    with chroma_at(chroma_path):
        for query in selected:
            text = f"{query.get('subject', '')}\n\n{query.get('body', '')}".strip()
            category = classify(text)
            results = retrieve(text, k=3, source_filter=["gmail_sent"], category=category)
            retrieved = [result["id"] for result in results]
            gold = query.get("gold_message_ids", [])
            hits = [i for i, result_id in enumerate(retrieved, 1) if result_id.removeprefix("gmail_sent_") in gold]
            rows.append({
                "query_type": query["type"],
                "query_id": query["id"],
                "recall_at_3": len(set(gold).intersection(x.removeprefix("gmail_sent_") for x in retrieved)) / len(gold) if gold else 0.0,
                "mrr": 1.0 / hits[0] if hits else 0.0,
                "retrieved_ids": ",".join(x.removeprefix("gmail_sent_") for x in retrieved),
                "gold_message_ids": ",".join(gold),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--chroma-path", type=Path, default=STORE)
    args = parser.parse_args()
    rows = evaluate(args.queries, args.chroma_path, args.split)
    print("query_type\tn\tRecall@3\tMRR")
    for query_type in sorted({row["query_type"] for row in rows}):
        group = [row for row in rows if row["query_type"] == query_type]
        print(f"{query_type}\t{len(group)}\t{sum(r['recall_at_3'] for r in group) / len(group):.4f}\t{sum(r['mrr'] for r in group) / len(group):.4f}")


if __name__ == "__main__":
    main()

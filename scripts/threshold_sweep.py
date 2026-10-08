#!/usr/bin/env python3
"""Threshold sweep on synthetic store for DEV split (odd IDs)."""
import contextlib
import io
import json
import os
import sys
from email.utils import getaddresses
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.reply.retriever import retrieve_context_raw
from agents.understanding.classifier import classify_email
from scripts.load_synthetic import STORE
from scripts.synthetic_chroma import chroma_at

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "data/synthetic/queries.jsonl"


def _normalize_address(raw_address):
    addresses = getaddresses([raw_address or ""])
    return addresses[0][1].strip().lower() if addresses else ""


def run_sweep():
    with Path(QUERIES).open(encoding="utf-8") as source:
        all_queries = [json.loads(line) for line in source if line.strip()]

    # DEV split (odd ids)
    dev_queries = [q for q in all_queries if int(q["id"]) % 2 == 1]

    limits = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for mode_name, person_aware in [("Switch OFF", False), ("Switch ON", True)]:
        os.environ["AGENTMAIL_PERSON_AWARE"] = "1" if person_aware else "0"

        # Pre-fetch retrieval results for all queries in dev split
        retrieved_per_query = []
        with chroma_at(STORE):
            for q in dev_queries:
                text = f"{q.get('subject', '')}\n\n{q.get('body', '')}".strip()
                category = classify_email(text)
                person_addr = _normalize_address(q.get("sender")) if person_aware else None
                thread_id = q.get("thread_id") if person_aware else None

                with contextlib.redirect_stdout(io.StringIO()):
                    results = retrieve_context_raw(
                        text,
                        k=3,
                        source_filter=["gmail_sent"],
                        category=category,
                        person_addr=person_addr,
                        thread_id=thread_id,
                    )
                gold = set(q.get("gold_message_ids", []))
                retrieved_per_query.append({
                    "query_id": q["id"],
                    "results": results,
                    "gold": gold,
                })

        print(f"\n=== {mode_name} (AGENTMAIL_PERSON_AWARE={1 if person_aware else 0}) ===")
        print(f"{'Limit':<8}{'Mean Kept':<15}{'Share Zero Kept':<20}{'Share Gold Kept':<20}")
        print("-" * 63)

        n_queries = len(dev_queries)
        for limit in limits:
            total_kept = 0
            zero_kept_count = 0
            gold_kept_count = 0

            for item in retrieved_per_query:
                results = item["results"]
                gold = item["gold"]

                # keep only results with distance < limit
                kept = [r for r in results if r.get("distance", float("inf")) < limit]
                num_kept = len(kept)
                total_kept += num_kept

                if num_kept == 0:
                    zero_kept_count += 1

                kept_ids = {r["id"].removeprefix("gmail_sent_") for r in kept}
                if gold and bool(kept_ids.intersection(gold)):
                    gold_kept_count += 1

            mean_kept = total_kept / n_queries
            share_zero = zero_kept_count / n_queries
            share_gold = gold_kept_count / n_queries

            print(f"{limit:<8.2f}{mean_kept:<15.4f}{share_zero:<20.4f}{share_gold:<20.4f}")


if __name__ == "__main__":
    run_sweep()

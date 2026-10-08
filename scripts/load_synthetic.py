#!/usr/bin/env python3
"""Load synthetic sent history through the normal sent-ingestion pipeline."""
import argparse
import json
from pathlib import Path

from pipelines.ingestion import ingest_sent_batch
from rag.retrieve import embed_text
from scripts.synthetic_chroma import chroma_at

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data/synthetic/history_threads.jsonl"
STORE = ROOT / "chroma_synthetic"


def load_synthetic(history_path=HISTORY, chroma_path=STORE, *, embed=None, understand=None):
    with Path(history_path).open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    emails = [{
        "message_id": record["message_id"],
        "thread_id": record.get("thread_id", ""),
        "sender_email": record.get("sender", ""),
        "recipient": record.get("recipient", ""),
        "timestamp": record.get("sent_at", ""),
        "subject": record.get("subject", ""),
        "body": record.get("body", ""),
    } for record in records]
    if understand is None:
        from agents.understanding.classifier import classify_email
        understand = lambda text: {"category": classify_email(text)}
    with chroma_at(chroma_path) as collection:
        return ingest_sent_batch(emails, collection, embed=embed or embed_text, understand=understand)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--chroma-path", type=Path, default=STORE)
    args = parser.parse_args()
    print(load_synthetic(args.history, args.chroma_path))


if __name__ == "__main__":
    main()

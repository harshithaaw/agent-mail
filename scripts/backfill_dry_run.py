#!/usr/bin/env python3
"""Preview sent-email metadata backfill without changing Chroma or Gmail."""
import argparse
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmail.fetch import fetch_emails
from pipelines.ingestion import _first_recipient_norm

COLLECTION_NAME = "email_memory"
DEFAULT_STORE = ROOT / "chroma_db"


def run_dry_run(max_emails=30, fetcher=fetch_emails, chroma_path=DEFAULT_STORE, emit=print):
    emails = fetcher(max_emails=max_emails)

    # Refuse to initialize a missing store, then read metadata only; this path
    # never creates a collection and never calls a Chroma write method.
    if not (Path(chroma_path) / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Chroma store does not exist: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(COLLECTION_NAME)
    stored = collection.get(include=["metadatas"])
    stored_by_id = {
        record_id: metadata or {}
        for record_id, metadata in zip(stored.get("ids", []), stored.get("metadatas", []))
    }

    fetched_ids = set()
    already_stored = 0
    for email in emails:
        record_id = f"gmail_sent_{email['message_id']}"
        fetched_ids.add(record_id)
        recipient_norm = _first_recipient_norm(email.get("recipient", ""))
        metadata = stored_by_id.get(record_id)
        exists = metadata is not None
        already_stored += int(exists)
        has_recipient_norm = bool(str((metadata or {}).get("recipient_norm", "")).strip())
        recipient_domain = recipient_norm.rsplit("@", 1)[1] if "@" in recipient_norm else "unknown"
        emit(
            f"id={record_id} exists_in_chroma={'yes' if exists else 'no'} "
            f"has_recipient_norm={'yes' if has_recipient_norm else 'no'} "
            f"recipient_domain={recipient_domain}"
        )

    emit(f"fetched={len(emails)}")
    emit(f"already_stored={already_stored}")
    emit(f"new={len(emails) - already_stored}")
    stored_sent_ids = {
        record_id for record_id, metadata in stored_by_id.items()
        if metadata.get("source") == "gmail_sent"
    }
    emit(f"stored_ids_not_among_fetched={len(stored_sent_ids - fetched_ids)}")
    return {
        "fetched": len(emails),
        "already_stored": already_stored,
        "new": len(emails) - already_stored,
        "stored_ids_not_among_fetched": len(stored_sent_ids - fetched_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=30)
    args = parser.parse_args()
    run_dry_run(max_emails=args.max)


if __name__ == "__main__":
    main()

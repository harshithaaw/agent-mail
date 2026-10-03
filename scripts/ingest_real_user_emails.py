"""Fetch a small Gmail Sent batch, classify it, and upsert it into RAG memory."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.understanding.classifier import classify_email
from gmail.fetch import fetch_emails
from agents.reply.retriever import retrieve_context_raw
from rag.ingest import ingest_real_emails
from rag.retrieve import collection_count, get_or_create_collection
from utils.preprocessing import clean_text


SOURCE = "gmail_sent"
DOCUMENT_TYPE = "sent_email_reply_example"


def prepare_sent_emails(fetched_emails):
    """Normalize and classify fetched Gmail messages before any embedding."""
    prepared = []
    skipped = []
    for email in fetched_emails:
        message_id = email.get("message_id")
        if not message_id:
            skipped.append("missing Gmail message ID")
            continue

        subject = email.get("subject") or ""
        body = email.get("body") or ""
        raw_document = f"Subject: {subject}\n\n{body}" if subject else body
        document_text, _ = clean_text(raw_document)
        if not document_text:
            skipped.append(f"empty cleaned body for message {message_id}")
            continue

        # Existing classifier cleans internally too. Supplying the normalized
        # document here makes classification and embedding use the same text.
        category = str(classify_email(document_text))
        timestamp = email.get("timestamp") or email.get("timestamp_raw")
        prepared.append({
            "gmail_message_id": message_id,
            "message_id": message_id,
            "thread_id": email.get("thread_id"),
            "message_id_rfc": email.get("message_id_rfc"),
            "sender": email.get("sender_email") or None,
            "sender_name": email.get("sender_name") or None,
            "recipients": email.get("recipients") or None,
            "recipient": email.get("recipient") or None,
            "cc": email.get("cc") or None,
            "bcc": email.get("bcc") or None,
            "subject": subject or None,
            "timestamp": timestamp,
            "date": timestamp,
            "timestamp_raw": email.get("timestamp_raw") or None,
            "labels": email.get("labels") or None,
            "category": category,
            "source": SOURCE,
            "type": DOCUMENT_TYPE,
            "document_text": document_text,
        })
    return prepared, skipped


def verify_stored_records(prepared_emails):
    collection = get_or_create_collection()
    ids = [f"gmail_sent_{email['gmail_message_id']}" for email in prepared_emails]
    stored = collection.get(ids=ids, include=["metadatas", "documents", "embeddings"])
    rows = {
        message_id: (metadata, document, embedding)
        for message_id, metadata, document, embedding in zip(
            stored.get("ids", []),
            stored.get("metadatas", []),
            stored.get("documents", []),
            stored.get("embeddings", []),
        )
    }

    checks_by_id = {}
    for email, chroma_id in zip(prepared_emails, ids):
        row = rows.get(chroma_id)
        if row is None:
            checks_by_id[chroma_id] = {"record_found": False}
            continue
        metadata, document, embedding = row
        expected_pairs = {
            "gmail_message_id": email["gmail_message_id"],
            "thread_id": email.get("thread_id"),
            "sender": email.get("sender"),
            "recipients": email.get("recipients"),
            "subject": email.get("subject"),
            "timestamp": email.get("timestamp"),
            "category": email["category"],
            "source": SOURCE,
            "type": DOCUMENT_TYPE,
        }
        metadata_checks = {
            key: (metadata.get(key) == expected if expected is not None else key not in metadata)
            for key, expected in expected_pairs.items()
        }
        metadata_checks["legacy_message_id_matches"] = (
            metadata.get("message_id") == email["gmail_message_id"]
        )
        if email.get("labels"):
            metadata_checks["labels_match"] = metadata.get("labels") == ", ".join(
                str(label) for label in email["labels"]
            )
        checks_by_id[chroma_id] = {
            "record_found": True,
            "document_matches_cleaned_text": document == email["document_text"],
            "embedding_stored": embedding is not None and len(embedding) > 0,
            "embedding_dimensions": len(embedding) if embedding is not None else 0,
            "metadata": metadata_checks,
        }
    return checks_by_id


def run(limit=3):
    emails = fetch_emails(max_emails=limit)
    prepared, skipped = prepare_sent_emails(emails)
    print(f"Fetched Sent emails: {len(emails)}")
    print(f"Classified and ready to ingest: {len(prepared)}")
    print(f"Skipped: {len(skipped)}")
    for reason in skipped:
        print(f"  {reason}")

    if not prepared:
        return {"fetched": len(emails), "ingested": 0, "verified": {}}

    before = collection_count()
    ingest_real_emails(emails=prepared, source=SOURCE, document_type=DOCUMENT_TYPE)
    after_first_ingest = collection_count()

    # Same Gmail message IDs must replace the same Chroma records on rerun.
    ingest_real_emails(emails=prepared, source=SOURCE, document_type=DOCUMENT_TYPE)
    after_rerun = collection_count()

    checks = verify_stored_records(prepared)
    all_checks_pass = all(
        row.get("record_found")
        and row.get("document_matches_cleaned_text")
        and row.get("embedding_stored")
        and all(row.get("metadata", {}).values())
        for row in checks.values()
    )

    # Confirm the existing RAG query returns at least one just-ingested Gmail
    # Sent record; the Reply Agent uses the same source/category filters.
    retrieval = retrieve_context_raw(
        prepared[0]["document_text"],
        k=min(10, max(1, collection_count())),
        source_filter=[SOURCE, "real_user"],
        category=prepared[0]["category"],
    )
    retrieved_ids = {result["id"] for result in retrieval}
    retrieval_ok = any(
        f"gmail_sent_{email['gmail_message_id']}" in retrieved_ids
        for email in prepared
    )

    report = {
        "fetched": len(emails),
        "classified_and_ingested": len(prepared),
        "categories": [email["category"] for email in prepared],
        "unique_deterministic_ids": len({
            email["gmail_message_id"] for email in prepared
        }) == len(prepared),
        "collection_count_before": before,
        "collection_count_after_first_ingest": after_first_ingest,
        "collection_count_after_rerun": after_rerun,
        "rerun_added_no_duplicates": after_rerun == after_first_ingest,
        "record_checks_pass": all_checks_pass,
        "retrieval_found_new_sent_record": retrieval_ok,
        "record_checks": checks,
    }
    print(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify and ingest a small batch of Gmail Sent messages."
    )
    parser.add_argument(
        "--limit", type=int, default=3,
        help="Maximum Sent messages to fetch (1-20; defaults to 3 for safe verification).",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        parser.error("--limit must be between 1 and 20")
    run(args.limit)

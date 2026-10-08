"""Read-only comparison of baseline and person-aware retrieval on unread inbox mail.

This script fetches unread inbox messages but does not mark them read. Chroma is
opened only when an existing database is present, and is queried without writes.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
from email.utils import getaddresses
import io
import os
from pathlib import Path
from typing import Callable, Optional

import rag.retrieve as rag_retrieve
from agents.reply.context import DISTANCE_THRESHOLD


def normalize_address(value: Optional[str]) -> str:
    """Return the first parsed email address as a trimmed lowercase address."""
    addresses = getaddresses([value or ""])
    return addresses[0][1].strip().lower() if addresses else ""


@contextmanager
def _existing_collection(store_path: Optional[str] = None):
    """Bind retrieval to an already-existing Chroma store, restoring globals."""
    path = Path(store_path or rag_retrieve.CHROMA_PATH)
    if not (path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Existing Chroma store not found at {path}")

    previous_client = rag_retrieve._client
    previous_collection = rag_retrieve._collection
    client = rag_retrieve.chromadb.PersistentClient(path=str(path))
    collection = client.get_collection(rag_retrieve.COLLECTION_NAME)
    rag_retrieve._client = client
    rag_retrieve._collection = collection
    try:
        yield collection
    finally:
        rag_retrieve._client = previous_client
        rag_retrieve._collection = previous_collection


def _retrieve_with_mode(text, category, person_addr, thread_id, enabled):
    """Call the production retrieval function with the requested env mode."""
    old_value = os.environ.get("AGENTMAIL_PERSON_AWARE")
    old_present = "AGENTMAIL_PERSON_AWARE" in os.environ
    try:
        if enabled:
            os.environ["AGENTMAIL_PERSON_AWARE"] = "1"
        else:
            os.environ.pop("AGENTMAIL_PERSON_AWARE", None)
        return rag_retrieve.retrieve_similar(
            text,
            k=3,
            source_filter=["gmail_sent"],
            category=category,
            person_addr=person_addr,
            thread_id=thread_id,
        )
    finally:
        if old_present:
            os.environ["AGENTMAIL_PERSON_AWARE"] = old_value
        else:
            os.environ.pop("AGENTMAIL_PERSON_AWARE", None)


def compare_person_aware(
    max_emails: int = 20,
    *,
    fetcher: Optional[Callable] = None,
    classifier: Optional[Callable[[str], str]] = None,
    store_path: Optional[str] = None,
    emit: Callable[[str], None] = print,
):
    """Fetch and compare retrieval; injectable fetch/classifier support tests."""
    if fetcher is None:
        from gmail.fetch import fetch_inbox_emails

        fetcher = fetch_inbox_emails
    if classifier is None:
        from agents.understanding.classifier import classify_email

        classifier = classify_email

    # The inbox fetcher reports progress to stdout. Keep the requested report
    # limited to the safe, address-redacted rows below.
    with redirect_stdout(io.StringIO()):
        emails = fetcher(max_emails=max_emails)

    known_count = 0
    unknown_count = 0
    different_count = 0
    unknown_different_count = 0

    with _existing_collection(store_path) as collection:
        stored = collection.get(where={"source": "gmail_sent"}, include=["metadatas"])
        known_recipients = {
            str(metadata.get("recipient_norm", "")).strip().lower()
            for metadata in (stored.get("metadatas") or [])
            if metadata and metadata.get("recipient_norm")
        }

        for row_number, email in enumerate(emails[:max_emails], start=1):
            person_addr = normalize_address(email.get("sender_email") or email.get("from"))
            domain = person_addr.rsplit("@", 1)[1] if "@" in person_addr else "unknown"
            is_known = bool(person_addr and person_addr in known_recipients)
            known_count += int(is_known)
            unknown_count += int(not is_known)

            thread_id = email.get("thread_id")
            subject = email.get("subject") or ""
            body = email.get("body") or email.get("body_text") or ""
            text = f"{subject}\n\n{body}".strip()
            category = classifier(text)

            off_results = _retrieve_with_mode(text, category, person_addr, thread_id, False)
            on_results = _retrieve_with_mode(text, category, person_addr, thread_id, True)
            off_ids = [result["id"] for result in off_results]
            on_ids = [result["id"] for result in on_results]
            identical = off_ids == on_ids
            different_count += int(not identical)
            unknown_different_count += int(not is_known and not identical)
            above_threshold = sum(
                result.get("distance", 0) > DISTANCE_THRESHOLD for result in on_results
            )
            off_display = ",".join(
                f"{result['id']}:{result['distance']:.4f}" for result in off_results
            )
            on_display = ",".join(
                f"{result['id']}:{result['distance']:.4f}" for result in on_results
            )
            emit(
                f"row={row_number} sender_domain={domain} "
                f"is_known_sender={'yes' if is_known else 'no'} "
                f"thread_id_present={'yes' if thread_id else 'no'} "
                f"off=[{off_display}] on=[{on_display}] "
                f"top3_ids_identical={'yes' if identical else 'no'} "
                f"on_above_threshold={above_threshold}"
            )

    emit(
        f"emails_checked={min(len(emails), max_emails)} known_senders={known_count} "
        f"unknown_senders={unknown_count} rows_top3_differ={different_count} "
        f"unknown_sender_rows_top3_differ={unknown_different_count}"
    )
    return {
        "emails_checked": min(len(emails), max_emails),
        "known_senders": known_count,
        "unknown_senders": unknown_count,
        "rows_top3_differ": different_count,
        "unknown_sender_rows_top3_differ": unknown_different_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=20, dest="max_emails")
    args = parser.parse_args()
    compare_person_aware(max_emails=args.max_emails)


if __name__ == "__main__":
    main()

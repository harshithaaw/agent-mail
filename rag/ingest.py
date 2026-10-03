"""
Seed the flat RAG collection with Enron examples for early testing.
Local-first: uses a real Enron subset if --source is provided, otherwise
falls back to a small bundled sample so the pipeline can be smoke-tested
without requiring the full ~1.7GB corpus.
"""
import argparse
import os
import random
from typing import List, Dict, Any, Optional
from rag.retrieve import add_examples, collection_count
from utils.preprocessing import clean_text

# Small bundled fallback sample - realistic sent-email style text,
# used ONLY if no real Enron source is supplied. Tagged distinctly in
# metadata so it's easy to find/strip once real data is seeded.
FALLBACK_SAMPLE = [
    "Subject: Re: Contract Review\n\nThanks for the draft. I made a few redline edits and attached the revised version. Let's sync tomorrow before we send it to legal.",
    "Subject: Quick question\n\nDo you have the numbers from last week's call? Need them for the report I'm putting together this afternoon.",
    "Subject: Following up\n\nJust circling back on this - any update on your end? Happy to hop on a call if that's easier.",
    "Subject: Meeting notes\n\nAttached are the notes from today's meeting. Key action items are highlighted at the top. Let me know if I missed anything.",
    "Subject: Re: Budget approval\n\nApproved on my end. Go ahead and move forward with the vendor. Let's revisit numbers at the end of Q3.",
]


def load_enron_subset_from_dir(source_dir: str, limit: int = 200):
    """
    Load raw text from a local Enron maildir-style directory.
    Expects plain-text email files (one email per file, headers + body).
    Returns list of (id, text) tuples.

    Sampling logic:
    1. Identify user folders at top level of source_dir
    2. Randomly select 10 users (fixed seed for reproducibility)
    3. For each user, look for sent folders (_sent_mail, sent, sent_items)
    4. Take up to 30 emails per user from sent folders
    5. Stop at --limit total emails or when all selected users exhausted
    """
    # Get user folders (top-level directories in source_dir)
    try:
        user_folders = [f for f in os.listdir(source_dir) 
                       if os.path.isdir(os.path.join(source_dir, f))]
    except Exception as e:
        print(f"Error reading source directory: {e}")
        return []

    if not user_folders:
        print("No user folders found in source directory")
        return []

    # Randomly select 10 users with fixed seed for reproducibility
    random.seed(42)  # Fixed seed for reproducible sampling
    num_users_to_sample = min(10, len(user_folders))
    selected_users = random.sample(user_folders, num_users_to_sample)
    
    print(f"Found {len(user_folders)} total users, selected {num_users_to_sample}: {selected_users}")

    # Sent folder names to look for
    sent_folder_names = ["_sent_mail", "sent", "sent_items"]
    
    examples = []
    count = 0
    per_user_limit = 30  # Max emails per user
    
    for user in selected_users:
        if count >= limit:
            break
            
        user_path = os.path.join(source_dir, user)
        user_email_count = 0
        
        # Find sent folders for this user
        sent_folders = []
        for folder_name in sent_folder_names:
            sent_path = os.path.join(user_path, folder_name)
            if os.path.isdir(sent_path):
                sent_folders.append(sent_path)
        
        if not sent_folders:
            print(f"  {user}: No sent folders found, skipping")
            continue
        
        # Load emails from sent folders
        for sent_folder in sent_folders:
            if count >= limit or user_email_count >= per_user_limit:
                break
                
            for root, _, files in os.walk(sent_folder):
                for fname in files:
                    if count >= limit or user_email_count >= per_user_limit:
                        break
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                        if text and len(text.split()) >= 10:  # skip near-empty files
                            examples.append((f"enron_{count}", text))
                            count += 1
                            user_email_count += 1
                    except Exception:
                        continue
        
        print(f"  {user}: Loaded {user_email_count} emails from sent folders")
    
    if count < limit:
        print(f"Warning: Only collected {count} emails (requested {limit}) from {num_users_to_sample} users")
    
    return examples


def ingest_real_emails(
    emails: List[Dict[str, Any]],
    subject_field: str = "subject",
    body_field: str = "body",
    sender_field: str = "sender",
    date_field: str = "date",
    thread_id_field: str = "thread_id",
    category_field: str = "category",
    source: str = "real_user",
    document_type: str = "sent_example",
) -> None:
    """
    Ingest real user emails into the RAG collection with rich metadata.

    Args:
        emails: List of email dicts containing email data
        subject_field: Key name for subject in email dicts
        body_field: Key name for body in email dicts
        sender_field: Key name for sender in email dicts
        date_field: Key name for date in email dicts
        thread_id_field: Key name for thread_id in email dicts
        category_field: Key name for category in email dicts

    Each email dict should contain at minimum: subject, body, sender, date, thread_id
    Additional fields will be preserved in metadata if present.
    """
    ids = []
    texts = []
    metadatas = []

    for i, email in enumerate(emails):
        # Gmail Sent records use their immutable Gmail message ID, so reruns
        # upsert the same Chroma document instead of making batch duplicates.
        message_id = email.get("gmail_message_id") or email.get("message_id")
        thread_id = email.get(thread_id_field)
        email_source = email.get("source", source)
        email_type = email.get("type", document_type)
        if email_source == "gmail_sent" and not message_id:
            print(f"Skipping sent email at index {i}: Gmail message ID is missing")
            continue
        if email_source == "gmail_sent" and not email.get(category_field):
            print(f"Skipping sent email at index {i}: classified category is missing")
            continue
        email_id = (
            f"gmail_sent_{message_id}"
            if email_source == "gmail_sent"
            else f"real_user_{thread_id}_{i}"
        )

        # Combine subject and body for embedding
        subject = email.get(subject_field, "")
        body = email.get(body_field, "")
        document_text = email.get("document_text")
        if document_text:
            text = document_text
        else:
            if subject and body:
                raw_text = f"Subject: {subject}\n\n{body}"
            elif body:
                raw_text = body
            elif subject:
                raw_text = f"Subject: {subject}"
            else:
                continue
            text, _ = clean_text(raw_text)
            if not text:
                continue

        # Keep existing keys (sender/date/thread_id) for downstream consumers
        # and add explicit Gmail names/provenance for traceability.
        metadata = {"source": email_source, "type": email_type}
        values = {
            "gmail_message_id": message_id,
            "message_id": message_id,
            "thread_id": thread_id,
            "sender": email.get(sender_field),
            "date": email.get(date_field),
            "timestamp": email.get("timestamp", email.get(date_field)),
            "subject": subject,
            "recipients": email.get("recipients"),
            "recipient": email.get("recipient"),
            "cc": email.get("cc"),
            "bcc": email.get("bcc"),
            "mailbox": "sent" if email_source == "gmail_sent" else None,
            "message_id_rfc": email.get("message_id_rfc"),
            "sender_name": email.get("sender_name"),
            "timestamp_raw": email.get("timestamp_raw"),
            "labels": email.get("labels"),
        }
        for key, value in values.items():
            if value is not None and value != "":
                if isinstance(value, (list, tuple, set)):
                    value = ", ".join(str(item) for item in value)
                metadata[key] = str(value)

        # Add category if present
        if category_field in email and email[category_field]:
            metadata["category"] = str(email[category_field])

        # Add any additional fields from the email dict
        for key, value in email.items():
            if key in {
                subject_field, body_field, sender_field, date_field,
                thread_id_field, category_field, "document_text", "source", "type",
                "gmail_message_id", "message_id", "timestamp", "recipients",
                "recipient", "cc", "bcc",
            } or value is None or value == "":
                continue
            if isinstance(value, (list, tuple, set)):
                value = ", ".join(str(item) for item in value)
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        ids.append(email_id)
        texts.append(text)
        metadatas.append(metadata)

    if ids:
        add_examples(
            ids=ids,
            texts=texts,
            metadatas=metadatas,
            upsert=any(item.get("source") == "gmail_sent" for item in metadatas),
        )
        print(f"Ingested {len(ids)} real user emails")
        print(f"Total collection count: {collection_count()}")
    else:
        print("No valid emails to ingest")


def seed(source_dir: str = None, limit: int = 200):
    if source_dir and os.path.isdir(source_dir):
        print(f"Loading Enron subset from: {source_dir}")
        examples = load_enron_subset_from_dir(source_dir, limit=limit)
        source_tag = "enron"
        if not examples:
            print("No usable files found in source directory - falling back to bundled sample.")
            examples = [(f"fallback_{i}", t) for i, t in enumerate(FALLBACK_SAMPLE)]
            source_tag = "fallback_sample"
    else:
        if source_dir:
            print(f"Source directory not found: {source_dir} - using bundled fallback sample.")
        else:
            print("No --source provided - using bundled fallback sample.")
        examples = [(f"fallback_{i}", t) for i, t in enumerate(FALLBACK_SAMPLE)]
        source_tag = "fallback_sample"

    ids = [e[0] for e in examples]
    texts = [e[1] for e in examples]
    metadatas = [{"source": source_tag, "type": "sent_example"} for _ in examples]

    add_examples(ids=ids, texts=texts, metadatas=metadatas)
    print(f"\nSeeding Summary:")
    print(f"  Total examples seeded: {len(examples)} (source={source_tag})")
    print(f"  Total collection count: {collection_count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed flat RAG with Enron or fallback sample data.")
    parser.add_argument("--source", type=str, default=None, help="Path to local Enron maildir-style directory")
    parser.add_argument("--limit", type=int, default=200, help="Max number of emails to load")
    args = parser.parse_args()

    seed(source_dir=args.source, limit=args.limit)

import base64
import json
import quopri
import re
from email.utils import parseaddr, parsedate_to_datetime
from datetime import datetime, timezone
import sys
import os

import html as html_module  # avoid name collision with the `html` parameter

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gmail.auth import get_gmail_service

INBOX_MAX_RESULTS = 25

# Automated sender patterns from agents/reply/gate.py
AUTOMATED_SENDER_PATTERNS = [
    r"no-?reply",
    r"do-?not-?reply",
    r"notifications?",
    r"mailer-daemon",
    r"noreply",
    r"donotreply",
    r"auto-?reply",
    r"bounce",
    r"postmaster",
    r"support@",
    r"info@",
    r"admin@",
]

def is_automated_sender(sender_email: str) -> bool:
    """Check if sender appears to be automated/bulk using gate.py logic."""
    if not sender_email or "@" not in sender_email:
        return False
    
    local_part = sender_email.split("@")[0].lower()
    domain = sender_email.split("@")[1].lower()
    full_lower = sender_email.lower()
    
    for pattern in AUTOMATED_SENDER_PATTERNS:
        if re.search(pattern, full_lower, re.IGNORECASE):
            return True
    
    bulk_domains = [
        "noreply", "no-reply", "donotreply", "do-not-reply",
        "notifications", "mailer", "bounce",
    ]
    
    for bulk_pattern in bulk_domains:
        if bulk_pattern in domain:
            return True
    
    return False


OUTPUT_FILE = "data/local/real_user_sent_emails.json"


def get_header(headers, name):
    """Get a specific header value from a headers list."""
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def decode_body(data, transfer_encoding=""):
    """
    Decode Gmail's base64url-encoded body, then undo quoted-printable
    encoding if required.
    """
    if not data:
        return ""

    try:
        raw = base64.urlsafe_b64decode(
            data + "=" * (-len(data) % 4)
        )
    except (ValueError, TypeError):
        return ""

    if transfer_encoding.lower() == "quoted-printable":
        raw = quopri.decodestring(raw)

    return raw.decode("utf-8", errors="replace")


def strip_html(html):
    """Convert basic HTML email content into readable plain text."""
    html = re.sub(
        r"<br\s*/?>",
        "\n",
        html,
        flags=re.IGNORECASE
    )

    html = re.sub(
        r"</p\s*>",
        "\n",
        html,
        flags=re.IGNORECASE
    )

    html = re.sub(
        r"<[^>]+>",
        "",
        html
    )

    # Decode HTML entities after removing tags.
    html = html_module.unescape(html)

    return html.strip()


def extract_body(payload):
    """
    Recursively search the Gmail payload for text/plain.

    If text/plain is unavailable, use text/html as a fallback.
    Handles quoted-printable encoding per MIME part.
    """
    plain_text = None
    html_text = None

    def walk(part):
        nonlocal plain_text, html_text

        mime_type = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")

        encoding = get_header(
            part.get("headers", []),
            "Content-Transfer-Encoding"
        )

        if body_data:
            if mime_type == "text/plain" and plain_text is None:
                plain_text = decode_body(
                    body_data,
                    encoding
                )

            elif mime_type == "text/html" and html_text is None:
                html_text = decode_body(
                    body_data,
                    encoding
                )

        for child in part.get("parts", []):
            walk(child)

    walk(payload)

    if plain_text:
        return plain_text.strip().replace("\r\n", "\n")

    if html_text:
        return strip_html(html_text).replace("\r\n", "\n")

    return ""


def parse_timestamp(raw_date):
    """Parse the raw Date header into a datetime object."""
    try:
        return parsedate_to_datetime(raw_date)
    except (TypeError, ValueError):
        return None


def _list_message_ids(service, query, max_emails, label="messages"):
    """
    Shared pagination helper: retrieve Gmail message IDs matching `query`,
    up to max_emails, following pageToken pagination.

    `label` is only used for the progress/print output (e.g. "sent",
    "inbox") so callers don't have to duplicate this loop and can't end
    up with print statements that describe the wrong mailbox (this is
    what caused the earlier list_all_inbox_messages / "sent emails"
    mismatch -- one shared implementation instead of copy-pasted ones).
    """
    messages = []
    page_token = None
    page_number = 0

    print(f"\nFetching up to {max_emails} {label} email IDs (query={query!r})...")

    while len(messages) < max_emails:
        page_number += 1

        query_params = {
            "userId": "me",
            "q": query,
            # Gmail allows up to 100 messages per page.
            "maxResults": min(100, max_emails - len(messages)),
        }

        if page_token:
            query_params["pageToken"] = page_token

        result = (
            service.users()
            .messages()
            .list(**query_params)
            .execute()
        )

        batch = result.get("messages", [])

        messages.extend(batch)

        print(
            f"  Page {page_number}: "
            f"{len(batch)} messages "
            f"(total collected: {len(messages)})"
        )

        page_token = result.get("nextPageToken")

        if not page_token or not batch:
            break

    # Gmail should not normally return duplicate IDs,
    # but keeping this makes the ingestion safe.
    unique_messages = []
    seen_ids = set()

    for message in messages:
        message_id = message.get("id")

        if message_id and message_id not in seen_ids:
            seen_ids.add(message_id)
            unique_messages.append(message)

    print(
        f"\nFinished listing {label} emails."
        f"\nUnique {label} message IDs: {len(unique_messages)}"
    )

    return unique_messages


def list_all_sent_messages(service, max_emails=20):
    """
    Retrieve Sent-mailbox Gmail message IDs, up to max_emails.
    Used by fetch_emails() -- sent-mail ingestion for the RAG corpus only.
    """
    return _list_message_ids(service, "in:sent", max_emails, label="sent")


def list_all_inbox_messages(service, max_emails=10):
    """
    Retrieve unread inbox Gmail message IDs, up to max_emails.
    Used by fetch_inbox_emails() -- incoming mail for the
    screening -> Understanding Agent -> Gate -> Reply Agent -> Draft flow.
    """
    return _list_message_ids(service, "in:inbox is:unread", max_emails, label="inbox")


def fetch_full_email(service, message_id):
    """
    Fetch one complete Gmail message and convert it into
    the project's normalized email structure.
    """
    full_message = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="full",
        )
        .execute()
    )

    payload = full_message.get("payload", {})
    headers = payload.get("headers", [])

    raw_from = get_header(headers, "From")
    sender_name, sender_email = parseaddr(raw_from)

    raw_date = get_header(headers, "Date")
    timestamp = parse_timestamp(raw_date)
    internal_date = full_message.get("internalDate")
    try:
        received_at = datetime.fromtimestamp(int(internal_date) / 1000, timezone.utc).isoformat() if internal_date is not None else ""
    except (TypeError, ValueError, OverflowError, OSError):
        received_at = ""

    email = {
        "message_id": full_message.get("id"),
        "thread_id": full_message.get("threadId"),
        "message_id_rfc": get_header(
            headers,
            "Message-ID"
        ),
        "sender_name": sender_name,
        "sender_email": sender_email,
        "subject": get_header(headers, "Subject"),
        "timestamp": (
            timestamp.isoformat()
            if timestamp
            else None
        ),
        "received_at": received_at,
        "timestamp_raw": raw_date,
        "body": extract_body(payload),
        "recipient": get_header(headers, "To"),
        "cc": get_header(headers, "Cc"),
        "bcc": get_header(headers, "Bcc"),
        "recipients": ", ".join(
            value for value in (
                get_header(headers, "To"),
                get_header(headers, "Cc"),
                get_header(headers, "Bcc"),
            ) if value
        ),
        "labels": full_message.get("labelIds", []),
        "list_unsubscribe": get_header(headers, "List-Unsubscribe"),
        "precedence": get_header(headers, "Precedence"),
    }

    return email


def fetch_emails(max_emails=20):
    """
    Fetch sent emails from the authenticated Gmail account.

    SENT-MAIL INGESTION ONLY. This is the RAG-corpus path:
    Sent Gmail -> (Understanding Agent classification, elsewhere) -> ChromaDB.
    It must never be used for incoming/unread mail -- use
    fetch_inbox_emails() for that.

    This function:
    1. Connects to Gmail.
    2. Lists messages in the Sent mailbox (up to max_emails).
    3. Follows all pagination pages.
    4. Fetches each message in full.
    5. Extracts the relevant headers/body.
    6. Filters out automated senders.
    """
    service = get_gmail_service()

    print("Checking Gmail Sent mailbox...")

    sent_count_result = (
        service.users()
        .messages()
        .list(
            userId="me",
            q="in:sent",
            maxResults=1
        )
        .execute()
    )

    sent_estimate = sent_count_result.get(
        "resultSizeEstimate",
        0
    )

    print(
        f"Estimated sent messages according to Gmail: "
        f"{sent_estimate}"
    )

    # --------------------------
    # Get only Sent message IDs.
    # --------------------------

    messages = list_all_sent_messages(service, max_emails=max_emails)

    if not messages:
        print("\nNo sent emails were found.")
        return []

    # --------------------------
    # Fetch complete messages
    # --------------------------

    print(
        f"\nFetching full content for "
        f"{len(messages)} sent emails..."
    )

    emails = []

    for index, message in enumerate(messages, start=1):
        message_id = message.get("id")

        if not message_id:
            print(
                f"  WARNING: message #{index} "
                f"has no ID — skipping."
            )
            continue

        try:
            email = fetch_full_email(
                service,
                message_id
            )

            emails.append(email)

            # Print progress every 10 emails
            # and for the final email.
            if index % 10 == 0 or index == len(messages):
                print(
                    f"  Fetched {index}/{len(messages)} "
                    f"emails"
                )

        except Exception as e:
            print(
                f"  ERROR fetching message "
                f"{index}/{len(messages)} "
                f"(ID={message_id}): {e}"
            )

    # Filter out any automated senders (shouldn't happen in sent folder, but verify)
    filtered_emails = []
    for email in emails:
        sender = email.get("sender_email", "")
        if not is_automated_sender(sender):
            filtered_emails.append(email)
        else:
            print(f"  Filtered out automated sender: {sender}")
    
    print(
        f"\nSuccessfully fetched "
        f"{len(filtered_emails)} / {len(messages)} sent emails "
        f"(filtered {len(emails) - len(filtered_emails)} automated senders)"
    )

    return filtered_emails


def fetch_inbox_emails(max_emails=INBOX_MAX_RESULTS):
    """
    Fetch UNREAD INBOX emails from the authenticated Gmail account.

    INCOMING-MAIL PATH ONLY: these emails are meant to go through
    screening (agents.security.agent.run_security_agent())
    -> Understanding Agent -> Gate -> Reply Agent -> Gmail Draft.
    This function does not touch Sent mail and does not write to
    ChromaDB -- that ingestion path is fetch_emails() above.

    No automated-sender filtering is applied here: that judgment call
    belongs to the gate (gate.py's should_generate_reply /
    is_automated_sender usage downstream), not to fetching.

    Args:
        max_emails: maximum number of unread inbox emails to fetch
                    (defaults to INBOX_MAX_RESULTS).

    Returns:
        List of normalized email dicts (same shape as fetch_emails()).
    """
    service = get_gmail_service()

    print("Checking Gmail account...")

    inbox_count_result = (
        service.users()
        .messages()
        .list(
            userId="me",
            q="in:inbox is:unread",
            maxResults=1
        )
        .execute()
    )

    unread_estimate = inbox_count_result.get("resultSizeEstimate", 0)

    print(f"Estimated unread inbox messages: {unread_estimate}")

    messages = list_all_inbox_messages(service, max_emails=max_emails)

    if not messages:
        print("\nNo unread inbox emails were found.")
        return []

    print(
        f"\nFetching full content for "
        f"{len(messages)} inbox email(s)..."
    )

    emails = []

    for index, message in enumerate(messages, start=1):
        message_id = message.get("id")

        if not message_id:
            print(
                f"  WARNING: message #{index} "
                f"has no ID — skipping."
            )
            continue

        try:
            email = fetch_full_email(service, message_id)
            emails.append(email)

            if index % 10 == 0 or index == len(messages):
                print(f"  Fetched {index}/{len(messages)} emails")

        except Exception as e:
            print(
                f"  ERROR fetching message "
                f"{index}/{len(messages)} "
                f"(ID={message_id}): {e}"
            )

    print(f"\nSuccessfully fetched {len(emails)} / {len(messages)} inbox emails")

    return emails


def save_emails(
    emails,
    path=OUTPUT_FILE
):
    """Persist parsed emails to a flat JSON file."""
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            emails,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved {len(emails)} email(s) to {path}"
    )


def print_sample(emails, count=3):
    """Print a small sample for manual verification."""
    print(
        "\n--- FIRST "
        f"{min(count, len(emails))} EMAILS (SAMPLE) ---"
    )

    for i, email in enumerate(
        emails[:count]
    ):
        print("\n" + "=" * 60)

        print(f"Email {i + 1}:")
        print(
            f"ID:        "
            f"{email.get('message_id')}"
        )
        print(
            f"Thread:    "
            f"{email.get('thread_id')}"
        )
        print(
            f"From:      "
            f"{email.get('sender_name')} "
            f"<{email.get('sender_email')}>"
        )
        print(
            f"Subject:   "
            f"{email.get('subject')}"
        )
        print(
            f"Date:      "
            f"{email.get('timestamp') or email.get('timestamp_raw')}"
        )
        print(
            f"Recipient: "
            f"{email.get('recipient', 'N/A')}"
        )

        print("Body (first 2 lines):")

        body = email.get("body", "")
        body_lines = body.split("\n")[:2]

        for line in body_lines:
            print(f"  {line}")


if __name__ == "__main__":

    emails = fetch_emails()

    print_sample(
        emails,
        count=3
    )

    print(
        f"\n--- TOTAL EMAILS FETCHED: "
        f"{len(emails)} ---"
    )

    save_emails(emails)

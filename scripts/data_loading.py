import email
from email import policy
from email.utils import parseaddr
import mailbox
import os

from gmail.fetch import strip_html
from utils.preprocessing import clean_text


def load_eml_body(filepath):
    with open(filepath, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    plain_text = None
    html_text = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")

            if part.is_multipart():
                continue
            if "attachment" in disposition:
                continue

            if content_type == "text/plain" and plain_text is None:
                plain_text = _decode_part(part)
            elif content_type == "text/html" and html_text is None:
                html_text = _decode_part(part)
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            plain_text = _decode_part(msg)
        elif content_type == "text/html":
            html_text = _decode_part(msg)

    if plain_text:
        return plain_text.strip().replace("\r\n", "\n")
    elif html_text:
        return strip_html(html_text).replace("\r\n", "\n")
    else:
        return ""


def _decode_part(part):
    raw_bytes = part.get_payload(decode=True)
    if raw_bytes is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw_bytes.decode(charset, errors="replace")
    except LookupError:
        return raw_bytes.decode("utf-8", errors="replace")


def _safe_decode(payload, charset):
    """
    Same fallback pattern as _decode_part(), reused here so load_nazario_emails()
    doesn't duplicate decode-error handling across three separate call sites.
    """
    charset = charset or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")

import re

HTML_SNIFF_PATTERN = re.compile(r"<(div|span|p|br|table|tr|td|script|style|a\s|b>|strong|tt|pre|font|img)\b", re.IGNORECASE)


def _looks_like_html(text):
    """
    Detects HTML mistakenly declared as text/plain (a known Nazario corpus issue).
    Looks for actual structural/tag markers rather than a bare '<', since plain
    text legitimately contains '<' (e.g. '<3', 'x < y') without being HTML.
    """
    return bool(HTML_SNIFF_PATTERN.search(text))


def load_nazario_emails(mbox_path):
    """
    Loads Nazario phishing corpus mbox file.
    Returns list[(sender_email, cleaned_text, urls)] for phishing_score() testing.

    KNOWN LIMITATION (documented, not fixed): some messages in this corpus are
    declared text/plain but actually contain raw HTML/CSS/JS, which fragments
    keyword substring matching. An attempted fix — sniffing for HTML and running
    it through strip_html() — was tried and reverted: it dropped Nazario High
    detection from 10.45% to 5.68%, likely because strip_html() discards <a href>
    targets, and phishing_score()'s co-occurrence logic depends on a URL being
    present. Fixing this properly requires an
    href-preserving HTML parse step, which is the same architecture already
    blocked under "KNOWN GAP — phishing.py" (anchor-text-vs-href-target mismatch).
    Deferred as a future improvement — candidate for an ML/DL-based approach
    rather than more heuristic patching.
    """
    mbox = mailbox.mbox(mbox_path)
    examples = []

    for msg in mbox:
        raw_from = msg.get('from', '')
        _, sender_email = parseaddr(raw_from)

        body = None
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' and body is None:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = _safe_decode(payload, part.get_content_charset())
                elif content_type == 'text/html' and body is None:
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_body = _safe_decode(payload, part.get_content_charset())
                        body = strip_html(html_body)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = _safe_decode(payload, msg.get_content_charset())

        if not body:
            continue  # skip empty/unparseable messages — documented known gap

        cleaned, urls = clean_text(body)
        examples.append((sender_email, cleaned, urls))

    return examples
    
def build_training_set(spam_dir="data/spamassassin/spam",
                        ham_dir="data/spamassassin/easy_ham"):
    samples = []

    for filename in os.listdir(spam_dir):
        filepath = os.path.join(spam_dir, filename)
        raw = load_eml_body(filepath)
        cleaned, _ = clean_text(raw)
        samples.append((cleaned, "spam"))

    for filename in os.listdir(ham_dir):
        filepath = os.path.join(ham_dir, filename)
        raw = load_eml_body(filepath)
        cleaned, _ = clean_text(raw)
        samples.append((cleaned, "ham"))

    return samples
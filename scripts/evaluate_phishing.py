# evaluate_phishing.py
import email
from email import policy
from email.utils import parseaddr
import os

from scripts.data_loading import load_nazario_emails
from gmail.fetch import strip_html
from utils.preprocessing import clean_text
from scripts.phishing import phishing_score

NAZARIO_PATH = "nazario_data/phishing3.mbox"
HAM_DIR = "data/spamassassin/easy_ham"


def load_ham_with_sender(ham_dir):
    """
    Loads SpamAssassin ham .eml files with sender + URLs extracted,
    since build_training_set() only returns (cleaned_text, label) and
    phishing_score() needs sender_email + urls too.
    Reuses load_eml_body()'s body-extraction logic path, but reads the
    From header directly since load_eml_body() only returns the body.
    """
    examples = []
    for filename in os.listdir(ham_dir):
        filepath = os.path.join(ham_dir, filename)
        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        raw_from = msg.get("from", "")
        _, sender_email = parseaddr(raw_from)

        body = None
        if msg.is_multipart():
            for part in msg.walk():
                if part.is_multipart():
                    continue
                content_type = part.get_content_type()
                if content_type == "text/plain" and body is None:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                elif content_type == "text/html" and body is None:
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                        body = strip_html(html_body)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")

        if not body:
            continue

        cleaned, urls = clean_text(body)
        examples.append((sender_email, cleaned, urls))

    return examples


def summarize(results, label):
    total = len(results)
    high = sum(1 for r in results if r["risk_level"] == "High")
    medium = sum(1 for r in results if r["risk_level"] == "Medium")
    low = total - high - medium

    print(f"\n--- {label} (n={total}) ---")
    print(f"High:   {high:5d}  ({100*high/total:.2f}%)")
    print(f"Medium: {medium:5d}  ({100*medium/total:.2f}%)")
    print(f"Low:    {low:5d}  ({100*low/total:.2f}%)")
    return high, medium, low, total


def main():
    print("Loading Nazario phishing set...")
    nazario_examples = load_nazario_emails(NAZARIO_PATH)
    print(f"Loaded {len(nazario_examples)} phishing emails.")

    print("Loading SpamAssassin ham set (paired clean/negative)...")
    ham_examples = load_ham_with_sender(HAM_DIR)
    print(f"Loaded {len(ham_examples)} ham emails.")

    phishing_results = []
    for sender_email, cleaned, urls in nazario_examples:
        _, details = phishing_score(cleaned, urls, sender_email)
        phishing_results.append(details)

    ham_results = []
    for sender_email, cleaned, urls in ham_examples:
        _, details = phishing_score(cleaned, urls, sender_email)
        ham_results.append(details)

    p_high, p_med, p_low, p_total = summarize(phishing_results, "Nazario (phishing, positive class)")
    h_high, h_med, h_low, h_total = summarize(ham_results, "SpamAssassin ham (clean, negative class)")

    print("\n=== HEADLINE NUMBERS ===")
    print(f"Detection rate (High on phishing set):     {100*p_high/p_total:.2f}%")
    print(f"False positive rate (High on ham set):     {100*h_high/h_total:.2f}%")
    print(f"Medium-only on phishing set (softer catch): {100*p_med/p_total:.2f}%")
    print(f"Medium-only on ham set (softer false hit):  {100*h_med/h_total:.2f}%")


if __name__ == "__main__":
    main()
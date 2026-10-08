#!/usr/bin/env python3
"""Validator for synthetic test data in data/synthetic/."""

import json
import os
import re
import sys

CONTACTS_FILE = os.path.join("data", "synthetic", "contacts.json")
HISTORY_FILE = os.path.join("data", "synthetic", "history_threads.jsonl")
QUERIES_FILE = os.path.join("data", "synthetic", "queries.jsonl")

# Allowed domains for synthetic data (must end in .example or contain -test. or example.com/org/net)
ALLOWED_SYNTHETIC_DOMAIN_PATTERNS = [
    r".*\.example$",
    r".*-test\..*",
    r".*example\.(com|org|net)$"
]

# Known real commercial or institution email domains that should NEVER appear
DISALLOWED_REAL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "protonmail.com", "aol.com", "live.com", "msn.com", "iiit.ac.in", "hyderabad.ac.in"
}

EXPECTED_TYPE_COUNTS = {
    "same_person_same_topic": 4,
    "same_person_other_topic": 4,
    "same_topic_tiebreak": 4,
    "new_sender": 4,
    "address_variant": 2,
    "thread_continuation": 2,
    "cc_multi": 4,
}

def extract_emails(text):
    """Extract all email addresses from a string."""
    if not isinstance(text, str):
        return []
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(pattern, text)

def is_synthetic_email(email):
    """Check if email address is synthetic (using safe test domains)."""
    domain = email.split("@")[-1].lower()
    if domain in DISALLOWED_REAL_DOMAINS:
        return False
    return any(re.match(pat, domain) for pat in ALLOWED_SYNTHETIC_DOMAIN_PATTERNS)

def validate():
    print("=== SYNTHETIC DATASET VALIDATION REPORT ===")
    errors = []

    # 1. Check file existence
    for path in [CONTACTS_FILE, HISTORY_FILE, QUERIES_FILE]:
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")
            print(f"FAILED: Missing file {path}")
            return False

    # 2. Validate Contacts
    with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
        contacts = json.load(f)

    print(f"Checking contacts.json: total {len(contacts)} contacts found.")
    if len(contacts) != 10:
        errors.append(f"Expected 10 contacts in contacts.json, got {len(contacts)}")

    persona_counts = {"professional": 0, "personal": 0}
    contact_emails = set()
    for c in contacts:
        p = c.get("persona")
        if p in persona_counts:
            persona_counts[p] += 1
        email = c.get("email", "")
        contact_emails.add(email)
        if not is_synthetic_email(email):
            errors.append(f"Non-synthetic contact email found: {email}")

    print(f"  - Professional persona count: {persona_counts['professional']} (expected 5)")
    print(f"  - Personal persona count: {persona_counts['personal']} (expected 5)")
    if persona_counts["professional"] != 5 or persona_counts["personal"] != 5:
        errors.append(f"Incorrect contact persona distribution: {persona_counts}")

    # 3. Validate History Threads
    history_records = []
    history_ids = set()
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            history_records.append(rec)
            msg_id = rec.get("message_id")
            if not msg_id:
                errors.append(f"History record line {line_num} missing message_id")
            elif msg_id in history_ids:
                errors.append(f"Duplicate message_id in history_threads: {msg_id}")
            else:
                history_ids.add(msg_id)

            # Email synthetic check
            for field in ["sender", "recipient"]:
                for email in extract_emails(rec.get(field, "")):
                    if not is_synthetic_email(email):
                        errors.append(f"Non-synthetic email in history {msg_id} ({field}): {email}")

    print(f"Checking history_threads.jsonl: total {len(history_records)} history records found.")
    if len(history_records) != 60:
        errors.append(f"Expected 60 history records, got {len(history_records)}")

    # 4. Validate Queries
    queries = []
    query_ids = set()
    dev_ids = []
    test_ids = []
    type_counts = {}

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            q = json.loads(line)
            queries.append(q)

            q_id = q.get("id")
            if q_id is None:
                errors.append(f"Query record line {line_num} missing id")
                continue

            try:
                numeric_id = int(q_id)
            except ValueError:
                errors.append(f"Query ID must be integer convertable, got: {q_id}")
                numeric_id = 0

            if numeric_id in query_ids:
                errors.append(f"Duplicate query id: {q_id}")
            else:
                query_ids.add(numeric_id)

            # Odd / Even DEV/TEST split check
            if numeric_id % 2 != 0:
                dev_ids.append(numeric_id)
            else:
                test_ids.append(numeric_id)

            # Query type count check
            q_type = q.get("type")
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

            # Check gold_message_ids exist in history
            gold_ids = q.get("gold_message_ids", [])
            for g_id in gold_ids:
                if g_id not in history_ids:
                    errors.append(f"Query {q_id} references non-existent gold_message_id: {g_id}")

            # Check trap_message_ids exist in history
            trap_ids = q.get("trap_message_ids", [])
            for t_id in trap_ids:
                if t_id not in history_ids:
                    errors.append(f"Query {q_id} references non-existent trap_message_id: {t_id}")

            # Email synthetic check
            for field in ["sender", "recipient"]:
                for email in extract_emails(q.get(field, "")):
                    if not is_synthetic_email(email):
                        errors.append(f"Non-synthetic email in query {q_id} ({field}): {email}")

            cc_list = q.get("cc", [])
            if isinstance(cc_list, list):
                for cc_item in cc_list:
                    for email in extract_emails(cc_item):
                        if not is_synthetic_email(email):
                            errors.append(f"Non-synthetic email in query {q_id} (cc): {email}")

    print(f"Checking queries.jsonl: total {len(queries)} queries found.")
    if len(queries) != 24:
        errors.append(f"Expected 24 queries, got {len(queries)}")

    print(f"  - DEV queries (odd IDs count): {len(dev_ids)} (IDs: {sorted(dev_ids)})")
    print(f"  - TEST queries (even IDs count): {len(test_ids)} (IDs: {sorted(test_ids)})")

    if len(dev_ids) != 12:
        errors.append(f"Expected 12 DEV queries (odd IDs), got {len(dev_ids)}")
    if len(test_ids) != 12:
        errors.append(f"Expected 12 TEST queries (even IDs), got {len(test_ids)}")

    print("\nCounts per query type:")
    for t_name, exp_count in EXPECTED_TYPE_COUNTS.items():
        act_count = type_counts.get(t_name, 0)
        status = "OK" if act_count == exp_count else "MISMATCH"
        print(f"  - {t_name}: {act_count} (expected {exp_count}) [{status}]")
        if act_count != exp_count:
            errors.append(f"Type '{t_name}' count mismatch: actual {act_count}, expected {exp_count}")

    # Summary
    print("\n=== SUMMARY ===")
    if not errors:
        print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
        return True
    else:
        print(f"VALIDATION FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  [ERROR] {err}")
        return False

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Read and summarize correspondent metadata from a Chroma store."""
import argparse
from collections import Counter, defaultdict
from email.utils import getaddresses
import json
from pathlib import Path

import chromadb

COLLECTION_NAME = "email_memory"


def _first_address(raw):
    addresses = getaddresses([raw or ""])
    return addresses[0][1].strip().lower() if addresses else ""


def _mask_address(address):
    local, separator, domain = address.partition("@")
    if not separator:
        return address
    return f"{local[:3]}***@{domain}"


def group_people(metadatas):
    people = defaultdict(lambda: {
        "replies": 0,
        "dates": [],
        "categories": Counter(),
        "without_recipient_norm": 0,
    })
    missing_address = 0
    for metadata in metadatas:
        metadata = metadata or {}
        if metadata.get("source") != "gmail_sent":
            continue
        person = str(metadata.get("recipient_norm") or "").strip().lower()
        lacked_norm = not person
        if lacked_norm:
            person = _first_address(metadata.get("recipient", ""))
        if not person:
            missing_address += 1
            continue
        row = people[person]
        row["replies"] += 1
        if lacked_norm:
            row["without_recipient_norm"] += 1
        date = metadata.get("date") or metadata.get("sent_at")
        if date:
            row["dates"].append(str(date))
        row["categories"][str(metadata.get("category") or "unknown")] += 1
    rows = []
    for address in sorted(people):
        data = people[address]
        dates = sorted(data["dates"])
        rows.append({
            "address": address,
            "replies": data["replies"],
            "first_date": dates[0] if dates else "unknown",
            "last_date": dates[-1] if dates else "unknown",
            "category_counts": dict(sorted(data["categories"].items())),
            "without_recipient_norm": data["without_recipient_norm"],
        })
    return rows, missing_address


def show_people(store="chroma_synthetic/", mask=False):
    client = chromadb.PersistentClient(path=str(store))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(include=["metadatas"])
    rows, missing_address = group_people(result.get("metadatas", []))
    print("address\treplies\tfirst_date\tlast_date\tcategory_counts\trecords_without_recipient_norm")
    for row in rows:
        address = _mask_address(row["address"]) if mask else row["address"]
        print(
            f"{address}\t{row['replies']}\t{row['first_date']}\t{row['last_date']}\t"
            f"{json.dumps(row['category_counts'], sort_keys=True)}\t{row['without_recipient_norm']}"
        )
    print(f"records_without_person_address\t{missing_address}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="chroma_synthetic/")
    parser.add_argument("--mask", action="store_true")
    args = parser.parse_args()
    show_people(args.store, args.mask)


if __name__ == "__main__":
    main()

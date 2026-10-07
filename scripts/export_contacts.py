#!/usr/bin/env python3
"""Export a private, editable contacts table from sent-mail metadata."""
import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import chromadb

from scripts.show_people import COLLECTION_NAME, _mask_address, group_people

FIELDS = [
    "address", "replies", "first_date", "last_date", "dominant_category",
    "category_counts", "relationship",
]


def _is_gitignored(path):
    """Check that git currently ignores the destination before writing PII."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _make_rows(metadatas):
    grouped, _ = group_people(metadatas)
    rows = []
    for person in grouped:
        counts = person["category_counts"]
        highest = max(counts.values())
        dominant = sorted(category for category, count in counts.items() if count == highest)[0]
        rows.append({
            "address": person["address"],
            "replies": person["replies"],
            "first_date": person["first_date"],
            "last_date": person["last_date"],
            "dominant_category": dominant,
            "category_counts": json.dumps(counts, sort_keys=True),
            "relationship": "",
        })
    return rows


def export_contacts(store="chroma_db/", out="data/contacts_real.csv", *, ignore_check=None):
    """Read sent metadata and write an ignored CSV, preserving handwritten labels."""
    destination = Path(out)
    check = ignore_check or _is_gitignored
    if not check(destination):
        raise RuntimeError(f"Refusing to write contacts file: {destination} is not gitignored")

    client = chromadb.PersistentClient(path=str(store))
    collection = client.get_collection(COLLECTION_NAME)
    metadata = collection.get(include=["metadatas"]).get("metadatas", [])
    rows = _make_rows(metadata)

    existing = {}
    if destination.exists():
        with destination.open(newline="", encoding="utf-8") as handle:
            for old in csv.DictReader(handle):
                address = (old.get("address") or "").strip().lower()
                relationship = old.get("relationship") or ""
                if address and relationship.strip():
                    existing[address] = relationship
    for row in rows:
        row["relationship"] = existing.get(row["address"], "")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("address\treplies\tfirst_date\tlast_date\tdominant_category\tcategory_counts\trelationship")
    for row in rows:
        # Relationship notes are private hand-entered data; keep them off terminal output.
        print(
            f"{_mask_address(row['address'])}\t{row['replies']}\t{row['first_date']}\t"
            f"{row['last_date']}\t{row['dominant_category']}\t{row['category_counts']}\t"
        )
    print(f"rows\t{len(rows)}")
    print(f"written\t{destination}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="chroma_db/")
    parser.add_argument("--out", default="data/contacts_real.csv")
    args = parser.parse_args()
    try:
        export_contacts(args.store, args.out)
    except RuntimeError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

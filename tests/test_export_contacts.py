import csv
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_contacts import export_contacts


def _temp_store(path):
    collection = chromadb.PersistentClient(path=str(path)).create_collection("email_memory")
    collection.add(
        ids=["a1", "a2", "b", "ignored"],
        embeddings=[[0.0], [0.1], [0.2], [0.3]],
        documents=["ignored"] * 4,
        metadatas=[
            {"source": "gmail_sent", "recipient_norm": "ana@example.test", "date": "2026-01-02", "category": "Academic"},
            {"source": "gmail_sent", "recipient": "Ananya <ANA@example.test>", "date": "2026-02-03", "category": "Academic"},
            {"source": "gmail_sent", "recipient_norm": "bob@example.test", "sent_at": "2026-03-04", "category": "Career"},
            {"source": "inbox", "recipient_norm": "ignored@example.test", "date": "2026-01-01", "category": "Other"},
        ],
    )


def test_export_groups_and_preserves_relationship_and_masks_terminal(tmp_path, capsys):
    store = tmp_path / "chroma"
    out = tmp_path / "contacts.csv"
    _temp_store(store)
    ignore = lambda _path: True

    rows = export_contacts(store, out, ignore_check=ignore)
    assert rows[0] == {
        "address": "ana@example.test", "replies": 2,
        "first_date": "2026-01-02", "last_date": "2026-02-03",
        "dominant_category": "Academic",
        "category_counts": '{"Academic": 2}', "relationship": "",
    }
    assert len(rows) == 2

    with out.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    persisted[0]["relationship"] = "colleague"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=persisted[0].keys())
        writer.writeheader()
        writer.writerows(persisted)

    export_contacts(store, out, ignore_check=ignore)
    terminal = capsys.readouterr().out
    with out.open(newline="", encoding="utf-8") as handle:
        rerun = list(csv.DictReader(handle))
    assert rerun[0]["relationship"] == "colleague"
    assert "ana@example.test" not in terminal
    assert "bob@example.test" not in terminal
    assert "ana***@example.test" in terminal
    assert "rows\t2" in terminal
    assert f"written\t{out}" in terminal


def test_export_refuses_nonignored_path_before_reading_chroma(tmp_path):
    out = tmp_path / "contacts.csv"
    try:
        export_contacts(tmp_path / "does-not-exist", out, ignore_check=lambda _path: False)
    except RuntimeError as error:
        assert "not gitignored" in str(error)
    else:
        raise AssertionError("expected refusal")
    assert not out.exists()

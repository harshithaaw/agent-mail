import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backfill_dry_run import run_dry_run


def test_backfill_dry_run_uses_temp_chroma_and_fake_fetch(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection("email_memory")
    collection.add(
        ids=["gmail_sent_existing", "gmail_sent_missing_norm", "gmail_sent_stale", "real_user_unrelated", "enron_unrelated"],
        embeddings=[[0.0], [0.1], [0.2], [0.3], [0.4]],
        documents=["one", "two", "stale", "unrelated", "enron"],
        metadatas=[
            {"source": "gmail_sent", "recipient_norm": "one@example.test"},
            {"source": "gmail_sent"},
            {"source": "gmail_sent", "recipient_norm": "stale@example.test"},
            {"source": "real_user", "recipient_norm": "unrelated@example.test"},
            {"source": "enron"},
        ],
    )
    fetched = []

    def fake_fetch(max_emails):
        fetched.append(max_emails)
        return [
            {"message_id": "existing", "recipient": "One <ONE@example.test>"},
            {"message_id": "missing_norm", "recipient": "Two <TWO@example.test>"},
            {"message_id": "new", "recipient": "Three <THREE@example.test>"},
        ]

    output = []
    totals = run_dry_run(7, fake_fetch, tmp_path / "chroma", output.append)

    assert fetched == [7]
    assert totals == {
        "fetched": 3,
        "already_stored": 2,
        "new": 1,
        "stored_ids_not_among_fetched": 1,
    }
    assert output[:3] == [
        "id=gmail_sent_existing exists_in_chroma=yes has_recipient_norm=yes recipient_domain=example.test",
        "id=gmail_sent_missing_norm exists_in_chroma=yes has_recipient_norm=no recipient_domain=example.test",
        "id=gmail_sent_new exists_in_chroma=no has_recipient_norm=no recipient_domain=example.test",
    ]
    assert output[-1] == "stored_ids_not_among_fetched=1"
    assert all("ONE@" not in line and "TWO@" not in line and "THREE@" not in line for line in output)

import chromadb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.show_people import group_people, show_people


def test_show_people_reads_temp_chroma_and_groups_fallback_recipient(tmp_path, capsys):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection("email_memory")
    collection.add(
        ids=["normalized", "fallback", "missing", "other-source"],
        embeddings=[[0.0], [0.1], [0.2], [0.3]],
        documents=["ignored"] * 4,
        metadatas=[
            {"source": "gmail_sent", "recipient_norm": "ana@example.test", "date": "2026-01-02", "category": "Academic"},
            {"source": "gmail_sent", "recipient": "Ananya <ANA@example.test>", "date": "2026-02-03", "category": "Career"},
            {"source": "gmail_sent", "date": "2026-03-04", "category": "Personal"},
            {"source": "enron", "recipient_norm": "ignored@example.test", "date": "2026-01-01", "category": "Career"},
        ],
    )

    rows, missing_address = group_people(collection.get(include=["metadatas"])["metadatas"])
    assert rows == [{
        "address": "ana@example.test",
        "replies": 2,
        "first_date": "2026-01-02",
        "last_date": "2026-02-03",
        "category_counts": {"Academic": 1, "Career": 1},
        "without_recipient_norm": 1,
    }]
    assert missing_address == 1

    show_people(tmp_path / "chroma", mask=True)
    output = capsys.readouterr().out
    assert "ana***@example.test\t2\t2026-01-02\t2026-02-03" in output
    assert "records_without_person_address\t1" in output

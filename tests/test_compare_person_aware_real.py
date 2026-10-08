import sys
from pathlib import Path

import chromadb
import sentence_transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _NoNetworkEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, **kwargs):
        return [1.0, 0.0]


sentence_transformers.SentenceTransformer = _NoNetworkEmbedder

import rag.retrieve as retrieve
from scripts.compare_person_aware_real import compare_person_aware


def test_comparison_known_sender_reranks_and_unknown_sender_stays_same(tmp_path, monkeypatch):
    store = tmp_path / "temporary-chroma"
    client = chromadb.PersistentClient(path=str(store))
    collection = client.create_collection("email_memory", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=["near-1", "near-2", "near-3", "known-person"],
        documents=["near one", "near two", "near three", "matching correspondent"],
        embeddings=[
            [1.0, 0.0], [0.9999, 0.014142], [0.9998, 0.019999], [0.995, 0.099875],
        ],
        metadatas=[
            {"source": "gmail_sent", "recipient_norm": "other1@example.test", "category": "Academic"},
            {"source": "gmail_sent", "recipient_norm": "other2@example.test", "category": "Academic"},
            {"source": "gmail_sent", "recipient_norm": "other3@example.test", "category": "Academic"},
            {"source": "gmail_sent", "recipient_norm": "known@example.test", "thread": "thread-known", "category": "Academic"},
        ],
    )
    monkeypatch.setattr(retrieve, "CHROMA_PATH", str(store))
    monkeypatch.setattr(retrieve, "_client", None)
    monkeypatch.setattr(retrieve, "_collection", None)
    monkeypatch.setattr(retrieve, "embed_text", lambda text: [1.0, 0.0])
    monkeypatch.delenv("AGENTMAIL_PERSON_AWARE", raising=False)

    fake_emails = [
        {"sender_email": "Known Person <known@example.test>", "thread_id": "thread-known",
         "subject": "Known topic", "body": "Question about the topic."},
        {"sender_email": "Stranger <stranger@example.test>", "thread_id": "thread-unknown",
         "subject": "Other topic", "body": "Another question."},
    ]
    output = []
    totals = compare_person_aware(
        20,
        fetcher=lambda max_emails: fake_emails[:max_emails],
        classifier=lambda text: "Academic",
        store_path=str(store),
        emit=output.append,
    )

    assert totals["known_senders"] == 1
    assert totals["unknown_senders"] == 1
    assert totals["unknown_sender_rows_top3_differ"] == 0
    assert "on=[known-person:" in output[0]
    assert "top3_ids_identical=no" in output[0]
    assert "top3_ids_identical=yes" in output[1]
    joined = "\n".join(output)
    assert "known@example.test" not in joined
    assert "stranger@example.test" not in joined
    assert "Known Person" not in joined
    assert "Stranger" not in joined

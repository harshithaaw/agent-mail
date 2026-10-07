import sentence_transformers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _NoNetworkEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, **kwargs):
        return [1.0, 0.0]


# The retrieval test supplies deterministic vectors and must not load model files.
sentence_transformers.SentenceTransformer = _NoNetworkEmbedder

import rag.retrieve as retrieve


def test_person_aware_switch_off_and_on_uses_temp_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieve, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(retrieve, "_client", None)
    monkeypatch.setattr(retrieve, "_collection", None)
    monkeypatch.setattr(retrieve, "embed_text", lambda text: [1.0, 0.0])
    collection = retrieve.get_or_create_collection()

    collection.add(
        ids=["near-1", "near-2", "near-3", "person-thread-match"],
        documents=["near one", "near two", "near three", "same correspondent and thread"],
        embeddings=[
            [1.0, 0.0],
            [0.9999, 0.014142],
            [0.9998, 0.019999],
            [0.995, 0.099875],
        ],
        metadatas=[
            {"source": "gmail_sent", "recipient_norm": "other1@example.test", "thread": "other-1"},
            {"source": "gmail_sent", "recipient_norm": "other2@example.test", "thread": "other-2"},
            {"source": "gmail_sent", "recipient_norm": "other3@example.test", "thread": "other-3"},
            {"source": "gmail_sent", "recipient_norm": "person@example.test", "thread": "thread-123"},
        ],
    )

    monkeypatch.delenv("AGENTMAIL_PERSON_AWARE", raising=False)
    baseline = retrieve.retrieve_similar(
        "query", k=3, source_filter=["gmail_sent"],
        person_addr="person@example.test", thread_id="thread-123",
    )
    assert [item["id"] for item in baseline] == ["near-1", "near-2", "near-3"]

    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    personalized = retrieve.retrieve_similar(
        "query", k=3, source_filter=["gmail_sent"],
        person_addr="person@example.test", thread_id="thread-123",
    )
    assert len(personalized) <= 3
    assert personalized[0]["id"] == "person-thread-match"
    assert set(personalized[0]) == {"id", "document", "metadata", "distance"}
    assert personalized[0]["distance"] == collection.query(
        query_embeddings=[[1.0, 0.0]], n_results=3,
        where={"$and": [
            {"source": "gmail_sent"},
            {"recipient_norm": "person@example.test"},
        ]},
    )["distances"][0][0]
    baseline_distances = {item["id"]: item["distance"] for item in baseline}
    assert all(item["distance"] == baseline_distances[item["id"]]
               for item in personalized if item["id"] in baseline_distances)

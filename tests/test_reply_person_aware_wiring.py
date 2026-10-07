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

import pipelines.reply as reply_pipeline
import rag.retrieve as retrieve_module


def _collection(path):
    collection = chromadb.PersistentClient(path=str(path)).create_collection("email_memory")
    collection.add(
        ids=["person-example"],
        embeddings=[[1.0, 0.0]],
        documents=["Past reply"],
        metadatas=[{
            "source": "gmail_sent",
            "category": "Academic",
            "recipient_norm": "ada@example.test",
            "thread": "thread-1",
        }],
    )
    return collection


def _reply(email, collection, embed=lambda _text: [1.0, 0.0]):
    return reply_pipeline.reply_to_email(
        email,
        {"category": "Academic"},
        collection,
        embed,
        generate=lambda _email, _examples: "Draft reply",
    )


def test_switch_off_uses_retrieve_examples_only(tmp_path, monkeypatch):
    collection = _collection(tmp_path / "off-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "0")
    calls = []
    monkeypatch.setattr(
        reply_pipeline, "retrieve_examples",
        lambda *args: calls.append(args) or [{"id": "old-path"}],
    )
    monkeypatch.setattr(
        retrieve_module, "retrieve_similar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected person-aware call")),
    )

    result = _reply({"subject": "hello", "body": "body"}, collection)

    assert result["retrieved"] == [{"id": "old-path"}]
    assert len(calls) == 1
    assert calls[0][1:3] == ("Academic", collection)
    assert calls[0][3] is _reply.__defaults__[0]
    assert calls[0][4] == 3


def test_switch_on_passes_normalized_person_thread_collection_and_embed(tmp_path, monkeypatch):
    collection = _collection(tmp_path / "on-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    embed_calls = []

    def embed(text):
        embed_calls.append(text)
        return [1.0, 0.0]

    original = retrieve_module.retrieve_similar
    received = {}

    def spy(query_text, **kwargs):
        received.update(query_text=query_text, **kwargs)
        return original(query_text, **kwargs)

    monkeypatch.setattr(retrieve_module, "retrieve_similar", spy)
    monkeypatch.setattr(
        reply_pipeline, "retrieve_examples",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )

    result = _reply({
        "sender_email": "Ada Example <ADA@Example.Test>",
        "thread_id": "thread-1",
        "subject": "hello",
        "body": "body",
    }, collection, embed)

    assert received["person_addr"] == "ada@example.test"
    assert received["thread_id"] == "thread-1"
    assert received["collection"] is collection
    assert received["embed"] is embed
    assert received["source_filter"] == ["gmail_sent"]
    assert embed_calls == ["Subject: hello\n\nbody"]
    assert result["retrieved"][0]["id"] == "person-example"
    assert set(result["retrieved"][0]) == {"id", "document", "metadata", "distance"}


def test_switch_on_without_sender_or_thread_uses_old_path(tmp_path, monkeypatch):
    collection = _collection(tmp_path / "missing-person-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    calls = []
    monkeypatch.setattr(
        reply_pipeline, "retrieve_examples",
        lambda *args: calls.append(args) or [{"id": "old-path"}],
    )
    monkeypatch.setattr(
        retrieve_module, "retrieve_similar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected person-aware call")),
    )

    result = _reply({"subject": "hello", "body": "body"}, collection)

    assert result["retrieved"] == [{"id": "old-path"}]
    assert len(calls) == 1


def test_person_aware_failure_falls_back_to_old_path(tmp_path, monkeypatch, caplog):
    collection = _collection(tmp_path / "fallback-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    calls = []

    def fail_person_aware(*args, **kwargs):
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(retrieve_module, "retrieve_similar", fail_person_aware)
    monkeypatch.setattr(
        reply_pipeline, "retrieve_examples",
        lambda *args: calls.append(args) or [{"id": "fallback"}],
    )

    result = _reply({"sender_email": "Ada@Example.Test", "subject": "hello", "body": "body"}, collection)

    assert result["retrieved"] == [{"id": "fallback"}]
    assert len(calls) == 1
    assert "falling back to category retrieval" in caplog.text


def test_retrieve_similar_uses_injected_collection_and_embed_for_person_lookup(tmp_path, monkeypatch):
    collection = chromadb.PersistentClient(path=str(tmp_path / "end-to-end-chroma")).create_collection(
        "email_memory", metadata={"hnsw:space": "cosine"}
    )
    vectors = [
        [1.0, 0.0],
        [0.99, (1.0 - 0.99 ** 2) ** 0.5],
        [0.97, (1.0 - 0.97 ** 2) ** 0.5],
        [0.945, (1.0 - 0.945 ** 2) ** 0.5],
    ]
    collection.add(
        ids=["near-1", "near-2", "near-3", "person-reply"],
        embeddings=vectors,
        documents=["Near reply 1", "Near reply 2", "Near reply 3", "Reply to person"],
        metadatas=[
            {"source": "gmail_sent", "recipient_norm": address}
            for address in ["other-1@example.com", "other-2@example.com",
                            "other-3@example.com", "person@example.com"]
        ],
    )

    def fail_if_internal_store_or_embed_is_used(*args, **kwargs):
        raise AssertionError("retrieve_similar used its internal store or embedder")

    monkeypatch.setattr(retrieve_module, "get_or_create_collection", fail_if_internal_store_or_embed_is_used)
    monkeypatch.setattr(retrieve_module, "embed_text", fail_if_internal_store_or_embed_is_used)
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")

    def fixed_embed(_query):
        return [1.0, 0.0]

    results_on = retrieve_module.retrieve_similar(
        "query", k=3, source_filter=["gmail_sent"],
        person_addr="person@example.com", collection=collection, embed=fixed_embed,
    )
    person_result = next(result for result in results_on if result["id"] == "person-reply")
    assert person_result["distance"] == collection.query(
        query_embeddings=[[1.0, 0.0]],
        n_results=1,
        where={"recipient_norm": "person@example.com"},
        include=["distances"],
    )["distances"][0][0]

    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "0")
    results_off = retrieve_module.retrieve_similar(
        "query", k=3, source_filter=["gmail_sent"],
        person_addr="person@example.com", collection=collection, embed=fixed_embed,
    )
    assert "person-reply" not in [result["id"] for result in results_off]

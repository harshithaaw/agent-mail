import sys
from pathlib import Path

import chromadb
import pytest
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


def _draft_examples_call(email, collection, generate):
    return reply_pipeline.reply_to_email(
        email,
        {"category": "Academic"},
        collection,
        lambda _text: [1.0, 0.0],
        generate=generate,
    )


def _examples_at_distances(distances):
    return [
        {
            "id": f"example-{index}",
            "document": f"Example {index}",
            "metadata": {"category": "Academic"},
            "distance": distance,
        }
        for index, distance in enumerate(distances)
    ]


def test_person_aware_drafts_only_with_examples_below_threshold(tmp_path, monkeypatch):
    collection = _collection(tmp_path / "validation-mixed-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    rows = _examples_at_distances([0.42, 0.71])
    monkeypatch.setattr(retrieve_module, "retrieve_similar", lambda *args, **kwargs: rows)
    monkeypatch.setattr(
        reply_pipeline, "retrieve_examples",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )
    received = []

    result = _draft_examples_call(
        {"sender_email": "person@example.com"},
        collection,
        lambda _email, examples: received.append(examples) or "Draft reply",
    )

    assert received == [[rows[0]]]
    assert result["draft"] == "Draft reply"
    assert result["retrieved"] == rows
    assert [example["distance"] for example in result["retrieved"]] == [0.42, 0.71]


def test_person_aware_drafts_without_context_when_all_examples_are_above_threshold(tmp_path, monkeypatch):
    collection = _collection(tmp_path / "validation-empty-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    rows = _examples_at_distances([0.62, 0.83])
    monkeypatch.setattr(retrieve_module, "retrieve_similar", lambda *args, **kwargs: rows)
    received = []

    result = _draft_examples_call(
        {"sender_email": "person@example.com"},
        collection,
        lambda _email, examples: received.append(examples) or "Draft without context",
    )

    assert received == [[]]
    assert result["draft"] == "Draft without context"
    assert result["retrieved"] == rows


def test_person_aware_drops_example_at_exact_threshold(tmp_path, monkeypatch):
    collection = _collection(tmp_path / "validation-boundary-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    rows = _examples_at_distances([0.60])
    monkeypatch.setattr(retrieve_module, "retrieve_similar", lambda *args, **kwargs: rows)
    received = []

    result = _draft_examples_call(
        {"sender_email": "person@example.com"},
        collection,
        lambda _email, examples: received.append(examples) or "Draft without boundary example",
    )

    assert received == [[]]
    assert result["retrieved"] == rows


def test_switch_off_never_validates_and_passes_old_examples_unchanged(tmp_path, monkeypatch):
    import agents.reply.context as reply_context

    collection = _collection(tmp_path / "validation-off-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "0")
    rows = _examples_at_distances([0.42, 0.71])
    monkeypatch.setattr(reply_pipeline, "retrieve_examples", lambda *args: rows)
    monkeypatch.setattr(
        retrieve_module, "retrieve_similar",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected person-aware call")),
    )
    monkeypatch.setattr(
        reply_context, "validate_context",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected validation")),
    )
    received = []

    result = _draft_examples_call(
        {"sender_email": "person@example.com"},
        collection,
        lambda _email, examples: received.append(examples) or "Draft reply",
    )

    assert received == [rows]
    assert result["retrieved"] == rows


def test_person_aware_failure_and_missing_person_context_skip_validation(tmp_path, monkeypatch):
    import agents.reply.context as reply_context

    collection = _collection(tmp_path / "validation-fallback-chroma")
    monkeypatch.setenv("AGENTMAIL_PERSON_AWARE", "1")
    rows = _examples_at_distances([0.42, 0.71])
    retrieve_calls = []
    fallback_calls = []

    def fail_retrieval(*args, **kwargs):
        retrieve_calls.append((args, kwargs))
        raise RuntimeError("person-aware retrieval failed")

    monkeypatch.setattr(retrieve_module, "retrieve_similar", fail_retrieval)
    monkeypatch.setattr(
        reply_pipeline, "retrieve_examples",
        lambda *args: fallback_calls.append(args) or rows,
    )
    monkeypatch.setattr(
        reply_context, "validate_context",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected validation")),
    )
    received = []
    generate = lambda _email, examples: received.append(examples) or "Draft reply"

    with_sender = _draft_examples_call({"sender_email": "person@example.com"}, collection, generate)
    without_sender_or_thread = _draft_examples_call({}, collection, generate)

    assert len(retrieve_calls) == 1
    assert len(fallback_calls) == 2
    assert received == [rows, rows]
    assert with_sender["retrieved"] == rows
    assert without_sender_or_thread["retrieved"] == rows


def test_build_generation_prompt_accepts_empty_examples_without_model_call():
    try:
        from agents.reply.generate import build_generation_prompt
    except Exception as error:
        pytest.skip(f"generation module import unavailable without network/model setup: {error}")

    prompt = build_generation_prompt("Incoming email", {"category": "Academic"}, [])

    assert "(none -- write the reply without relying on past examples)" in prompt

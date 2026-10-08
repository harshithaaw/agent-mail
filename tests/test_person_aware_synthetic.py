import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sentence_transformers


class _TestEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, **kwargs):
        return [1.0, 0.0, 0.0]


# Keep test collection independent of local model caches and network access.
sentence_transformers.SentenceTransformer = _TestEmbedder

import scripts.eval_person_aware as evaluator
import scripts.load_synthetic as loader
import rag.retrieve as retrieve_module


def test_synthetic_load_is_idempotent_and_eval_uses_isolated_store(tmp_path, monkeypatch):
    history = tmp_path / "history.jsonl"
    history.write_text("\n".join(json.dumps(item) for item in [
        {"message_id": "m1", "thread_id": "t1", "sender": "Owner <owner@example.test>",
         "recipient": "First Person <FIRST@example.test>, second@example.test", "sent_at": "today",
         "subject": "Research update", "body": "The research results are ready."},
        {"message_id": "m2", "thread_id": "t2", "sender": "Owner <owner@example.test>",
         "recipient": "second@example.test", "sent_at": "today", "subject": "Other research",
         "body": "Another research note."},
    ]) + "\n", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({
        "id": 1, "type": "same_person_same_topic", "subject": "Research update",
        "body": "Could you send the research results?", "gold_message_ids": ["m1"],
    }) + "\n", encoding="utf-8")
    store = tmp_path / "isolated-chroma"
    monkeypatch.setattr(loader, "embed_text", lambda text: [1.0, 0.0, 0.0])
    monkeypatch.setattr(retrieve_module, "embed_text", lambda text: [1.0, 0.0, 0.0])
    classify = lambda text: "Academic"

    first = loader.load_synthetic(history, store, understand=lambda text: {"category": classify(text)})
    second = loader.load_synthetic(history, store, understand=lambda text: {"category": classify(text)})
    assert first["after"] == second["after"] == 2
    assert second["before"] == 2
    with loader.chroma_at(store) as collection:
        metadata = collection.get(ids=["gmail_sent_m1"], include=["metadatas"])["metadatas"][0]
    assert metadata["recipient_norm"] == "first@example.test"
    rows = evaluator.evaluate(queries, store, "dev", classify=classify)
    assert len(rows) == 1
    assert rows[0]["recall_at_3"] == 1.0
    assert rows[0]["mrr"] > 0

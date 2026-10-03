import json
from datetime import datetime, timezone
import app
import pytest
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from agents.understanding.categories import CATEGORIES
from agents.understanding.classifier import classify_email, clf, vectorizer
from app import process_inbox
from pipelines.ingestion import ingest_sent_batch
from pipelines.reply import reply_to_email

ROOT = Path(__file__).resolve().parents[1]


_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)


# Skip tests that require trained models if they're not available
models_available = clf is not None and vectorizer is not None
models_skip_reason = (
    "Required category classifier models not found. "
    "Rebuild with: python -m scripts.train_category_classifier"
)
skip_if_no_models = pytest.mark.skipif(
    not models_available,
    reason=models_skip_reason
)


def real_embedding(text):
    return _embedder.encode(text, normalize_embeddings=True).tolist()


def fixtures(name):
    return json.loads((ROOT / "data/fixtures" / name).read_text())


@skip_if_no_models
def test_pipeline_b_idempotent_and_metadata_roundtrip(caplog):
    sent = fixtures("synthetic_sent_emails.json")
    client = chromadb.Client()
    collection = client.create_collection("stage_b")
    assert len(sent) >= 20
    # The real category classifier is the exact function registered as the
    # Understanding Agent's `classify` tool; no fixture categorizer is used.
    classify = lambda text: {"category": classify_email(text)}
    first = ingest_sent_batch(sent, collection, real_embedding, understand=classify)
    assert first["after"] == len(sent)
    assert collection.count() == len(sent)
    # ingest_sent_batch compares metadata for every fixture immediately after upsert.
    categories = {m["category"] for m in collection.get(include=["metadatas"])["metadatas"]}
    assert categories.issubset(set(CATEGORIES)) and len(categories) >= 3
    second = ingest_sent_batch(sent, collection, real_embedding, understand=classify)
    assert second["before"] == second["after"] == len(sent)


@skip_if_no_models
def test_pipeline_a_retrieval_for_ten_fixture_inbox_messages():
    sent = fixtures("synthetic_sent_emails.json")
    collection = chromadb.Client().create_collection("stage_a")
    classify = lambda text: {"category": classify_email(text)}
    ingest_sent_batch(sent, collection, real_embedding, understand=classify)
    inbox = fixtures("inbox_10.json")
    assert len(inbox) >= 10
    for email in inbox:
        text = f"{email['subject']}\n{email['body']}"
        understood = {"category": classify_email(text)}
        result = reply_to_email(email, understood, collection, real_embedding,
                                generate=lambda current, examples: "Fixture grounded reply: " + current["subject"], k=3)
        assert result["draft"] and len(result["retrieved"]) > 0
        assert all(x["metadata"]["category"] == understood["category"] for x in result["retrieved"])


def test_pipeline_a_routes_only_clean_to_draft(tmp_path):
    emails = fixtures("test_all_routes.json")
    routes = {
        "test_1_clean": "clean_full_pipeline",
        "test_2_flagged_injection": "flagged_skip_reply_generation",
        "test_3_flagged_phishing_high": "flagged_skip_reply_generation",
        "test_4_flagged_phishing_medium": "flagged_classify_summarize_only",
        "test_5_low_priority_spam": "low_priority_skip_downstream",
        "test_6_low_priority_safe": "clean_full_pipeline",
    }
    def security(text, sender, urls):
        # Fixture route labels exercise app routing independent of live models.
        current = next(e for e in emails if e["sender_email"] == sender and e["body"] in text)
        return {"route": routes[current["message_id"]]}
    def fetcher(max_emails):
        return [{**e, "thread_id": e["message_id"], "timestamp": "2026-10-01T00:00:00+00:00"} for e in emails]
    outputs = process_inbox(fetcher=fetcher, security=security,
        understanding=lambda text: {"category": "Personal", "summary": text[:40]},
        drafter=lambda email, result: "Fixture draft based on: " + email["subject"],
        gate=lambda route, sender, *headers: True,
        processed_db=tmp_path / "processed.db", now=datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert len(outputs) == len(emails)
    for output in outputs:
        assert bool(output["draft"]) == (output["route"] == "clean_full_pipeline")
        assert output["status"] == ("draft_created" if output["draft"] else output["route"])


def test_empty_reply_is_visible_as_generation_failed_and_never_calls_gmail(monkeypatch, capsys, tmp_path):
    email = {
        "message_id": "fixture_bad_generation",
        "thread_id": "fixture_thread",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sender_email": "person@example.org",
        "subject": "Can you review this?",
        "body": "Please let me know what you think about the attached proposal.",
    }
    monkeypatch.setattr(app, "reply_to_email", lambda *args, **kwargs: {
        "draft": "", "retrieved": [{"id": "example-1"}],
    })
    monkeypatch.setattr("gmail.draft.create_gmail_draft", lambda *args, **kwargs: pytest.fail("must not call Gmail for empty generation"))

    result = app.process_inbox(
        fetcher=lambda max_emails: [email],
        security=lambda *args: {"route": "clean_full_pipeline"},
        understanding=lambda text: {"category": "Personal"},
        gate=lambda route, sender, *headers: True,
        collection=object(), embed=lambda text: [0.0],
        processed_db=tmp_path / "processed.db", now=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )[0]

    assert result["status"] == "generation_failed"
    assert result["draft"] is None
    assert result["reply_result"]["retrieved"][0]["id"] == "example-1"
    assert "status=generation_failed" in capsys.readouterr().out

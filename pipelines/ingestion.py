"""Pipeline B: sent fixtures/Gmail boundary -> shared understanding -> Chroma."""
from __future__ import annotations
import logging
from typing import Callable

from agents.understanding.categories import CATEGORIES
from agents.understanding.classifier import classify_email

log = logging.getLogger(__name__)


def ingest_sent_batch(emails, collection, embed: Callable | None = None, understand: Callable = None):
    """Idempotently upsert normalized sent emails by Gmail message ID."""
    if embed is None:
        from rag.retrieve import embed_text
        embed = embed_text
    if understand is None:
        # Invoke the real category tool used by the Understanding Agent.
        # Pipeline B only consumes category, so it doesn't run unrelated
        # LLM-selected summary/task tools for each historical sent message.
        understand = lambda text: {"category": classify_email(text)}
    count_before = collection.count()
    wrote = 0
    for e in emails:
        text = f"Subject: {e.get('subject', '')}\n\n{e.get('body', '')}".strip()
        message_id = str(e["message_id"])
        # Standardized ID format to match scripts/ingest_real_user_emails.py
        chroma_id = f"gmail_sent_{message_id}"
        expected_identity = {
            "sender": str(e.get("sender_email", "")),
            "date": str(e.get("timestamp", "")),
            "thread": str(e.get("thread_id", "")),
            "source": "gmail_sent",
        }
        current = collection.get(ids=[chroma_id], include=["documents", "metadatas"])
        if current.get("ids") and current["documents"][0] == text:
            existing_meta = current["metadatas"][0]
            if all(existing_meta.get(k) == v for k, v in expected_identity.items()) and existing_meta.get("category") in CATEGORIES:
                wrote += 1
                log.info("idempotent-skip id=%s category=%s metadata=%s", chroma_id, existing_meta["category"], existing_meta)
                continue
        # Boundary evidence: normalized fetch -> understanding input.
        log.info("fetch->understanding id=%s shape={message_id,subject,body}->text chars=%d content_preview=%r",
                 e.get("message_id"), len(text), text[:180])
        understanding = understand(text)
        category = understanding["category"]
        if category not in CATEGORIES:
            raise ValueError(f"Understanding Agent classifier emitted {category!r}; taxonomy={CATEGORIES}")
        log.info("understanding->embedding id=%s output=%s document_chars=%d content_preview=%r",
                 e["message_id"], {"category": category, "summary": understanding.get("summary"),
                                   "tasks": understanding.get("tasks")}, len(text), text[:180])
        metadata = {
            "category": category,
            "sender": e.get("sender_email", ""),
            "date": e.get("timestamp", ""),
            "thread": e.get("thread_id", ""),
            "source": "gmail_sent",
        }
        vector = embed(text)
        log.info("embedding->chroma id=%s vector_dim=%d metadata=%s", chroma_id, len(vector), metadata)
        collection.upsert(ids=[chroma_id], documents=[text], embeddings=[vector], metadatas=[metadata])
        saved = collection.get(ids=[chroma_id], include=["metadatas", "documents"])
        actual = saved["metadatas"][0]
        expected = {k: str(v) for k, v in metadata.items()}
        # The shared collection may contain additional sent-email metadata
        # written by the Gmail normalization pipeline. Validate this pipeline's
        # contract without rejecting those unrelated fields.
        mismatched = {k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v}
        if mismatched:
            raise AssertionError(f"Chroma metadata mismatch for {chroma_id}: expected_fields={mismatched!r} got={actual!r}")
        wrote += 1
    return {"fetched": len(emails), "upserted": wrote, "before": count_before, "after": collection.count()}

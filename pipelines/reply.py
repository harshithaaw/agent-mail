"""Pipeline A Reply Agent: category-filtered semantic retrieval + grounded draft."""
import logging
import os
from agents.understanding.categories import CATEGORIES

log = logging.getLogger(__name__)


def retrieve_examples(email_text, category, collection, embed, k=3):
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; expected one of {CATEGORIES}")
    vector = embed(email_text)
    result = collection.query(query_embeddings=[vector], n_results=k, where={"category": category}, include=["documents", "metadatas", "distances"])
    rows = []
    ids = result["ids"][0]
    for i, doc in enumerate(result["documents"][0]):
        rows.append({"id": ids[i], "document": doc, "metadata": result["metadatas"][0][i], "distance": result["distances"][0][i]})
    log.info("reply->chroma category=%s query_chars=%d query_preview=%r count=%d matches=%s", category,
             len(email_text), email_text[:180], len(rows),
             [{"id": x["id"], "category": x["metadata"].get("category"),
               "distance": x["distance"], "document_preview": x["document"][:100]} for x in rows])
    return rows


def generate_grounded_draft(email, examples, generate=None):
    if generate:
        return generate(email, examples)
    from agents.reply.generate import generate_reply
    text = f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
    return generate_reply(text, {"category": examples[0]["metadata"]["category"] if examples else "Unknown"}, examples)


def reply_to_email(email, understanding, collection, embed, generate=None, k=3):
    query_text = f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
    examples = None
    if os.getenv("AGENTMAIL_PERSON_AWARE") == "1":
        sender = email.get("sender_email")
        thread_id = email.get("thread_id")
        if sender or thread_id:
            try:
                from agents.reply.agent import _normalize_address
                from rag.retrieve import retrieve_similar

                examples = retrieve_similar(
                    query_text,
                    k=k,
                    source_filter=["gmail_sent"],
                    category=understanding["category"],
                    person_addr=_normalize_address(sender) or None,
                    thread_id=thread_id or None,
                    collection=collection,
                    embed=embed,
                )
            except Exception:
                log.warning("Person-aware retrieval failed; falling back to category retrieval")
    if examples is None:
        examples = retrieve_examples(query_text, understanding["category"], collection, embed, k)
    sender_name = email.get("sender_name", "")
    email_text = (
        f"Sender: {sender_name} <{email.get('sender_email', '')}>\n"
        f"Subject: {email.get('subject', '')}\n\n{email.get('body', '')}"
    )
    # Boundary evidence is available to callers in result, and logged by app.
    draft = generate_grounded_draft(email, examples, generate) if generate else _generate_with_context(email_text, {**understanding, "sender_name": sender_name, "sender_email": email.get("sender_email", "")}, examples)
    return {"draft": draft, "retrieved": examples, "category": understanding["category"]}


def _generate_with_context(email_text, understanding, examples):
    from agents.reply.generate import generate_reply
    return generate_reply(email_text, understanding, examples)

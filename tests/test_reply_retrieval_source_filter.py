import agents.reply.agent as reply_agent
import rag.retrieve as rag_retrieve


def test_reply_retrieval_uses_gmail_sent_only_and_excludes_enron(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_retrieve, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_retrieve, "_client", None)
    monkeypatch.setattr(rag_retrieve, "_collection", None)

    collection = rag_retrieve.get_or_create_collection()
    text = "Thanks for sending the report. I will review the timeline and reply tomorrow."
    near_text = "Thanks for sending the report. I will review the timeline and reply tomorrow!"
    collection.add(
        ids=["fixture-real-user", "fixture-gmail-sent", "fixture-enron"],
        documents=[text, near_text, text],
        embeddings=[
            rag_retrieve.embed_text(text),
            rag_retrieve.embed_text(near_text),
            rag_retrieve.embed_text(text),
        ],
        metadatas=[
            {"source": "real_user", "type": "sent_example", "category": "Career"},
            {"source": "gmail_sent", "type": "sent_email_reply_example", "category": "Career"},
            {"source": "enron", "type": "sent_example", "category": "Career"},
        ],
    )

    result = reply_agent._retrieve_node({
        "retrieval_attempts": 0,
        "retrieval_k": 3,
        "retrieval_query": text,
        "agent_output": {"category": "Career"},
        "trace": [],
    })

    returned_ids = {item["id"] for item in result["retrieved_examples"]}
    assert "fixture-gmail-sent" in returned_ids
    assert "fixture-real-user" not in returned_ids
    assert "fixture-enron" not in returned_ids
    assert result["trace"][-1]["source_filter"] == ["gmail_sent"]

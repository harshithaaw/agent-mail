# AgentMail pipelines

```text
Pipeline B (scheduled, sent mail)
fetch_emails() [replaceable Gmail edge]
  -> shared category taxonomy + Understanding Agent
  -> all-MiniLM-L6-v2 embeddings
  -> Chroma collection: email_memory (cosine)
     ID = Gmail message_id; upsert; metadata = category, sender, date, thread, source

Pipeline A (unread inbox)
fetch_inbox_emails() [replaceable Gmail edge]
  -> Security Agent: SAFE / REVIEW / BLOCK + route
  -> Understanding Agent (only routes that continue)
  -> category-filtered semantic retrieval from the exact same `email_memory` collection
  -> Reply Agent (existing local Qwen generation module)
  -> drafts.create only, clean_full_pipeline route
```

The two pipelines share Chroma only at retrieval: Pipeline B writes sent-message vectors and metadata; Pipeline A queries those same vectors. Ingestion, scheduling, security, and drafting remain separate.

## Taxonomy drift risk

`agents/understanding/categories.py` is the canonical label set (`Academic`, `Career`, `Personal`, `Promotional`) used by the trained classifier and validated at both Chroma write and query boundaries. Pipeline B calls `classify_email`, the category tool registered in the Understanding Agent; Pipeline A's app runs the full graph. The real classifier can still semantically misclassify a particular message; the fixture report includes assigned labels and category-cluster comparison so that behavior is visible.

## Fixture evidence

Run `python -m pytest -q tests/test_two_pipelines.py` for idempotency, metadata round-trip, retrieval, and route-gating regression checks. Run `python -m scripts.build_pipeline_test_report` for the real-model report: Pipeline B uses the trained `classify_email` tool registered inside the Understanding Agent (the only Understanding output it needs is category), Stage 2/3 use actual `all-MiniLM-L6-v2` embeddings, and Stage 3 uses local Qwen generation. Pipeline A's production app runs the full Understanding Agent graph. The report uses in-memory Chroma and fixture security routes; it does not call Gmail or claim live security-model accuracy.

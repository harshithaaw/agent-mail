# AgentMail

AgentMail is a Gmail assistant with two separate pipelines. Incoming mail passes a security gate and can produce a Gmail draft for human review. Sent mail is ingested as long-term retrieval context. The system does not send, move, or delete messages.

## Architecture

```text
Pipeline A — unread inbox
fetch_inbox_emails() -> Security Agent -> Understanding Agent
  -> Reply Agent / category-aware Chroma retrieval -> Gmail drafts.create

Pipeline B — scheduled sent-mail ingestion
fetch_emails() -> Understanding Agent category -> embedding -> Chroma upsert

Both pipelines share the `email_memory` Chroma collection. Pipeline B writes
sent-message examples; Pipeline A retrieves them for reply grounding.
```

The runnable Pipeline A orchestrator is [app.py](app.py). Its fetch, screening, understanding, and draft dependencies can be injected for fixtures. Pipeline B logic is in [pipelines/ingestion.py](pipelines/ingestion.py), with the callable scheduled-job entrypoint [scripts/ingest_sent_batch.py](scripts/ingest_sent_batch.py). The category labels live in [agents/understanding/categories.py](agents/understanding/categories.py). Architecture details and the category-model drift risk are in [docs/pipelines.md](docs/pipelines.md).

## Fixture tests

```bash
python -m pytest -q tests/test_two_pipelines.py
python -m scripts.build_pipeline_test_report
```

The regression suite uses checked-in sent-message samples, ten inbox fixtures, in-memory Chroma, the trained category classifier, and real MiniLM vectors; it injects the draft writer and does not access Gmail. The report command additionally runs the local Qwen Reply Agent and writes retrieved examples beside each generated draft in [docs/pipeline-test-report.md](docs/pipeline-test-report.md).

## Production entrypoints

- Run the incoming flow by invoking `app.process_inbox()` with its default Gmail/security integrations. Only `clean_full_pipeline` proceeds to draft creation.
- Or run it from the project root with `python -m app`.
- Run `python -m scripts.ingest_sent_batch` to fetch sent mail and idempotently upsert it into Chroma. This command uses Gmail and the persistent Chroma collection.
- Configure Gmail OAuth credentials and local model dependencies before running either production path.

`screening.py` remains as a compatibility adapter for older callers and delegates to `agents.security.agent`; it contains no separate security implementation. The former Gemini function-calling orchestrator, combined legacy runner, Gmail labeling runner, and dashboard-style demo have been removed because they duplicated or contradicted these two scoped pipelines. Gemini orchestration is not part of the current production flow.

## Models

The project uses trained classifiers for spam detection and email categorization. These model files are not committed to git (see `.gitignore`) and must be trained locally before running the full pipeline.

### Required Model Files

- `models/category/category_classifier.pkl` - Trained category classifier (~720 KB)
- `models/category/category_tfidf_vectorizer.pkl` - TF-IDF vectorizer for category classification (~546 KB)
- `models/spam/spam_classifier.pkl` - Trained spam classifier (~1.8 MB)
- `models/spam/tfidf_vectorizer.pkl` - TF-IDF vectorizer for spam detection (~6.4 MB)

### Training Data

Training data is not shipped with the repo. Models are trained on user-provided or public datasets:

- **Category classifier**: Trained on `data/datasets/combined_dataset.csv` (user-supplied labelled CSV)
- **Spam classifier**: Trained on SpamAssassin corpus in `data/spamassassin/` (spam and easy_ham directories)

### Rebuild Commands

To rebuild the models after cloning the repository:

```bash
# Train category classifier
python -m scripts.train_category_classifier

# Train spam classifier
python -m scripts.train_spam_classifier
```

After training, the model files will be created in the `models/` directory and tests that require them will pass.

## Setup with your own data

- **Gmail Account & Credentials**: You need your own Gmail account and OAuth credentials. Credentials are never committed to the repository. Place your OAuth client secrets at `credentials/client_secret.json` (or under `credentials/`); the authorized token will be saved to `token.json`.
- **Local Chroma Vector Database**: The repository contains synthetic fixture data only. No vector database ships with the repo. Chroma is built locally from your own Gmail sent emails by running `python -m scripts.ingest_sent_batch`.
- **Training Data**: Training data is not included in the repository. You must download or supply the datasets yourself:
  - **Category CSV**: `data/datasets/combined_dataset.csv` (CSV format with required columns `text` and `label`; label values `Academic`, `Career`, `Personal`, `Promotional`). Users must supply their own labelled CSV in this format; download from public sources or verify the source and licence yourself.
  - **SpamAssassin**: `data/spamassassin/` (EML/raw email files structured in `data/spamassassin/spam/` and `data/spamassassin/easy_ham/`). Download from public sources yourself; verify the source and licence yourself.
  - **Nazario Phishing**: `nazario_data/phishing3.mbox` (mbox format file used for phishing test evaluation). Download from public sources yourself; verify the source and licence yourself.
- **Model Security**: `.pkl` model files are not committed to git and can execute arbitrary code when loaded. Only load model files you built yourself using `python -m scripts.train_category_classifier` and `python -m scripts.train_spam_classifier`.
- **Chroma Retrieval Settings**: Collection name `email_memory`, embedding model `all-MiniLM-L6-v2`, and `DISTANCE_THRESHOLD = 0.60` (defined in `agents/reply/context.py`).
- **Safety Rules**: The system operates in draft-only mode. It only creates Gmail drafts for human review and NEVER sends, deletes, moves, or labels emails.

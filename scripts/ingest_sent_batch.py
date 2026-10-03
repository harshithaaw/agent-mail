"""Callable scheduled job for Pipeline B; scheduler/UI only needs to call run()."""
from gmail.fetch import fetch_emails
from pipelines.ingestion import ingest_sent_batch
from rag.retrieve import get_or_create_collection


def run(max_emails=100, fetcher=fetch_emails, collection=None):
    batch = fetcher(max_emails=max_emails)
    return ingest_sent_batch(batch, collection or get_or_create_collection())


if __name__ == "__main__":
    print(run())

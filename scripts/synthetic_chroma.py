"""Utilities shared by the synthetic Chroma loader and evaluator."""
from contextlib import contextmanager

import rag.retrieve as retrieve_module


@contextmanager
def chroma_at(path):
    """Temporarily point the existing RAG layer at a selected store."""
    previous_path = retrieve_module.CHROMA_PATH
    previous_client = retrieve_module._client
    previous_collection = retrieve_module._collection
    retrieve_module.CHROMA_PATH = str(path)
    retrieve_module._client = None
    retrieve_module._collection = None
    try:
        yield retrieve_module.get_or_create_collection()
    finally:
        retrieve_module.CHROMA_PATH = previous_path
        retrieve_module._client = previous_client
        retrieve_module._collection = previous_collection

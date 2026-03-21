"""LangChain-compatible embedding instance (Voyage)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_voyageai import VoyageAIEmbeddings

from src.config import VOYAGE_API_KEY, VOYAGE_MODEL


def get_embedding_model() -> Embeddings:
    """Return Voyage embedding model."""
    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY required. Get one at voyageai.com")
    return VoyageAIEmbeddings(voyage_api_key=VOYAGE_API_KEY, model=VOYAGE_MODEL)

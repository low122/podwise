"""LangChain-compatible embedding instance (Ollama default)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from src.config import (
    EMBEDDING_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)


def get_embedding_model() -> Embeddings:
    """Return embedding model from config (Ollama by default)."""
    if EMBEDDING_PROVIDER.lower() != "ollama":
        raise ValueError(
            f"Only 'ollama' is supported for EMBEDDING_PROVIDER; got {EMBEDDING_PROVIDER!r}"
        )
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

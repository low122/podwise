"""Load settings from environment for podwise."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Suppress noisy Pydantic V1 warning from LangChain on Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

load_dotenv()


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser().resolve()


# ChromaDB
CHROMA_PATH: Path = _path("CHROMA_PATH", "./data/chroma")
COLLECTION_NAME: str = _str("COLLECTION_NAME", "podwise")

# Transcripts cache
TRANSCRIPTS_PATH: Path = _path("TRANSCRIPTS_PATH", "./data/transcripts")

# Embedding (Ollama default)
EMBEDDING_PROVIDER: str = _str("EMBEDDING_PROVIDER", "ollama")
OLLAMA_BASE_URL: str = _str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL: str = _str("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# LLM
LLM_PROVIDER: str = _str("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY: str = _str("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = _str("LLM_MODEL", "claude-sonnet-4-20250514")


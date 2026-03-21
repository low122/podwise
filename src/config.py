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


# Supabase (pgvector)
SUPABASE_URL: str = _str("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY: str = _str("SUPABASE_SERVICE_KEY", "")
SUPABASE_TABLE_NAME: str = _str("SUPABASE_TABLE_NAME", "documents")
SUPABASE_QUERY_NAME: str = _str("SUPABASE_QUERY_NAME", "match_documents")

# Transcripts cache
TRANSCRIPTS_PATH: Path = _path("TRANSCRIPTS_PATH", "./data/transcripts")

# Embedding (Voyage)
VOYAGE_API_KEY: str = _str("VOYAGE_API_KEY", "")
VOYAGE_MODEL: str = _str("VOYAGE_MODEL", "voyage-3")  # 1024 dims; voyage-3-lite=512

# LLM
LLM_PROVIDER: str = _str("LLM_PROVIDER", "anthropic")
ANTHROPIC_API_KEY: str = _str("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = _str("LLM_MODEL", "claude-sonnet-4-20250514")


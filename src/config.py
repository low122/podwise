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

SYSTEM_PROMPT = """You are a podcast knowledge assistant. You have access to transcripts from
multiple podcasts and YouTube videos. When answering:
- ALWAYS cite the source: [Episode Title | MM:SS | link]
- If comparing across episodes, organize by speaker/episode
- If asked for notes or summary, structure with headers and bullet points
- If you don't find relevant content, say so honestly
"""

ANTHROPIC_TOOLS = [
    {
        "name": "search_transcripts_tool",
        "description": "Search across all ingested podcast/YouTube transcripts by meaning. Use this to find relevant quotes and segments when answering the user's question. Returns chunks with [Episode Title | Timestamp | YouTube link] and snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query over transcript content"},
                "top_k": {"type": "integer", "description": "Number of results to return (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_episode_list_tool",
        "description": "List all ingested episodes (video_id, title, channel, language). Call this when the user asks what's in the index or which podcasts are available.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_episode_context_tool",
        "description": "Get more context around a specific timestamp in one episode. video_id: YouTube video ID (e.g. rIwgZWzUKm8). timestamp: time in seconds. Use after search when you need surrounding sentences at that moment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "YouTube video ID"},
                "timestamp": {"type": "number", "description": "Time in seconds"},
                "window": {"type": "integer", "description": "Chunks before/after (default 2)"},
            },
            "required": ["video_id", "timestamp"],
        },
    },
]

JWT_SECRET: str = _str("JWT_SECRET", "")
GOOGLE_CLIENT_ID: str = _str("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET: str = _str("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT: str = _str("GOOGLE_REDIRECT", "http://localhost:8000")
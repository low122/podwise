"""Ingestion orchestrator: YouTube URL -> Supabase (pgvector)."""

from __future__ import annotations

from typing import Dict

from src.ingestion.youtube import fetch_transcript
from src.ingestion.cleaner import merge_segments
from src.ingestion.chunker import semantic_chunk_documents
from src.storage.supabase_store import SupabaseStore


def ingest_youtube(url: str) -> Dict[str, object]:
    """End-to-end ingestion for a single YouTube URL.

    Returns small dict with counts and video metadata.
    """
    print(f"Fetching transcript for {url} ...")
    data = fetch_transcript(url)
    metadata = data["metadata"]
    segments = data["segments"]

    print(
        f"Downloaded transcript for '{metadata.title}' "
        f"({metadata.video_id}) — {len(segments)} segments."
    )

    blocks = merge_segments(segments)
    print(f"Merged into {len(blocks)} time-aware blocks. Running semantic chunking...")

    docs = semantic_chunk_documents(
        blocks=blocks,
        video_id=metadata.video_id,
        title=metadata.title,
        channel=metadata.channel,
        language=metadata.language_code,
    )

    store = SupabaseStore()
    store.upsert_documents(docs)

    print(
        f"Ingested '{metadata.title}' — "
        f"{len(segments)} segments -> {len(blocks)} blocks -> {len(docs)} chunks -> stored."
    )

    return {
        "video_id": metadata.video_id,
        "title": metadata.title,
        "channel": metadata.channel,
        "language": metadata.language_code,
        "segments": len(segments),
        "blocks": len(blocks),
        "chunks": len(docs),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.index <youtube_url>")
        raise SystemExit(1)

    ingest_youtube(sys.argv[1])


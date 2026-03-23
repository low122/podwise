"""Ingestion orchestrator: YouTube URL -> Supabase (pgvector)."""

from __future__ import annotations

from typing import Dict, Generator

from src.ingestion.youtube import fetch_transcript
from src.ingestion.cleaner import merge_segments
from src.ingestion.chunker import semantic_chunk_documents
from src.storage.supabase_store import SupabaseStore

TOTAL_STEPS = 5


def ingest_youtube_stream(url: str) -> Generator[Dict[str, object], None, None]:
    """Yields progress dicts for each pipeline stage. Last event has 'result'."""

    yield {"step": 1, "total": TOTAL_STEPS, "status": "Fetching transcript..."}
    data = fetch_transcript(url)
    metadata = data["metadata"]
    segments = data["segments"]

    yield {
        "step": 2, "total": TOTAL_STEPS,
        "status": "Cleaning transcript...",
        "detail": f"'{metadata.title}' - {len(segments)} segments",
    }
    blocks = merge_segments(segments)

    yield {
        "step": 3, "total": TOTAL_STEPS,
        "status": "Chunking into segments...",
        "detail": f"Merged into {len(blocks)} blocks",
    }
    docs = semantic_chunk_documents(
        blocks=blocks,
        video_id=metadata.video_id,
        title=metadata.title,
        channel=metadata.channel,
        language=metadata.language_code,
    )

    yield {
        "step": 4, "total": TOTAL_STEPS,
        "status": "Embedding and storing...",
        "detail": f"{len(docs)} chunks to store",
    }
    store = SupabaseStore()
    store.upsert_documents(docs)

    result = {
        "video_id": metadata.video_id,
        "title": metadata.title,
        "channel": metadata.channel,
        "language": metadata.language_code,
        "segments": len(segments),
        "blocks": len(blocks),
        "chunks": len(docs),
    }
    yield {
        "step": 5, "total": TOTAL_STEPS,
        "status": "Done!",
        "detail": f"Indexed {len(docs)} chunks",
        "result": result,
    }


def ingest_youtube(url: str) -> Dict[str, object]:
    """Blocking version - runs the full pipeline and returns the result."""
    result = None
    for event in ingest_youtube_stream(url):
        print(f"[{event['step']}/{event['total']}] {event['status']}")
        if "result" in event:
            result = event["result"]
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.index <youtube_url>")
        raise SystemExit(1)

    ingest_youtube(sys.argv[1])

